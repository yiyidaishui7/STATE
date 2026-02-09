# 三路 DataLoader 与 MMD Loss 详解

---

## Part 1: 三路并行 DataLoader

### 1.1 设计目标

在 MMD-AAE 中，我们需要**同时**从三个不同细胞系加载数据，确保每个 mini-batch 包含来自所有域的样本。

```
目标: 每个 batch 包含 K562 + RPE1 + Jurkat 各 32 个样本
     总 batch size = 96
```

### 1.2 为什么不能用单个 DataLoader？

**方案 A: 合并所有数据（❌ 不推荐）**
```python
# 问题: 无法保证每个 batch 中三个域的样本数相等
combined_dataset = ConcatDataset([k562, rpe1, jurkat])
loader = DataLoader(combined_dataset, batch_size=96, shuffle=True)

# 可能出现的情况:
# batch_1: 80 个 K562 + 16 个 RPE1 + 0 个 Jurkat  ← 不平衡!
# batch_2: 20 个 K562 + 50 个 RPE1 + 26 个 Jurkat
```

**方案 B: 三路并行加载（✅ 我们的方案）**
```python
# 每个域独立的 DataLoader，然后 zip 在一起
loader_k562 = DataLoader(k562, batch_size=32, ...)
loader_rpe1 = DataLoader(rpe1, batch_size=32, ...)
loader_jurkat = DataLoader(jurkat, batch_size=32, ...)

# 同步迭代
for (batch_k562, batch_rpe1, batch_jurkat) in zip(loader_k562, loader_rpe1, loader_jurkat):
    # 每个域恰好 32 个样本!
```

---

### 1.3 代码实现详解

#### Step 1: 单域 Dataset

```python
class SimpleH5Dataset(Dataset):
    """从 H5 文件读取单个细胞系的数据"""
    
    def __init__(self, h5_path, domain_id=0, domain_name="unknown"):
        self.h5_path = h5_path
        self.domain_id = domain_id      # 域标签: 0=K562, 1=RPE1, 2=Jurkat
        self.domain_name = domain_name
        
        # 打开 H5 文件获取形状
        with h5py.File(h5_path, 'r') as f:
            self.shape = f['X'].shape        # (num_cells, num_genes)
            self.num_cells = self.shape[0]   # 例如 18465
            self.num_genes = self.shape[1]   # 18080
        
        self._file = None  # 延迟打开，避免 pickle 问题
    
    def __len__(self):
        return self.num_cells
    
    def __getitem__(self, idx):
        # 读取第 idx 个细胞的基因表达
        f = self._get_file()
        counts = torch.tensor(f['X'][idx], dtype=torch.float32)
        # counts.shape = (18080,)
        
        return counts, self.domain_id
        # 返回: (基因表达向量, 域标签)
```

**数据示例**:
```python
dataset = SimpleH5Dataset("k562.h5", domain_id=0, domain_name="K562")

sample = dataset[0]
# sample[0].shape = torch.Size([18080])  # 基因表达
# sample[1] = 0                          # 域标签
```

---

#### Step 2: Collate 函数

```python
def collate_fn(batch):
    """
    将多个样本整理成一个 batch
    
    输入: [(counts_1, domain_1), (counts_2, domain_2), ...]
    输出: (stacked_counts, stacked_domains)
    """
    # batch 是一个 list，每个元素是 (counts, domain_id)
    counts = torch.stack([item[0] for item in batch])   # (32, 18080)
    domains = torch.tensor([item[1] for item in batch]) # (32,)
    
    return counts, domains
```

**处理过程**:
```python
# 假设 batch_size = 32
batch = [
    (tensor([0.1, 0.2, ...]), 0),  # 细胞 1
    (tensor([0.3, 0.1, ...]), 0),  # 细胞 2
    ...
    (tensor([0.2, 0.4, ...]), 0),  # 细胞 32
]

# Collate 后
counts.shape = (32, 18080)
domains = tensor([0, 0, 0, ..., 0])  # 32 个 0 (K562)
```

---

#### Step 3: 创建三个独立的 DataLoader

```python
# 配置
DOMAIN_CONFIGS = [
    {"name": "K562",   "path": "k562.h5",   "domain_id": 0},
    {"name": "RPE1",   "path": "rpe1.h5",   "domain_id": 1},
    {"name": "Jurkat", "path": "jurkat.h5", "domain_id": 2},
]

loaders = []
for i, domain in enumerate(DOMAIN_CONFIGS):
    # 创建 Dataset
    dataset = SimpleH5Dataset(
        domain["path"], 
        domain_id=i, 
        domain_name=domain["name"]
    )
    
    # 创建 DataLoader
    loader = DataLoader(
        dataset,
        batch_size=32,           # 每个域 32 个样本
        shuffle=True,            # 随机打乱
        collate_fn=collate_fn,
        num_workers=2,           # 多进程加载
        drop_last=True,          # ⚠️ 重要! 丢弃最后不完整的 batch
        pin_memory=True,         # GPU 加速
    )
    loaders.append(loader)

# 结果:
# loaders[0]: K562 DataLoader,  577 个 batch
# loaders[1]: RPE1 DataLoader,  697 个 batch  
# loaders[2]: Jurkat DataLoader, 669 个 batch
```

**为什么需要 `drop_last=True`?**
```python
# 假设 K562 有 18465 个细胞，batch_size=32
# 18465 / 32 = 577.03
# 最后一个 batch 只有 18465 - 577*32 = 1 个样本

# 如果不 drop_last:
# batch_577 = (32 个样本, 32 个样本, 1 个样本) ← K562 只有 1 个!
# zip 会在最短迭代器结束时停止，但 batch 内部不平衡

# 使用 drop_last=True:
# 每个 batch 都保证有完整的 32 个样本
```

---

#### Step 4: ParallelZipLoader 封装

```python
class ParallelZipLoader:
    """
    将三个 DataLoader 封装成一个统一接口
    同步迭代，每次返回三个域的 batch
    """
    
    def __init__(self, loaders, domain_names):
        self.loaders = loaders           # [loader_k562, loader_rpe1, loader_jurkat]
        self.domain_names = domain_names # ["K562", "RPE1", "Jurkat"]
    
    def __iter__(self):
        # 使用 zip 同步迭代三个 loader
        return zip(*self.loaders)
        # 每次返回: (batch_k562, batch_rpe1, batch_jurkat)
    
    def __len__(self):
        # 长度以最短的 loader 为准 (木桶效应)
        return min(len(l) for l in self.loaders)
        # min(577, 697, 669) = 577
    
    @property
    def batch_size(self):
        return sum(l.batch_size for l in self.loaders)
        # 32 + 32 + 32 = 96
```

**迭代过程可视化**:
```python
parallel_loader = ParallelZipLoader(loaders, domain_names)

for batch_idx, domain_batches in enumerate(parallel_loader):
    # domain_batches 是一个 tuple:
    # (
    #   (counts_k562, domains_k562),   # K562:  (32, 18080), (32,)
    #   (counts_rpe1, domains_rpe1),   # RPE1:  (32, 18080), (32,)
    #   (counts_jurkat, domains_jurkat) # Jurkat: (32, 18080), (32,)
    # )
    
    # 合并成一个大 batch
    all_counts = []
    all_domains = []
    for domain_id, (counts, domains) in enumerate(domain_batches):
        all_counts.append(counts)
        all_domains.append(torch.full((counts.size(0),), domain_id))
    
    x = torch.cat(all_counts, dim=0)           # (96, 18080)
    domain_labels = torch.cat(all_domains, dim=0)  # (96,) = [0,0,...,1,1,...,2,2,...]
```

---

### 1.4 数据流可视化

```
                    ┌─────────────────────────────────────────┐
                    │              训练开始                    │
                    └─────────────────────────────────────────┘
                                        │
          ┌─────────────────────────────┼─────────────────────────────┐
          │                             │                             │
          ▼                             ▼                             ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │   k562.h5     │          │   rpe1.h5     │          │  jurkat.h5    │
    │ 18,465 cells  │          │ 22,317 cells  │          │ 21,412 cells  │
    └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
            │                          │                          │
            ▼                          ▼                          ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │SimpleH5Dataset│          │SimpleH5Dataset│          │SimpleH5Dataset│
    │  domain_id=0  │          │  domain_id=1  │          │  domain_id=2  │
    └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
            │                          │                          │
            ▼                          ▼                          ▼
    ┌───────────────┐          ┌───────────────┐          ┌───────────────┐
    │  DataLoader   │          │  DataLoader   │          │  DataLoader   │
    │  batch_size=32│          │  batch_size=32│          │  batch_size=32│
    │  577 batches  │          │  697 batches  │          │  669 batches  │
    └───────┬───────┘          └───────┬───────┘          └───────┬───────┘
            │                          │                          │
            └──────────────────────────┼──────────────────────────┘
                                       │
                                       ▼
                          ┌─────────────────────────┐
                          │   ParallelZipLoader     │
                          │   zip(L0, L1, L2)       │
                          │   总 batch_size = 96    │
                          │   577 iterations/epoch  │
                          └───────────┬─────────────┘
                                      │
                                      ▼
                          ┌─────────────────────────┐
                          │   每个 iteration:       │
                          │   x.shape = (96, 18080) │
                          │   domains = [0]*32 +    │
                          │             [1]*32 +    │
                          │             [2]*32      │
                          └─────────────────────────┘
```

---

## Part 2: MMD Loss 详解

### 2.1 MMD 的直觉理解

**问题**: 如何衡量两个分布 P 和 Q 是否相同？

**思路**: 如果两个分布相同，那么从它们中采样的任意函数的期望值应该相等。

```
如果 P = Q，那么对于任意函数 f:
    E_P[f(x)] = E_Q[f(y)]
```

**MMD 的定义**:
```
MMD(P, Q) = sup_{||f||_H ≤ 1} |E_P[f(x)] - E_Q[f(y)]|
```
- 在所有满足 ||f||_H ≤ 1 的函数中，找出能最大化两个分布期望差异的那个
- 如果这个最大差异为 0，说明两个分布相同

### 2.2 核技巧 (Kernel Trick)

直接优化上面的公式很难，但可以用核函数来简化：

```
MMD²(P, Q) = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
```

其中 K(a, b) 是核函数，我们使用 RBF (高斯) 核：

```
K(a, b) = exp(-γ × ||a - b||²)

γ = 1 / (2σ²)
```

**直觉理解**:
- K(a, b) 衡量 a 和 b 的相似度
- 如果 a 和 b 很近，K(a, b) ≈ 1
- 如果 a 和 b 很远，K(a, b) ≈ 0

### 2.3 MMD² 公式拆解

```
MMD²(P, Q) = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
             ─────────   ─────────   ──────────
                 A           B           C

A: P 分布内部的平均相似度
B: Q 分布内部的平均相似度  
C: P 和 Q 之间的平均相似度
```

**直觉**:
- 如果 P = Q：A = B = C，所以 MMD² = 0
- 如果 P ≠ Q：C 会变小（跨分布相似度降低），MMD² > 0

### 2.4 代码逐行讲解

```python
def compute_mmd_multi_kernel(x, y):
    """
    计算 x (来自域 P) 和 y (来自域 Q) 之间的 MMD
    
    参数:
        x: shape (N, D) - N 个样本，D 维特征 (例如 32 个 K562 细胞，512 维隐向量)
        y: shape (M, D) - M 个样本，D 维特征 (例如 32 个 RPE1 细胞，512 维隐向量)
    
    返回:
        MMD² 值 (标量)
    """
    
    # ================================================
    # Step 1: 使用多个 sigma 值
    # ================================================
    # 不同的 sigma 捕捉不同尺度的分布差异
    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    # σ = 0.01: 对非常局部的差异敏感 (高频信号)
    # σ = 100:  对全局差异敏感 (低频信号)
    # → 综合使用可以捕捉各种尺度的分布差异
    
    # ================================================
    # Step 2: 计算点积矩阵
    # ================================================
    xx = torch.mm(x, x.t())  # (N, N): x_i · x_j
    yy = torch.mm(y, y.t())  # (M, M): y_i · y_j
    xy = torch.mm(x, y.t())  # (N, M): x_i · y_j
    
    # 例如 N=M=32, D=512:
    # xx.shape = (32, 32)
    # yy.shape = (32, 32)
    # xy.shape = (32, 32)
    
    # ================================================
    # Step 3: 计算平方欧氏距离矩阵
    # ================================================
    # ||a - b||² = ||a||² + ||b||² - 2(a·b)
    #            = a·a + b·b - 2(a·b)
    
    # 提取对角线 = 每个向量的平方模
    rx = xx.diag().unsqueeze(0).expand_as(xx)  # (N, N): 每行都是 [||x₁||², ||x₂||², ...]
    ry = yy.diag().unsqueeze(0).expand_as(yy)  # (M, M): 每行都是 [||y₁||², ||y₂||², ...]
    
    # 计算距离矩阵
    dxx = rx.t() + rx - 2. * xx  # (N, N): ||x_i - x_j||²
    dyy = ry.t() + ry - 2. * yy  # (M, M): ||y_i - y_j||²
    
    # 对于跨分布距离，需要特殊处理
    dxy = (rx.t().expand(x.size(0), y.size(0)) + 
           ry.expand(x.size(0), y.size(0)) - 
           2. * xy)  # (N, M): ||x_i - y_j||²
    
    # ================================================
    # Step 4: 初始化 MMD (保持梯度连接!)
    # ================================================
    mmd = x.new_zeros(1)  # 使用 x.new_zeros 而不是 torch.tensor
    # 这样保证 mmd 和 x 在同一设备，且保持计算图连接
    
    # ================================================
    # Step 5: 对每个 sigma 计算并累加
    # ================================================
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2)
        
        # 计算 RBF 核矩阵
        XX = torch.exp(-gamma * dxx)  # K(x_i, x_j)
        YY = torch.exp(-gamma * dyy)  # K(y_i, y_j)
        XY = torch.exp(-gamma * dxy)  # K(x_i, y_j)
        
        # MMD² = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
        mmd = mmd + XX.mean() + YY.mean() - 2. * XY.mean()
    
    # ================================================
    # Step 6: 取平均
    # ================================================
    return mmd / len(sigmas)
```

### 2.5 数值示例

假设我们有简单的 2D 数据：

```python
# 两个域的样本
z_k562 = torch.tensor([
    [1.0, 1.0],
    [1.2, 0.8],
    [0.9, 1.1],
])  # 3 个 K562 样本

z_rpe1 = torch.tensor([
    [3.0, 3.0],
    [3.2, 2.8],
    [2.9, 3.1],
])  # 3 个 RPE1 样本 (明显不同的分布)

# 计算 MMD
# 由于两个分布中心相距较远，MMD 应该较大

# 如果训练成功，两个分布应该变得接近：
z_k562_aligned = torch.tensor([
    [2.0, 2.0],
    [2.1, 1.9],
])
z_rpe1_aligned = torch.tensor([
    [2.0, 2.1],
    [1.9, 2.0],
])
# 此时 MMD ≈ 0
```

### 2.6 在训练中的使用

```python
def compute_loss(self, x, domain_labels, ...):
    # 前向传播得到隐空间表示
    x_recon, z, _ = self.forward(x)
    # z.shape = (96, 512)
    # domain_labels = [0]*32 + [1]*32 + [2]*32
    
    # ========================================
    # 计算 MMD 损失
    # ========================================
    mmd_losses = []
    
    # 遍历所有域对: (0,1), (0,2), (1,2)
    for i in range(3):
        for j in range(i + 1, 3):
            # 获取域 i 的样本
            mask_i = domain_labels == i
            z_i = z[mask_i]  # (32, 512)
            
            # 获取域 j 的样本
            mask_j = domain_labels == j
            z_j = z[mask_j]  # (32, 512)
            
            # 计算这对域之间的 MMD
            mmd_ij = compute_mmd_multi_kernel(z_i, z_j)
            mmd_losses.append(mmd_ij)
    
    # mmd_losses = [
    #   MMD(K562, RPE1),
    #   MMD(K562, Jurkat),
    #   MMD(RPE1, Jurkat)
    # ]
    
    # 取平均
    mmd_loss = torch.stack(mmd_losses).mean()
```

### 2.7 MMD 损失的梯度流

```
                    L_mmd (MMD²)
                        │
                        ▼
            ∂L/∂K = ∂(XX.mean + YY.mean - 2*XY.mean)/∂K
                        │
                        ▼
            ∂K/∂d = ∂exp(-γd)/∂d = -γ × exp(-γd)
                        │
                        ▼
            ∂d/∂z = ∂||z_i - z_j||²/∂z = 2(z_i - z_j)
                        │
                        ▼
                    ∂L/∂z (隐空间表示的梯度)
                        │
                        ▼
                    Encoder 参数更新
                        │
                        ▼
              Encoder 学习生成"域间相似"的表示
```

### 2.8 为什么之前 MMD 不变？

**问题代码**:
```python
mmd_loss = torch.tensor(0.0, device=x.device)  # ❌ 梯度断开!
for ...:
    mmd_loss = mmd_loss + compute_mmd(...)
```

`torch.tensor(0.0)` 创建的是一个没有 `requires_grad` 的新 tensor，导致梯度无法传播。

**修复代码**:
```python
mmd = x.new_zeros(1)  # ✅ 与 x 共享计算图
# 或
mmd_losses = []
for ...:
    mmd_losses.append(compute_mmd(...))
mmd_loss = torch.stack(mmd_losses).mean()  # ✅ stack 保持梯度
```

---

## 总结

| 组件 | 作用 | 关键点 |
|------|------|--------|
| **三路 DataLoader** | 确保域平衡 | `drop_last=True`, `zip` 同步迭代 |
| **ParallelZipLoader** | 统一接口 | 每次返回 (batch_K562, batch_RPE1, batch_Jurkat) |
| **MMD Loss** | 域对齐 | 多核 RBF，正确的梯度连接 |

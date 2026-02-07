# MMD-AAE 三种损失函数详解

本文档详细讲解 `train_mmd_aae.py` 中三种损失函数的计算方式。

---

## 📐 整体损失公式

```python
L_total = λ₁ × L_recon + λ₂ × L_mmd + λ₃ × L_adv
```

| 参数 | 默认值 | 含义 |
|------|--------|------|
| λ₁ (weight_recon) | 1.0 | 重建损失权重 |
| λ₂ (weight_mmd) | 1.0 | MMD 损失权重 |
| λ₃ (weight_adv) | 0.1 | 对抗损失权重 |

---

## 1️⃣ 重建损失 (Reconstruction Loss)

### 目的
确保 Encoder 学到的隐空间表示 `z` 能够保留输入 `x` 的信息，不会丢失太多细节。

### 公式

```
L_recon = MSE(x, x̂) = (1/N) × Σᵢ ||xᵢ - x̂ᵢ||²
```

### 代码实现

```python
def compute_loss(self, x, domain_labels, ...):
    # 前向传播: x → Encoder → z → Decoder → x_recon
    x_recon, z, _ = self.forward(x)
    
    # 1. 重建损失 = MSE(原始输入, 重建输出)
    recon_loss = nn.functional.mse_loss(x_recon, x)
```

### 逐步解析

```python
# 输入
x.shape = (96, 18080)  # 96 个细胞，18080 个基因

# Step 1: 编码
z = self.encoder(x)
z.shape = (96, 512)    # 压缩到 512 维隐空间

# Step 2: 解码/重建
x_recon = self.decoder(z)
x_recon.shape = (96, 18080)  # 重建回 18080 维

# Step 3: 计算 MSE
# 对每个元素计算 (x - x_recon)²，然后取平均
recon_loss = ((x - x_recon) ** 2).mean()
# 输出: 一个标量，例如 0.1320
```

### 梯度流

```
L_recon
   ↓ (∂L/∂x_recon)
Decoder 参数更新
   ↓ (∂L/∂z)
Encoder 参数更新
```

---

## 2️⃣ MMD 损失 (Maximum Mean Discrepancy)

### 目的
最小化不同域（K562, RPE1, Jurkat）在隐空间中的分布差异，实现**域对齐**。

### 核心思想

MMD 测量两个分布 P 和 Q 之间的距离：
- 如果两个分布完全相同 → MMD = 0
- 分布差异越大 → MMD 越大

### 公式

```
MMD²(P, Q) = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
```

其中 K 是核函数（我们使用 RBF/高斯核）：

```
K(x, y) = exp(-γ × ||x - y||²)
γ = 1 / (2σ²)
```

### 代码实现

```python
def compute_mmd_multi_kernel(x, y):
    """
    使用多核 MMD (多个 sigma 值的 RBF 核)
    """
    # 多个尺度的 sigma，捕捉不同尺度的分布差异
    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    # 计算点积矩阵
    xx = torch.mm(x, x.t())  # (N, N)
    yy = torch.mm(y, y.t())  # (M, M)
    xy = torch.mm(x, y.t())  # (N, M)
    
    # 计算平方欧氏距离
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    
    dxx = rx.t() + rx - 2. * xx  # ||xi - xj||²
    dyy = ry.t() + ry - 2. * yy  # ||yi - yj||²
    dxy = rx.t().expand(x.size(0), y.size(0)) + ry.expand(x.size(0), y.size(0)) - 2. * xy
    
    # 初始化 MMD（保持梯度连接！）
    mmd = x.new_zeros(1)
    
    # 对每个 sigma 计算 RBF 核 MMD
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2)
        XX = torch.exp(-gamma * dxx)  # K(xi, xj)
        YY = torch.exp(-gamma * dyy)  # K(yi, yj)
        XY = torch.exp(-gamma * dxy)  # K(xi, yj)
        mmd = mmd + XX.mean() + YY.mean() - 2. * XY.mean()
    
    return mmd / len(sigmas)
```

### 逐步解析

假设我们有:
- `z_k562`: K562 细胞的隐空间表示，shape = (32, 512)
- `z_rpe1`: RPE1 细胞的隐空间表示，shape = (32, 512)

```python
# Step 1: 计算点积矩阵
xx = torch.mm(z_k562, z_k562.t())  # (32, 32) - K562 vs K562
yy = torch.mm(z_rpe1, z_rpe1.t())  # (32, 32) - RPE1 vs RPE1
xy = torch.mm(z_k562, z_rpe1.t())  # (32, 32) - K562 vs RPE1

# Step 2: 计算平方距离
# ||xi - xj||² = ||xi||² + ||xj||² - 2 * xi·xj
# 
# xx.diag() 提取对角线 = [||x₁||², ||x₂||², ..., ||x₃₂||²]
# 
# rx = [||x₁||², ||x₁||², ..., ||x₁||²]    <- 复制到每一列
#      [||x₂||², ||x₂||², ..., ||x₂||²]
#      ...
#      [||x₃₂||², ||x₃₂||², ..., ||x₃₂||²]
#
# rx.t() + rx - 2*xx = ||xi - xj||² 矩阵

# Step 3: 计算 RBF 核
# gamma = 1 / (2 * sigma²)
# K(xi, xj) = exp(-gamma * ||xi - xj||²)

# 例如 sigma = 1.0:
gamma = 0.5
XX = torch.exp(-0.5 * dxx)  # 核矩阵 (32, 32)
YY = torch.exp(-0.5 * dyy)  # 核矩阵 (32, 32)
XY = torch.exp(-0.5 * dxy)  # 核矩阵 (32, 32)

# Step 4: 计算 MMD
# MMD² = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
#      = XX.mean() + YY.mean() - 2 * XY.mean()
mmd = XX.mean() + YY.mean() - 2 * XY.mean()
```

### 在 compute_loss 中的调用

```python
def compute_loss(self, x, domain_labels, ...):
    ...
    
    # 2. MMD 损失
    mmd_losses = []
    
    # 遍历所有域对: (0,1), (0,2), (1,2)
    for i in range(3):           # i = 0, 1, 2
        for j in range(i + 1, 3):  # j > i
            # 获取域 i 和域 j 的样本
            mask_i = domain_labels == i  # K562: [True, True, ..., False, False, ...]
            mask_j = domain_labels == j  # RPE1: [False, False, ..., True, True, ...]
            
            if mask_i.sum() > 1 and mask_j.sum() > 1:
                # 计算这对域之间的 MMD
                mmd_ij = compute_mmd_multi_kernel(z[mask_i], z[mask_j])
                mmd_losses.append(mmd_ij)
    
    # 取所有域对 MMD 的平均
    mmd_loss = torch.stack(mmd_losses).mean()
```

### 为什么使用多核?

```
不同的 sigma 捕捉不同尺度的分布差异：

σ = 0.01 (很小): 对局部差异敏感，检测细粒度分布差异
σ = 0.1
σ = 1.0  (中等): 平衡局部和全局
σ = 10.0
σ = 100  (很大): 对全局差异敏感，检测粗粒度分布shift
```

### 梯度流

```
L_mmd
   ↓ (∂L/∂K)
   ↓ (∂K/∂distance)
   ↓ (∂distance/∂z)
Encoder 参数更新
```

---

## 3️⃣ 对抗损失 (Adversarial Loss)

### 目的
通过对抗训练，让 Encoder 学习**域不变特征**——即使 Discriminator 也无法从隐空间表示 `z` 中判断样本来自哪个域。

### 核心思想

```
┌─────────────────────────────────────────────────────────────┐
│                     对抗博弈                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  Discriminator 目标: 正确分类域标签 (minimize CE loss)       │
│                                                             │
│  Encoder 目标: 欺骗 Discriminator，让它分不出来              │
│              (maximize CE loss = minimize -CE loss)          │
│                                                             │
│  → 使用梯度反转层 (GRL) 实现一体化训练                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 梯度反转层 (Gradient Reversal Layer)

```python
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)  # 前向传播不变
    
    @staticmethod
    def backward(ctx, grad_output):
        # 反向传播时反转梯度！
        return -ctx.alpha * grad_output, None
```

**工作原理**:

```
前向: z → GRL → z (不变)
反向: ∂L/∂z → GRL → -α × ∂L/∂z (反转!)

效果:
- Discriminator 收到正常梯度 → 学习分类
- Encoder 收到反转梯度 → 学习欺骗 Discriminator
```

### 代码实现

```python
def compute_loss(self, x, domain_labels, ...):
    ...
    
    # 3. 对抗损失
    
    # Step 1: 应用梯度反转
    z_grl = GradientReversalFunction.apply(z, grl_alpha)
    # z_grl 的值 = z（前向一样）
    # 但反向传播时梯度会被反转
    
    # Step 2: Discriminator 预测域标签
    domain_logits = self.discriminator(z_grl)
    # domain_logits.shape = (96, 3)  # 3 个类别的 logits
    
    # Step 3: 计算交叉熵损失
    adv_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
    # domain_labels = [0, 0, ..., 1, 1, ..., 2, 2, ...]
    #                  K562      RPE1       Jurkat
```

### 逐步解析

```python
# 输入
z.shape = (96, 512)              # 隐空间表示
domain_labels = [0]*32 + [1]*32 + [2]*32  # 域标签

# Step 1: 梯度反转层
z_grl = GradientReversalFunction.apply(z, alpha=1.0)
# 前向: z_grl 的值和 z 完全一样
# 反向: 梯度会被乘以 -1

# Step 2: Discriminator 网络
# Linear(512 → 256) → ReLU → Linear(256 → 256) → ReLU → Linear(256 → 3)
domain_logits = self.discriminator(z_grl)
domain_logits.shape = (96, 3)

# 例如一个样本的输出:
# domain_logits[0] = [2.1, -0.5, 0.3]
#                     K562  RPE1  Jurkat
# 这表示 Discriminator 认为这个样本最可能来自 K562

# Step 3: Cross-Entropy Loss
# CE = -log(softmax(logits)[correct_class])
#
# 对于 domain_labels[0] = 0 (K562):
# softmax([2.1, -0.5, 0.3]) = [0.72, 0.05, 0.12]
# CE = -log(0.72) = 0.33
#
adv_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
# 输出: 标量，例如 1.0744
```

### 对抗训练的梯度流

```
                        L_adv (CE Loss)
                            │
                            ▼
              ┌─────────────────────────────┐
              │      Discriminator          │
              │   (收到正常梯度 ∂L/∂θ_D)     │
              │   → 学习正确分类域          │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │  Gradient Reversal Layer    │
              │   (梯度反转: -α × ∂L/∂z)    │
              └─────────────────────────────┘
                            │
                            ▼
              ┌─────────────────────────────┐
              │         Encoder             │
              │  (收到反转梯度 -∂L/∂θ_E)    │
              │  → 学习欺骗 Discriminator   │
              └─────────────────────────────┘
```

### α (grl_alpha) 的作用

```python
grl_alpha = 1.0  # 默认值

# α 控制对抗训练的强度:
# α = 0: 完全不反转梯度 → Encoder 不受对抗训练影响
# α = 1: 完全反转梯度 → 正常对抗训练
# α > 1: 更强的对抗训练
# α < 1: 更弱的对抗训练

# 常见策略: 训练初期 α 较小，后期逐渐增大
# 例如: α = 2 / (1 + exp(-10 * progress)) - 1
```

---

## 📊 总损失计算

```python
def compute_loss(self, x, domain_labels, 
                 weight_recon=1.0, weight_mmd=1.0, weight_adv=0.1, 
                 grl_alpha=1.0):
    
    # 1. 前向传播
    x_recon, z, _ = self.forward(x)
    
    # 2. 计算三个损失
    recon_loss = MSE(x, x_recon)           # 重建
    mmd_loss = Σ MMD(z_domain_i, z_domain_j)  # 域对齐
    adv_loss = CE(Discriminator(GRL(z)), labels)  # 对抗
    
    # 3. 加权求和
    total_loss = 1.0 * recon_loss + 1.0 * mmd_loss + 0.1 * adv_loss
    
    return total_loss
```

---

## 🎯 三种损失的协同作用

| 损失 | 作用 | 对 Encoder 的影响 |
|------|------|------------------|
| **L_recon** | 保持信息不丢失 | 学习保留输入信息的表示 |
| **L_mmd** | 显式域对齐 | 学习域间分布相似的表示 |
| **L_adv** | 隐式域不变 | 学习让 Discriminator 无法区分的表示 |

**理想结果**:
- Encoder 输出的 `z` 能够重建输入 (L_recon ↓)
- 不同域的 `z` 分布接近 (L_mmd ↓)
- Discriminator 无法区分域 (L_adv → log(3) ≈ 1.1，即随机猜测)

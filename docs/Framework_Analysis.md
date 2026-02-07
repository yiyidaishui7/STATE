# MMD-AAE Framework 架构分析

## 📊 整体架构总览

你的 Framework 是一个 **多分支多目标训练架构**，包含三个核心分支：

```
                    ┌─────────────────────────────────────────┐
                    │        Multi-Domain Data Loading        │
                    │   K562 + RPE1 + Jurkat → Zip/Concat    │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────┐
                              │   Encoder Q     │
                              │   θ_Q 参数      │
                              └────────┬────────┘
                                       │
                                       ▼
                            ┌─────────────────────┐
                            │   Hidden Code H     │
                            │     H = Q(X)        │
                            │   (隐空间表示)       │
                            └─────────┬───────────┘
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
        1. 还原流              2. 对齐流              3. 对抗流
                 │                    │                    │
                 ▼                    ▼                    ▼
        ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
        │  Decoder P    │    │   MMD Loss    │    │Discriminator D│
        │  重建输入      │    │   域对齐       │    │  对抗训练      │
        └───────┬───────┘    └───────┬───────┘    └───────┬───────┘
                │                    │                    │
                ▼                    ▼                    ▼
        ┌───────────────┐    ┌───────────────┐    ┌───────────────┐
        │   L_recon     │    │    R_mmd      │    │    J_gan      │
        │   重建损失     │    │   MMD损失     │    │   对抗损失     │
        └───────────────┘    └───────────────┘    └───────────────┘
```

---

## 🔍 各模块详细分析

### 1️⃣ 数据加载层 (Multi-Domain Data Loading)

```
┌────────────┐  ┌────────────┐  ┌────────────┐
│   K562     │  │   RPE1     │  │   Jurkat   │
│ DataLoader │  │ DataLoader │  │ DataLoader │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │
      └───────────────┼───────────────┘
                      │
                      ▼
              ┌───────────────┐
              │  Zip/Concat   │
              │   混合拼接     │
              └───────┬───────┘
                      │
                      ▼
        ┌──────────────────────────┐
        │     Input Layer          │
        │  Features X + Labels Y   │
        │  (96 samples per batch)  │
        └──────────────────────────┘
```

**实现状态**: ✅ 已实现 (`ParallelZipLoader` in `train_mmd_aae.py`)

**关键设计**:
- 每个域独立的 DataLoader，保证 batch 内域分布均衡
- `drop_last=True` 确保每个域样本数相同
- 支持 shuffle 以增加训练多样性

---

### 2️⃣ 编码器 (Encoder Q)

**功能**: 将高维基因表达 X ∈ R^18080 映射到低维隐空间 H ∈ R^512

**架构设计**:
```
Input: X (18080 genes)
    │
    ▼
Linear(18080 → 1024) + BatchNorm + ReLU + Dropout
    │
    ▼
Linear(1024 → 512) + BatchNorm + ReLU + Dropout
    │
    ▼
Linear(512 → 512)
    │
    ▼
Output: H (512-dim latent code)
```

**实现状态**: ✅ 已实现 (`Encoder` class in `mmd_aae.py`)

---

### 3️⃣ Branch A: 还原流 (Reconstruction Flow)

**目的**: 保证隐空间 H 保留输入信息，避免信息丢失

```
H = Q(X)
    │
    ▼
┌───────────────┐
│  Decoder P    │
│  θ_P 参数     │
└───────┬───────┘
        │
        ▼
    X' = P(H)
        │
        ▼
L_recon = MSE(X, X')
```

**损失函数**:
```python
L_recon = ||X - P(Q(X))||²  # MSE Loss
```

**实现状态**: ✅ 已实现 (`Decoder` class + `recon_loss`)

---

### 4️⃣ Branch B: 对齐流 (Alignment Flow via MMD)

**目的**: 最小化不同域在隐空间中的分布差异

```
      H = Q(X)
          │
          ▼
┌─────────────────────┐
│   Domain Splitter   │◀─── Label Y (域标签)
│   按标签切分         │
└─────────┬───────────┘
    ┌─────┴─────┐─────────┐
    │           │         │
    ▼           ▼         ▼
H_K562     H_RPE1     H_Jurkat
    │           │         │
    └───────────┼─────────┘
          两两配对
                │
                ▼
    ┌─────────────────────┐
    │  Multi-Kernel Calc  │
    │   多核距离计算       │
    └─────────┬───────────┘
              │
              ▼
        R_mmd: MMD Loss
```

**MMD 计算**:
```python
R_mmd = MMD(H_K562, H_RPE1) + MMD(H_RPE1, H_Jurkat) + MMD(H_K562, H_Jurkat)
      = Σ_{i<j} MMD(H_i, H_j)
```

**RBF 核函数**:
```python
K(x, y) = exp(-||x - y||² / (2σ²))
MMD² = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
```

**实现状态**: ✅ 已实现 (`compute_mmd_rbf` function)

---

### 5️⃣ Branch C: 对抗流 (Adversarial Flow)

**目的**: 
1. 让隐空间分布接近先验分布 p(z) (Laplace/Gaussian)
2. 进一步增强域不变性

```
       隐变量 H (Fake Sample)
              │
              │ 生成数据
              ▼
    ┌─────────────────────────┐
    │     Prior p(z)          │
    │   Laplace Distribution  │
    │         采样 ↓           │
    │   h_prior (Real Sample) │
    └─────────────────────────┘
              │
              │ 真实数据
              ▼
    ┌─────────────────────┐
    │   Discriminator D   │◀─── H (隐变量)
    │   判别器             │
    └─────────┬───────────┘
              │
              ▼
        J_gan: Adversarial Loss
```

**对抗训练**:
- **Discriminator**: 区分 H (Encoder 输出) vs h_prior (先验采样)
- **Encoder**: 让 H 的分布接近先验 p(z)，欺骗 Discriminator

**GAN Loss**:
```python
# Discriminator 损失
L_D = -E[log D(h_prior)] - E[log(1 - D(H))]

# Encoder 损失 (通过梯度反转)
J_gan = -E[log D(H)]
```

**实现状态**: ⚠️ 部分实现 
- 已有 `DomainDiscriminator`（域判别）
- 缺少 Prior Distribution 采样的对抗训练

---

## 📐 总损失函数

```python
L_total = λ₁ × L_recon + λ₂ × R_mmd + λ₃ × J_gan

# 当前默认权重:
# λ₁ = 1.0  (重建)
# λ₂ = 0.5  (MMD)
# λ₃ = 0.1  (对抗)
```

---

## 🔧 与当前代码的对应关系

| 架构组件 | 代码位置 | 实现状态 |
|---------|---------|---------|
| Multi-Domain DataLoader | `train_mmd_aae.py` → `ParallelZipLoader` | ✅ 完成 |
| Encoder Q | `mmd_aae.py` → `Encoder` class | ✅ 完成 |
| Decoder P | `mmd_aae.py` → `Decoder` class | ✅ 完成 |
| Domain Splitter | `compute_loss()` → mask by domain | ✅ 完成 |
| MMD Loss | `compute_mmd_rbf()` | ✅ 完成 |
| Discriminator D (域判别) | `DomainDiscriminator` | ✅ 完成 |
| Prior p(z) 采样 | - | ❌ 待实现 |
| GAN-style 对抗训练 | - | ❌ 待实现 |

---

## 🚀 下一步优化建议

### 1. 添加 Prior 分布对抗训练

```python
class PriorDiscriminator(nn.Module):
    """判别 H 是否来自 Prior 分布"""
    def forward(self, z):
        return self.net(z)  # 输出 [0, 1] 概率

def sample_prior(batch_size, latent_dim, device):
    """从 Laplace 分布采样"""
    return torch.distributions.Laplace(0, 1).sample((batch_size, latent_dim)).to(device)

# 训练时:
h_real = sample_prior(batch_size, latent_dim)  # 真实样本
h_fake = encoder(x)  # 生成样本

D_real = discriminator(h_real)
D_fake = discriminator(h_fake)
```

### 2. 多核 MMD

```python
def multi_kernel_mmd(x, y, sigmas=[0.1, 0.5, 1.0, 5.0, 10.0]):
    """使用多个 sigma 值的 RBF 核"""
    mmd = 0
    for sigma in sigmas:
        mmd += compute_mmd_rbf(x, y, sigma)
    return mmd / len(sigmas)
```

### 3. 动态权重调整

```python
# 根据训练进度动态调整损失权重
def get_loss_weights(epoch, max_epochs):
    progress = epoch / max_epochs
    return {
        'recon': 1.0,
        'mmd': min(1.0, progress * 2),  # 逐渐增加 MMD 权重
        'adv': min(0.5, progress),       # 逐渐增加对抗权重
    }
```

---

## 📊 预期训练效果

| 指标 | 初期 | 中期 | 末期 |
|-----|------|------|------|
| L_recon | ~0.3 | ~0.1 | ~0.05 |
| R_mmd | ~0.5 | ~0.1 | ~0.01 |
| J_gan | ~0.7 | ~0.5 | ~0.3 |
| 域分离度 | 高 | 中 | 低 (域对齐) |

---

## 🎯 总结

你的 Framework 设计符合 **Domain Adaptation / Domain Generalization** 的经典范式：

1. **多域数据并行** - 确保每个 batch 包含所有域的样本
2. **共享 Encoder** - 学习域不变的特征表示
3. **三分支损失** - 重建 + 对齐 + 对抗，相互制衡
4. **MMD 对齐** - 显式最小化域间分布差异
5. **对抗训练** - 隐式强化域不变性

**当前实现完成度**: ~85%，主要缺少 Prior 分布的对抗训练部分。

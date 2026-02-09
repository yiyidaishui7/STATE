# MMD-AAE 损失函数完整指南

## 1️⃣ 三个损失函数详解

### Recon Loss (重建损失)

```python
# 公式
L_recon = MSE(x, x̂) = (1/N) × Σ ||xᵢ - Decoder(Encoder(xᵢ))||²

# 代码 (train_mmd_aae.py 第 214 行)
recon_loss = nn.functional.mse_loss(x_recon, x)
```

| 项目 | 说明 |
|------|------|
| **含义** | 原始输入与重建输出的均方误差 |
| **目的** | 确保隐空间表示保留足够信息 |
| **数值范围** | 0 ~ 1 (取决于输入归一化) |
| **理想值** | < 0.15 |

---

### MMD Loss (最大均值差异损失)

```python
# 公式
L_mmd = (1/3) × [MMD(z_K562, z_RPE1) + MMD(z_K562, z_Jurkat) + MMD(z_RPE1, z_Jurkat)]

MMD²(P, Q) = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
K(a,b) = exp(-γ||a-b||²)  # RBF 核

# 代码 (train_mmd_aae.py 第 101-130 行)
def compute_mmd_multi_kernel(x, y):
    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ...
```

| 项目 | 说明 |
|------|------|
| **含义** | 不同域隐空间分布的差异度量 |
| **目的** | 显式地拉近不同域的分布 |
| **数值范围** | 0 ~ ∞ (通常 0 ~ 0.5) |
| **理想值** | → 0 (越小越好) |

---

### Adv Loss (对抗损失)

```python
# 公式
L_adv = CrossEntropy(Discriminator(z), domain_labels)
      = -Σ log(softmax(D(z))_真实域)

# 代码 (train_mmd_aae.py 第 233-234 行)
z_grl = GradientReversalFunction.apply(z, grl_alpha)  # 梯度反转
adv_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
```

| 项目 | 说明 |
|------|------|
| **含义** | 判别器预测域标签的交叉熵 |
| **目的** | 让 Encoder 生成"域不可区分"的特征 |
| **数值范围** | 0 ~ ∞ |
| **理想值** | **≈ log(3) = 1.099** (3 类随机猜测的熵) |

---

## 2️⃣ 权重配置

### 当前设置

```python
# train_mmd_aae.py 第 210 行
def compute_loss(self, x, domain_labels, 
                 weight_recon=1.0,   # λ_recon
                 weight_mmd=1.0,     # λ_mmd  
                 weight_adv=0.1,     # λ_adv
                 grl_alpha=1.0):
```

### 总损失公式

```
L_total = 1.0 × Recon + 1.0 × MMD + 0.1 × Adv

例如你的结果:
L_total = 1.0 × 0.1291 + 1.0 × 0.0402 + 0.1 × 1.0947
        = 0.1291 + 0.0402 + 0.1095
        = 0.2788 ✓
```

---

## 3️⃣ 权重调优指南

### 设计原则

| 权重大 | 效果 | 风险 |
|--------|------|------|
| λ_recon ↑ | 更好的重建质量 | 可能忽视域对齐 |
| λ_mmd ↑ | 更强的域对齐 | 可能损坏重建/使分布坍缩 |
| λ_adv ↑ | 更强的域不变性 | 训练不稳定 |

### 推荐配置

| 场景 | λ_recon | λ_mmd | λ_adv |
|------|---------|-------|-------|
| **默认 (当前)** | 1.0 | 1.0 | 0.1 |
| 强调重建 | 2.0 | 0.5 | 0.1 |
| 强调对齐 | 1.0 | 2.0 | 0.2 |
| 保守训练 | 1.0 | 0.5 | 0.05 |

### 动态权重策略

```python
# 训练初期: 重建为主
# 训练后期: 增加对齐

def get_weights(epoch, max_epochs):
    progress = epoch / max_epochs
    return {
        'recon': 1.0,
        'mmd': min(2.0, 0.5 + 1.5 * progress),  # 0.5 → 2.0
        'adv': min(0.2, 0.05 + 0.15 * progress), # 0.05 → 0.2
    }
```

---

## 4️⃣ 判断标准

### 健康的训练曲线

| 指标 | Epoch 1 | Epoch 10 | Epoch 20 | 趋势 |
|------|---------|----------|----------|------|
| Recon | 0.60 | 0.15 | 0.13 | ↓ 快速下降后稳定 |
| MMD | 0.08 | 0.05 | 0.04 | ↓ 持续下降 |
| Adv | 1.0 | 1.05 | 1.09 | → 趋近 1.099 |

### 异常情况诊断

| 症状 | 原因 | 解决方案 |
|------|------|----------|
| Recon 不下降 | Encoder/Decoder 太小 | 增大网络容量 |
| MMD 不下降 | 梯度问题或λ太小 | 检查梯度 / 增大 λ_mmd |
| Adv → 0 | Discriminator 太强 | 减小 λ_adv 或减小 D 容量 |
| Adv > 2.0 | Discriminator 太弱 | 增大 D 容量 |
| 所有 loss 震荡 | 学习率太大 | 减小 LR |

---

## 5️⃣ 你的训练结果评估

```
Epoch 20: Recon=0.1291, MMD=0.0402, Adv=1.0947
```

| 指标 | 值 | 评估 | 说明 |
|------|-----|------|------|
| Recon | 0.129 | ✅ 良好 | < 0.15，重建质量好 |
| MMD | 0.040 | ✅ 良好 | 明显 < 初始值，正在对齐 |
| Adv | 1.095 | ✅ 完美 | ≈ log(3)=1.099，域不可区分 |

**结论: 训练成功，三个损失都达到理想状态！**

# MMD-AAE × STATE 集成项目说明书

## 📋 项目概述 (Project Overview)

将 **MMD-AAE** (Maximum Mean Discrepancy - Adversarial Autoencoder) 域泛化方法集成到 **STATE** 细胞扰动预测框架中，实现跨细胞系的域适应训练。

### 相关资源 (Resources)
| 项目 | 链接 |
|------|------|
| STATE 原始仓库 | https://github.com/ArcInstitute/state |
| MMD-AAE 参考实现 | https://github.com/YuqiCui/MMD_AAE |
| 你的 Fork | https://github.com/yiyidaishui7/STATE |

---

## ✅ 已完成工作 (Completed Work)

### 1. 数据验证 (Data Verification)

**创建文件**: `src/verify_h5_data.py`

| 细胞系 (Cell Line) | 细胞数 (Cells) | 基因数 (Genes) | 文件路径 |
|-------------------|---------------|---------------|----------|
| K562 | 18,465 | 18,080 | `competition_support_set/k562.h5` |
| RPE1 | 22,317 | 18,080 | `competition_support_set/rpe1.h5` |
| Jurkat | 21,412 | 18,080 | `competition_support_set/jurkat.h5` |
| **总计** | **62,194** | **18,080** | |

---

### 2. 三路并行 DataLoader (Three-Way Parallel DataLoader)

**创建文件**: `src/run_mmd_aae_simple.py`

实现了 `ParallelZipLoader` 类，同时从三个 DataLoader 各取一个 batch：
- 每域 Batch Size: 32
- 总 Batch Size: 96
- 每轮迭代数: 577

---

### 3. MMD-AAE 模型架构 (Model Architecture)

**创建文件**: `src/state/emb/nn/mmd_aae.py`

```
Input (18080 genes)
    │
    ▼
┌──────────┐
│ Encoder  │ → z ∈ R^512
└────┬─────┘
     │
┌────┴───────────────────┐
│                        │
▼                        ▼
┌──────────┐       ┌─────────────┐
│ Decoder  │       │Discriminator│
└────┬─────┘       └──────┬──────┘
     │                    │
     ▼                    ▼
Reconstruction      Domain Labels
(MSE Loss)          (Adversarial Loss)

        + MMD Loss (域对齐)
```

---

### 4. 完整训练脚本 (Training Script)

**创建文件**: `src/train_mmd_aae.py`

```bash
# 运行训练
cd ~/state/src
python train_mmd_aae.py
```

**默认超参数**:
| 参数 | 值 |
|------|-----|
| Batch Size | 32 × 3 = 96 |
| Hidden Dim | 1024 |
| Latent Dim | 512 |
| Learning Rate | 1e-4 |
| Epochs | 20 |

---

## 📁 文件结构 (File Structure)

```
state/
├── configs/
│   └── mmd_aae_config.yaml          # 配置文件
├── src/
│   ├── verify_h5_data.py            # 数据验证
│   ├── run_mmd_aae_simple.py        # 简化测试
│   ├── train_mmd_aae.py             # ⭐ 主训练脚本
│   └── state/emb/nn/
│       └── mmd_aae.py               # ⭐ 模型定义
└── competition_support_set/
    ├── k562.h5
    ├── rpe1.h5
    └── jurkat.h5
```

---

## 🚧 待完成工作 (TODO)

### 高优先级
- [ ] 运行完整训练: `python train_mmd_aae.py`
- [ ] 验证训练过程，观察损失收敛

### 中优先级
- [ ] 集成到原版 STATE Transformer 架构
- [ ] 添加验证集评估 (HepG2)
- [ ] 超参数调优

### 低优先级
- [ ] WandB 日志可视化
- [ ] 隐空间 t-SNE 可视化
- [ ] 添加下游任务损失

---

## 🔧 快速开始 (Quick Start)

```bash
# 1. 更新代码
cd ~/state
git pull origin main

# 2. 验证数据
cd src
python verify_h5_data.py

# 3. 测试 DataLoader
python run_mmd_aae_simple.py

# 4. 开始训练
python train_mmd_aae.py
```

---

## 📊 预期输出示例

```
[HH:MM:SS] INFO: MMD-AAE 训练
[HH:MM:SS] INFO: Device: cuda
[HH:MM:SS] INFO:   K562: 18465 cells
[HH:MM:SS] INFO:   RPE1: 22317 cells
[HH:MM:SS] INFO:   Jurkat: 21412 cells

[HH:MM:SS] INFO: === Epoch 1/20 ===
[HH:MM:SS] INFO: Loss: 0.2345 (Recon: 0.2000, MMD: 0.0300, Adv: 0.0045)
...
[HH:MM:SS] INFO: ✅ 训练完成!
```

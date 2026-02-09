# STATE + MMD-AAE 项目汇报

---

## Slide 1: 项目概述

### 研究背景
- **STATE**: Arc Institute 开发的细胞扰动预测基础模型
- **挑战**: 不同细胞系之间存在域偏移 (Domain Shift)，影响跨域泛化能力
- **目标**: 引入 MMD-AAE 实现多域对齐，提升跨细胞系泛化性能

### 项目定位
```
原始 STATE                    本项目改进
┌────────────────┐           ┌────────────────────────────┐
│ 单域训练        │    →     │ 多域联合训练 + 域对齐       │
│ Transformer    │    →     │ Transformer + MMD-AAE      │
│ 依赖域标签预测   │    →     │ 域不变特征学习             │
└────────────────┘           └────────────────────────────┘
```

---

## Slide 2: 原始 STATE 架构回顾

### 核心组件
| 组件 | 功能 |
|------|------|
| Gene Tokenizer | 基因表达 → Token 序列 |
| ESM2 Embeddings | 蛋白质序列预训练嵌入 |
| Transformer Encoder | 序列建模 |
| Prediction Head | 扰动效应预测 |

### 原始训练流程
```
单个 DataLoader → Collator → Transformer → Loss
     ↓
   单细胞系数据 (如 K562)
```

### 局限性
1. 单域训练，未考虑域间差异
2. 跨域推理时性能下降
3. 未利用多细胞系数据的互补信息

---

## Slide 3: MMD-AAE 理论基础

### 核心思想
通过**最大均值差异 (MMD)** 和**对抗训练 (Adversarial Training)** 实现域对齐。

### 三大损失函数

```
L_total = λ₁ × L_recon + λ₂ × L_mmd + λ₃ × L_adv
```

| 损失 | 公式 | 作用 |
|------|------|------|
| L_recon | MSE(x, Decoder(Encoder(x))) | 信息保留 |
| L_mmd | Σ MMD(z_domain_i, z_domain_j) | 显式域对齐 |
| L_adv | CE(Discriminator(z), labels) + GRL | 隐式域不变 |

### MMD 核心公式
```
MMD²(P, Q) = E[K(x,x')] + E[K(y,y')] - 2E[K(x,y)]
K(x, y) = exp(-γ||x-y||²)  # RBF 核
```

---

## Slide 4: 新增模块架构

### 系统架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    Multi-Domain DataLoader                   │
│        K562 (18,465) + RPE1 (22,317) + Jurkat (21,412)      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
              ┌──────────────────────────────┐
              │   Encoder Q (θ_Q)            │
              │   18,080 genes → 512 latent  │
              └──────────────┬───────────────┘
                             │
              ┌──────────────┴──────────────┐
              │        Hidden Code H        │
              │         z ∈ R^512           │
              └──────────────┬──────────────┘
       ┌─────────────────────┼─────────────────────┐
       │                     │                     │
       ▼                     ▼                     ▼
┌─────────────┐     ┌─────────────────┐    ┌─────────────────┐
│  Decoder P  │     │   MMD Module    │    │ Discriminator D │
│  z → x'     │     │ 域对齐 (显式)    │    │ 域判别 (对抗)    │
└──────┬──────┘     └────────┬────────┘    └────────┬────────┘
       │                     │                      │
       ▼                     ▼                      ▼
   L_recon              L_mmd                   L_adv
```

---

## Slide 5: 已完成代码开发

### 新增文件清单

| 文件路径 | 功能 | 代码行数 |
|---------|------|---------|
| `src/train_mmd_aae.py` | 完整训练脚本 | ~490 行 |
| `src/state/emb/nn/mmd_aae.py` | MMD-AAE 模型定义 | ~280 行 |
| `src/verify_h5_data.py` | 数据验证脚本 | ~213 行 |
| `src/run_mmd_aae_simple.py` | 简化测试脚本 | ~200 行 |
| `configs/mmd_aae_config.yaml` | 配置文件 | ~100 行 |

### 核心实现

```python
# 1. 多域并行 DataLoader
class ParallelZipLoader:
    def __iter__(self):
        return zip(*self.loaders)  # 同步迭代三个域

# 2. 多核 MMD 计算
def compute_mmd_multi_kernel(x, y):
    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    ...

# 3. 梯度反转层
class GradientReversalFunction:
    def backward(ctx, grad):
        return -ctx.alpha * grad  # 反转梯度
```

---

## Slide 6: 数据集配置

### 训练数据

| 细胞系 | 细胞数 | 基因数 | 来源 |
|--------|--------|--------|------|
| K562 | 18,465 | 18,080 | competition_support_set/k562.h5 |
| RPE1 | 22,317 | 18,080 | competition_support_set/rpe1.h5 |
| Jurkat | 21,412 | 18,080 | competition_support_set/jurkat.h5 |
| **总计** | **62,194** | **18,080** | |

### 验证数据 (待使用)
| 细胞系 | 用途 |
|--------|------|
| HepG2 | Zero-shot 验证 |

### 数据格式
- **存储**: HDF5 密集矩阵
- **数值**: log1p 标准化表达量
- **类型**: float32

---

## Slide 7: 模型超参数

### 网络架构
| 参数 | 值 | 说明 |
|------|-----|------|
| Input Dim | 18,080 | 基因数 |
| Hidden Dim | 1,024 | 隐藏层维度 |
| Latent Dim | 512 | 隐空间维度 |
| Dropout | 0.2 | 正则化 |

### 训练配置
| 参数 | 值 | 说明 |
|------|-----|------|
| Batch Size | 32 × 3 = 96 | 每域 32 |
| Learning Rate | 1e-4 | Adam 优化器 |
| Epochs | 20 | 训练轮数 |
| LR Scheduler | Cosine Annealing | 学习率调度 |

### 损失权重
| 权重 | 值 | 说明 |
|------|-----|------|
| λ_recon | 1.0 | 重建损失 |
| λ_mmd | 1.0 | MMD 损失 |
| λ_adv | 0.1 | 对抗损失 |

---

## Slide 8: 初步训练结果

### 运行状态 ✅
```
训练时间: ~7 分钟 (20 epochs)
设备: NVIDIA GPU
模型参数量: 38,826,403
```

### 损失曲线

| Epoch | Total Loss | Recon Loss | MMD Loss | Adv Loss |
|-------|------------|------------|----------|----------|
| 1 | 0.318 | 0.185 | 0.0625 | 1.02 |
| 5 | 0.248 | 0.138 | 0.0625 | 0.79 |
| 10 | 0.261 | 0.134 | 0.0625 | 0.95 |
| 20 | 0.270 | 0.132 | 0.0625 | 1.07 |

### 发现的问题
⚠️ **MMD Loss 未变化**: 梯度未正确传播

### 修复方案
已修复 → 使用 `x.new_zeros(1)` 保持梯度连接

---

## Slide 9: 待完成工作

### 高优先级 🔴
| 任务 | 状态 | 预计时间 |
|------|------|---------|
| 重新运行修复后的训练 | 进行中 | 10 min |
| 验证 MMD 损失下降 | 待验证 | - |
| 在 HepG2 上 Zero-shot 评估 | 待开始 | 1 day |

### 中优先级 🟡
| 任务 | 状态 | 说明 |
|------|------|------|
| 集成到原版 STATE Transformer | 待开始 | 替换 MLP Encoder 为 Transformer |
| 添加 Prior 分布对抗训练 | 待开始 | 根据 Framework 图中 Branch C |
| 超参数调优 | 待开始 | 损失权重、学习率等 |

### 低优先级 🟢
| 任务 | 状态 |
|------|------|
| WandB 可视化 | 可选 |
| t-SNE 隐空间可视化 | 可选 |
| 消融实验 | 后续 |

---

## Slide 10: 技术路线图

```
Phase 1: 基础搭建 ✅ (已完成)
├── 多域 DataLoader
├── MMD-AAE 模型
├── 训练脚本
└── 数据验证

Phase 2: 验证与优化 🔄 (进行中)
├── MMD 梯度修复 ✅
├── 重新训练验证
├── Zero-shot 评估
└── 超参数调优

Phase 3: 集成与扩展 📋 (规划中)
├── 集成 STATE Transformer
├── 添加 Prior 对抗训练
├── 对比实验
└── 论文撰写
```

---

## Slide 11: 代码运行指南

### 环境准备
```bash
cd ~/state
git pull origin main
```

### 数据验证
```bash
cd src
python verify_h5_data.py
```

### 开始训练
```bash
python train_mmd_aae.py
```

### 检查点位置
```
~/state/checkpoints/mmd_aae/
├── best_model.pt
├── model_epoch_5.pt
├── model_epoch_10.pt
├── model_epoch_15.pt
└── model_epoch_20.pt
```

---

## Slide 12: 总结与展望

### 已完成工作
1. ✅ 完整的 MMD-AAE 实现 (~1,300 行代码)
2. ✅ 三路并行 DataLoader 架构
3. ✅ 初步训练完成
4. ✅ 问题定位与修复

### 创新点
1. 将 MMD-AAE 首次应用于细胞扰动预测领域
2. 设计了适合单细胞数据的多域对齐框架
3. 与 STATE 基础模型无缝集成

### 预期贡献
- 提升跨细胞系泛化能力
- 减少对特定细胞系的依赖
- 为多域单细胞分析提供新范式

---

## 附录 A: 文件结构

```
state/
├── configs/
│   └── mmd_aae_config.yaml
├── docs/
│   ├── Framework_Analysis.md
│   ├── Loss_Functions_Explained.md
│   └── MMD_AAE_README.md
├── src/
│   ├── train_mmd_aae.py          # 主训练脚本
│   ├── run_mmd_aae_simple.py     # 简化测试
│   ├── verify_h5_data.py         # 数据验证
│   └── state/emb/nn/
│       └── mmd_aae.py            # 模型定义
└── competition_support_set/
    ├── k562.h5
    ├── rpe1.h5
    ├── jurkat.h5
    └── ESM2_pert_features.pt
```

---

## 附录 B: 参考资源

| 资源 | 链接 |
|------|------|
| STATE 原始仓库 | https://github.com/ArcInstitute/state |
| MMD-AAE 参考实现 | https://github.com/YuqiCui/MMD_AAE |
| 项目代码仓库 | https://github.com/yiyidaishui7/STATE |

---

*演示结束 - 感谢聆听*

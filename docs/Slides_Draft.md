# STATE + MMD-AAE：域对齐集成实验汇报
## 面向导师汇报版（含详细讲稿）

> 图片路径：`/media/mldadmin/home/s125mdg34_03/state/figures/`
> 数据截止：2026-04-03

---

## Slide 1 — 封面

### 标题
**STATE + MMD-AAE：多域对齐策略的集成、训练与评估**

### 副标题
问题动机 · MMD/GRL原理 · 训练配置 · 对比实验 · 结果分析

### 实验信息
| 项目 | 值 |
|------|----|
| 训练域 | K562 / RPE1 / Jurkat（共 62,194 细胞） |
| 评估域 | HepG2（零样本，训练中从未出现） |
| 硬件 | 1× RTX 3090 (24GB) |
| 实验对比 | No-MMD Baseline vs STATE+MMD |

---
**讲稿**

今天汇报我们在 STATE 模型上集成 MMD-AAE 域对齐模块的完整实验。内容分四部分：为什么做、怎么做、训练过程怎么样、最终结果说明什么。

---

## Slide 2 — 问题：为什么需要域对齐

### 核心问题

STATE 将每个细胞编码为 **512 维 CLS embedding**（L2 归一化，落在单位超球面上）。

在 K562、RPE1、Jurkat 三个细胞系上训练后，embedding 在高维空间中形成**三个分离的簇**：

```
Baseline UMAP（见 fig4 左图）：
  K562   → 右下角独立簇（红色）
  RPE1   → 左侧独立簇（蓝色）
  Jurkat → 左上角独立簇（绿色）
  
HepG2（测试，肝癌细胞）→ 模型从未见过，
  不知道应该落在哪里 → 零样本预测有偏差
```

### 解决思路

强制三个训练域的 embedding **在同一区域混合** → HepG2 等新域自然映射进来

**图：fig4_umap_comparison.png（左图 = baseline 三个分离簇，右图 = MMD对齐后混合）**

---
**讲稿**

这里放的是 UMAP 图。左边是没有 MMD 的 baseline，可以看到三个细胞系的 embedding 完全分开，在空间上各占一个角落。右边是加了 MMD 之后，三个簇开始互相交叠，说明对齐确实发生了。

HepG2 是肝癌细胞，生物学上和 K562（白血病）、RPE1（视网膜）、Jurkat（T细胞）都很不一样。我们的假设是：如果三个训练域共享同一个 embedding 空间，那么 HepG2 进来时也能更好地被解码。

---

## Slide 3 — 方法：损失函数设计

### 总体损失公式

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \alpha \cdot (\underbrace{2.0 \times \mathcal{L}_{MMD}}_{\text{分布对齐}} + \underbrace{0.1 \times \mathcal{L}_{adv}}_{\text{对抗判别}})$$

### 三项的含义

**① $\mathcal{L}_{pred}$（预测损失）**：TabularLoss，衡量模型对基因扰动响应的预测质量，是**主任务**，始终存在。

**② $\mathcal{L}_{MMD}$（最大均值差异）**：直接测量三个域 CLS embedding 的分布距离：

$$\mathcal{L}_{MMD} = \frac{1}{|\text{pairs}|} \sum_{i<j} MMD^2(P_i, P_j)$$

$$MMD^2(P,Q) = \frac{1}{|\sigma|}\sum_{\sigma}\left[\mathbb{E}_{x,x'\sim P}k(x,x') + \mathbb{E}_{y,y'\sim Q}k(y,y') - 2\mathbb{E}_{x\sim P,y\sim Q}k(x,y)\right]$$

使用多尺度 RBF kernel：$k(x,y) = \exp\left(-\frac{\|x-y\|^2}{2\sigma^2}\right)$，$\sigma \in \{0.1, 1.0, 10.0\}$

- MMD=0 意味着两个分布完全相同
- 三个尺度同时使用，避免单一带宽遗漏不同尺度的分布差异

**③ $\mathcal{L}_{adv}$（对抗判别）**：梯度反转层（GRL）+ 域分类器：

```
编码器 → CLS embedding → GRL → 域分类器（512→256→256→3）
                  ↑
          前向：恒等变换
          反向：梯度取反
```

- 分类器正向学会区分 K562/RPE1/Jurkat
- 编码器收到反向梯度，被迫生成**让分类器无法区分域**的 embedding

### $\alpha$：预热调度（Warmup Schedule）

$$\alpha = \begin{cases} 0 & \text{epoch} < 5 \\ \frac{\text{epoch}-5}{5} & 5 \leq \text{epoch} < 10 \\ 1.0 & \text{epoch} \geq 10 \end{cases}$$

- epoch 0–4：$\alpha=0$，只优化预测（让模型先学会预测）
- epoch 5–9：$\alpha$ 从 0 线性增加到 1
- epoch 10–15：$\alpha=1$，完整多任务训练

**为什么要 warmup？** 若一开始就施加对齐压力，encoder 尚未学到有效预测特征，GRL 会在信息被编码前就消除它，导致预测损失崩溃。

---
**讲稿**

这是整个方法的核心公式，有三点需要解释清楚。

第一，$\alpha$ 是什么。它是对齐损失的权重系数，控制什么时候、以多大力度引入域对齐。前5个epoch $\alpha=0$，模型只学预测任务，没有任何对齐压力。从第5个epoch开始线性增加，到第10个epoch $\alpha=1$，对齐全面激活。

第二，权重 2.0 和 0.1 是什么意思。MMD 权重 2.0 说明我们主要依靠 MMD 来做分布对齐，这是一个稳定的、可微的损失。ADV 权重只有 0.1，对抗训练只是辅助信号，避免它压倒主任务。

第三，多尺度 kernel 的意思。用三个不同的 $\sigma$ 值（0.1、1.0、10.0），就像用三种不同焦距的相机同时拍照——$\sigma=0.1$ 捕捉细粒度差异，$\sigma=10$ 捕捉全局偏移。

关于 $\alpha$ 的形状：目前实现是按 epoch 整数计算的，所以曲线是台阶状的（见 fig2 右下）。老师建议改成连续的 cosine schedule，我们已经实现了新版本正在训练，后面会展示。

---

## Slide 4 — 训练数据流水线

### ParallelZipLoader：三域均衡采样

```
K562  DataLoader (batch=16) ─┐
RPE1  DataLoader (batch=16) ─┤→ 合并 → batch_size=48
Jurkat DataLoader (batch=16)─┘

batch[9]: 域标签 = [0,0,...,0, 1,1,...,1, 2,2,...,2]
           K562×16        RPE1×16      Jurkat×16
```

每步保证三个域均衡出现 → MMD 计算每步都有有效的域对样本

### 训练配置

| 参数 | 值 |
|------|----|
| batch_size | 48（16×3域） |
| epochs | 16 |
| max_lr | 1e-5（微调量级） |
| gradient_accumulation | 8步（等效 batch=384） |
| optimizer | AdamW + Cosine LR |
| val集 | HepG2（9,386 细胞，纯 forward，无梯度更新） |
| alignment_warmup | 5 epochs |

### 关键设计原则

- **HepG2 不参与训练**：只在 val，仅用于监控 val_loss，不产生梯度
- **三域均衡**：每步 K562/RPE1/Jurkat 各 16 个细胞，MMD 计算公平

---
**讲稿**

训练数据这里有一个工程细节值得说明。原始 STATE 的 DataLoader 只支持单一数据集，我们重写了一个 ParallelZipLoader，把三个细胞系的 DataLoader 并联起来，每个 batch 里保证三个域各占三分之一。这样 MMD 在每一步都能计算三对域之间的分布距离。

验证集是 HepG2，9386 个细胞，但它只做 forward pass 计算 val_loss，不更新梯度。这是真正的零样本设置。

---

## Slide 5 — 训练曲线分析

**图：fig2_training_curves.png**

### 四个子图解读

**左上：Training Loss**
- Baseline（蓝）最终 ~20.6，STATE+MMD（橙）最终 ~20.0
- MMD 模型训练损失**更低**：表明对齐损失不干扰预测，反而略有帮助
- MMD 曲线在 step ~900 处（α 开始增加）有短暂凸起，随后恢复下降

**右上：Validation Loss（HepG2 零样本）**
- 两者收敛到几乎相同水平（~21.9）
- 说明 HepG2 的零样本预测质量相同

**左下：MMD & ADV Loss**
- MMD loss（橙）从 ~0.03 下降至 ~0.01：分布距离持续缩小 ✅
- ADV loss（紫）稳定在 ~1.099 ≈ ln(3) = 1.0986：判别器被击败，输出均匀分布 ✅

**右下：α Schedule（阶梯状）**
- step ~900 → α=0.2，每个 epoch 增加 0.2，step ~1500 → α=1.0
- **阶梯状是当前实现的局限**（见 Slide 9 改进版）

---
**讲稿**

这里重点看两个现象。

第一，训练损失 MMD 模型比 baseline 更低。这说明加了 MMD 之后，模型对训练域（K562/RPE1/Jurkat）的预测质量反而提升了。一个可能的解释是：域对齐迫使模型学到更普适的特征，去掉了细胞系特异性的"快捷方式"，反而提高了泛化性。

第二，ADV loss 在 ln(3) 附近。ln(3)≈1.0986 是三分类问题在完全随机预测时的熵值。ADV loss 稳定在这里，说明判别器完全无法区分三个域，GRL 从对抗博弈的角度成功了。

但是注意：这个"成功"是对抗博弈里的成功，不代表 embedding 完全没有域信息，后面的 DomainClsAcc 指标会揭示这个悖论。

---

## Slide 6 — 定量结果 I：HepG2 零样本 Pearson r

**图：fig1_pearson_bar.png**

### 细胞级 Pearson r（Mode A）

对每个 HepG2 细胞：CLS embedding → binary_decoder → 预测基因表达 vs 真实表达

| 模型 | Mean r | Std | Median | N cells |
|------|--------|-----|--------|---------|
| No-MMD Baseline | **0.7003** | 0.0456 | 0.7039 | 2000 |
| STATE+MMD | **0.6986** | 0.0460 | 0.7016 | 2000 |
| Δ (MMD - Baseline) | **-0.0017** | — | — | — |

### 统计解读

- Δ = -0.0017，远小于 std = 0.046
- 95% 置信区间完全重叠
- **结论：两者没有统计意义上的差异（p > 0.05）**

### 这个 0.70 意味着什么？

- R² ≈ 0.49：模型解释了 HepG2 基因表达方差的约 49%
- CV = 6.5%：2000 个细胞的表现高度一致
- 零样本（HepG2 从未参与训练）能达到 r=0.70 是强泛化能力的体现

---
**讲稿**

Pearson r 是我们最关心的指标，因为它直接衡量模型对零样本细胞系的预测能力。

结果是：baseline 和 MMD 模型的 r 值几乎完全一样，差值只有 0.0017，而单个细胞的标准差是 0.046，差值是标准差的 1/27。从统计上讲，这个差异完全在噪声范围内。

这说明：**MMD 没有改善零样本预测性能**。但反过来看，r=0.70 本身是一个很好的结果——一个从没见过 HepG2 的模型，能把它的基因表达预测到 r=0.70，说明多域预训练确实学到了跨细胞系泛化的能力。

---

## Slide 7 — 定量结果 II：域对齐指标

**图：fig3_alignment_metrics.png**

### 四项指标对比

| 指标 | Baseline | STATE+MMD | 含义 |
|------|----------|-----------|------|
| DomainClsAcc ↓ | 99.66% | 99.34% | 分类器区分域的准确率 |
| Silhouette ↓ | **0.3965** | **0.0412** | 局部几何混合程度 |
| MMD (mean) ↓ | **0.1878** | **0.0009** | 分布统计距离 |
| CORAL ↓ | 0.000 | 0.000 | 协方差矩阵差异 |

### 指标解读

**Silhouette 0.40 → 0.04**（↓ 10×）
- Silhouette 衡量每个样本离本域中心 vs 离他域中心的比
- 0 = 完美混合，+1 = 完全分离
- 0.04 ≈ 0：三个域在局部几何上已经**互相穿插** ✅

**MMD 0.1878 → 0.0009**（↓ 200×）
- MMD 直接测量 RBF kernel 空间中的分布距离
- 从明显分离（0.19）降至几乎为零（0.0009） ✅

**DomainClsAcc 99.66% → 99.34%（几乎不变）**
- 一个新训练的 MLP 仍然能以 99% 准确率区分三个域
- 这是"GRL 悖论"——见下一页

---
**讲稿**

这里有一个很有意思的现象：MMD 和 Silhouette 都说对齐成功了，但 DomainClsAcc 却还是 99%，几乎没有变化。

理解这个矛盾的关键是：这三个指标测量的是不同层次的东西。

MMD 测的是分布的低阶统计量——均值、方差在 kernel 空间里是否相同。Silhouette 测的是局部几何——每个点的近邻里有多少来自其他域。

而 DomainClsAcc 用的是一个全新训练的 MLP，它可以在 512 维空间里自由地找任何判别边界，包括非线性的、高阶的特征。MMD 对齐了统计量，但 embedding 里还藏着高阶的域特异性信息，新的 MLP 能找到。

这个结果其实揭示了 MMD+GRL 方法的局限：它对齐了"分布的外形"，但没有彻底消除域信息。

---

## Slide 8 — GRL 悖论：为什么 DomainClsAcc 还是 99%？

### 悖论描述

```
训练时：GRL 判别器 ADV loss ≈ ln(3) = 1.0986（随机水平）
         ↓ 说明训练时的判别器被完全击败

评估时：新 MLP 分类器 Acc = 99.3%
         ↓ 说明 embedding 里仍有很强的域信息
```

### 根本原因

GRL 击败的只是**对抗博弈中的那一个判别器**，不是所有可能的分类器：

| | GRL 判别器（训练中） | 后验 MLP（评估时） |
|--|---------------------|-------------------|
| 是否受 GRL 影响 | ✅ 是，梯度被反转 | ❌ 否，独立训练 |
| 编码器是否防御 | ✅ 是，主动混淆它 | ❌ 否，编码器冻结 |
| 结论 | 被击败：loss ≈ ln(3) | 自由分类：Acc=99% |

### 更深层的原因：adv_weight 太小

$$\mathcal{L}_{total} = \mathcal{L}_{pred} + \alpha \cdot (2.0 \times \mathcal{L}_{MMD} + \underbrace{0.1}_{\text{很小}} \times \mathcal{L}_{adv})$$

- adv_weight=0.1，GRL 对 encoder 施加的压力很弱
- encoder 只需要"稍微"混淆判别器即可满足损失要求
- 大量域特异性信息仍然保留在 embedding 里

### 物理图像

```
低阶统计（MMD 测量）  →  ✅ 对齐
局部几何（Silhouette） →  ✅ 对齐  
高阶判别特征（MLP）    →  ❌ 未对齐（adv 压力不足）
```

---
**讲稿**

这是整个实验最值得深入讨论的地方。

训练时，GRL 判别器的 loss 稳定在 ln(3)，说明它已经分不清三个域了，被完全击败。但评估时，我们用一个新的独立 MLP 来测，它能 99% 区分三个域，说明域信息还在 embedding 里。

为什么会这样？因为 GRL 训练的是一个"对抗博弈"：编码器学会欺骗特定的那个判别器。一旦换一个判别器，编码器没有针对它防御，当然就能被区分了。

这是 GRL 方法的已知局限，在 domain adaptation 文献里有讨论。要真正消除域信息，需要更强的对抗压力（增大 adv_weight），或者用更鲁棒的方法比如 DANN 的全部训练框架。

---

## Slide 9 — 改进：cosine α schedule（正在训练）

### 当前问题：阶梯状 α

```python
# 当前实现（离散，按 epoch 整数）
alpha = (current_epoch - 5) / 5
# epoch 5→0.2, epoch 6→0.4, ... 每个 epoch 内所有 step 的 α 相同
```

→ 每个 epoch 边界处 loss 出现**跳变**（见 fig2 右下阶梯图）

### 新版本：cosine 连续 α

```python
# 新实现（连续，按 global_step）
progress = (global_step - warmup_steps) / ramp_steps  # 0→1
alpha = 0.5 * (1 - cos(π × progress))                 # S 形曲线
```

| 对比 | 旧版（阶梯） | 新版（cosine） |
|------|------------|---------------|
| 变化频率 | 每 epoch 跳一次 | 每 step 连续变化 |
| 曲线形状 | 5个台阶 | S 形平滑曲线 |
| loss 稳定性 | epoch 边界处跳变 | 无跳变 |
| 实验名 | `mmd_aae` | `mmd_aae_cosine` |

**cosine 版本当前正在训练，完成后与旧版对比**

---
**讲稿**

老师上次指出 α 最好是连续的，这个建议非常准确。当前的实现用 epoch 整数计算，每个 epoch 内所有 step 的 α 完全相同，epoch 切换时突然跳变，会导致 loss 不稳定。

我们新实现的版本用 global_step 来计算，α 在每一步都会微小地增加，形成 cosine 的 S 形曲线——开始增加慢，中间快，接近 1 时又慢下来。这样 loss 的变化更平滑，训练更稳定。新版本目前正在运行，结果出来后会做对比。

---

## Slide 10 — 核心发现与结论

### 量化总结

| 维度 | 发现 |
|------|------|
| **任务性能** | Baseline r=0.700，MMD r=0.699，**Δ=-0.0017（无意义）** |
| **分布对齐** | Silhouette 0.40→0.04，MMD 0.19→0.001，**对齐成功** |
| **高阶域信息** | DomainClsAcc 99.7%→99.3%，**几乎不变** |

### 核心结论

> **r ≈ 0.70 来自多域预训练本身，而非域对齐。MMD 成功对齐了分布的统计量，但没有改善零样本预测性能。**

### 解释

1. **多域预训练已经足够**：同时在 K562/RPE1/Jurkat 上训练，模型已经学到了跨域泛化的能力，不需要额外的显式对齐
2. **对齐是必要但不充分条件**：统计量对齐 ≠ 下游任务改善
3. **adv_weight 太弱**：GRL 未能消除 embedding 中的高阶域特异性信息

### 局限与下一步

| 局限 | 改进方向 |
|------|---------|
| α 是离散的 | ✅ cosine 版本已实现（正在训练） |
| adv_weight=0.1 太弱 | 消融：adv_weight = 0.5/1.0 |
| 只评估了细胞级 Pearson | 增加 iLISI、kBET 等 scRNA-seq 标准指标 |
| HepG2 生物学差距太大 | 在更相似的细胞系上测试（如 GM12878） |

---
**讲稿**

最终的结论是：MMD 对齐从统计指标来看是成功的——Silhouette 和 MMD 数值都大幅下降，UMAP 上也能看到三个域开始混合。但这种对齐没有带来下游任务（零样本 Pearson r）的提升。

这是一个"负结果"，但是有意义的负结果。它告诉我们：r=0.70 这个性能不是来自域对齐，而是来自多域联合训练本身。三个不同细胞系的数据同时训练，模型学到了更普适的基因调控规律，这才是泛化能力的来源。

两个明确的改进方向：一是 adv_weight 太小，GRL 力度不够，可以做消融实验；二是 cosine α schedule，已经实现了，结果很快就能看到。

---

## 附录 A — 完整指标表

| 指标 | Baseline | STATE+MMD | 变化 |
|------|----------|-----------|------|
| Pearson r (mean) | 0.7003 | 0.6986 | -0.0017 |
| Pearson r (std) | 0.0456 | 0.0460 | +0.0004 |
| Pearson r (median) | 0.7039 | 0.7016 | -0.0023 |
| DomainClsAcc | 0.9966 | 0.9934 | -0.0032 |
| Silhouette | 0.3965 | 0.0412 | **-0.355** |
| MMD (mean) | 0.1878 | 0.0009 | **-0.187** |
| CORAL | 0.000 | 0.000 | 0 |

## 附录 B — 实验设计说明（回应老师问题）

**Q1：α 是什么？**
α 是对齐损失的权重系数，$\mathcal{L} = \mathcal{L}_{pred} + \alpha \cdot \mathcal{L}_{align}$。epoch 0–4 时 α=0（只学预测），之后线性增加到 1。

**Q2：α 为什么是阶梯状？**
当前用 `current_epoch`（整数）计算，每个 epoch 内 α 不变，epoch 边界跳一级。新版 cosine 已实现（正在训练），用 `global_step` 每步连续更新。

**Q3：baseline 分数是否太高（HepG2 被喂进去训练了）？**
已验证：训练代码硬编码 K562/RPE1/Jurkat 三个域，HepG2 仅在 val（无梯度更新）。r=0.700 是真正的零样本结果。之前 3.17 日的对比无效（两个模型用的是同一个 checkpoint），已重新训练独立 baseline。

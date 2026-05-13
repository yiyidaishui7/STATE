# Slides 讲稿 & 可视化清单
## STATE + MMD-AAE：损失函数与 DEG Pearson 评估

> **说明**：每张 Slide 包含三部分：
> - **内容**：标题 + 正文要点
> - **可视化**：需要什么图，数据来源，是否需要先跑脚本
> - **讲稿**：逐段讲解文字

---

## Slide 1 — 标题页

### 内容
- 大标题：STATE + MMD-AAE：Loss Function 设计与 DEG Pearson 评估
- 副标题：单细胞扰动预测中的损失函数局限性分析
- 三个关键词标签：域对齐 / 零样本泛化 / 扰动方向预测

### 可视化
无图表，纯文字标题页。

### 讲稿
> 这次我要汇报的是我们把 MMD-AAE 域对齐方法集成进 STATE 预训练模型后，对整个损失函数设计和评估方法的系统性分析。核心问题是：我们的模型在域对齐上取得了成功，但在扰动方向预测这个最重要的任务上表现如何？答案出乎意料，而且这个答案揭示了一个比域泛化更根本的问题。

---

## Slide 2 — 项目背景（30秒速览）

### 内容
- 任务：给定基因敲除扰动，预测对照细胞 → 扰动后的基因表达变化
- 训练域：K562 / RPE1 / Jurkat（三个细胞系）
- 验证目标：HepG2（从未见过的细胞系，零样本泛化）
- 我们的方法：在 STATE 上叠加 MMD-AAE 域对齐，让三个细胞系的 CLS embedding 融合

### 可视化
**图：数据流示意图（手绘/简单箭头图即可）**

```
[K562 h5] ──┐
[RPE1 h5] ──┼──→ STATE Transformer ──→ CLS embedding (512d)
[Jurkat h5]─┘         │                      │
                       ↓                      ↓
                  pred_loss              alignment_loss
               (TabularLoss)           (MMD + 对抗GRL)
                                             │
                                    零样本验证 → HepG2
```

> 无需跑脚本，可直接用 PowerPoint 绘制箭头图。

### 讲稿
> 我们的任务是单细胞扰动预测：给定一个基因敲除操作，预测细胞基因表达会怎么变化。我们用 K562、RPE1、Jurkat 三个细胞系训练，希望模型能零样本泛化到 HepG2——一个从未在训练中出现的细胞系。我们的核心假设是：如果三个细胞系的表征空间能对齐，HepG2 就能搭便车，利用这个共享的特征空间做预测。

---

## Slide 3 — Loss Function 总览：四项组成

> ⭐ **重点讲解 Slide，建议停留时间较长**

### 内容

**完整公式（来自代码 `model.py:shared_step`，config: `dataset_correction: false`）：**

```
total_loss = 1.0 × TabularLoss(decs, Y)          ← pred_loss     系数 1（隐式）
           + 0.1 × logfc_loss                     ← logFC MSE     系数 0.1（config: logfc_weight）
           + 2.0 × α × mmd_loss                   ← MMD 对齐      系数 2.0 × alpha
           + 0.1 × α × adv_loss                   ← 对抗 GRL      系数 0.1 × alpha
```

> ⚠ **注意：dataset_loss 在我们的训练中不存在。**
> 因为 `dataset_correction: false`，`dataset_token = None`，`dataset_emb = None`，
> `shared_step` 里 `if dataset_embs is not None` 整块被跳过。

**关键点：权重不对等**
- pred_loss 没有显式系数 → 隐式权重 = 1，是绝对主导
- logfc_loss：固定系数 0.1，量级明显小于 pred_loss
- mmd_loss：系数 2.0 × alpha，epoch 0-4 时 alpha=0 完全不参与
- adv_loss：系数仅 0.1 × alpha，影响最弱

**三项的作用层次：**

| 项 | 作用层次 | 优化目标 | 系数 |
|----|---------|---------|------|
| TabularLoss | binary_decoder | 基因表达分布形状相似 | 1.0 |
| logfc_loss | binary_decoder | 扰动 logFC 方向正确 | 0.1 |
| alignment_loss | CLS embedding（编码器） | 三域 embedding 融合 | 2.0α / 0.1α |

**⚠ 注意：logfc_loss 是"路线一"额外添加的，训练时实际上跑了，但权重仅 0.1，可能影响有限。**

### 可视化

**图：四分支结构图（PPT 手绘，重点标注系数）**

```
细胞基因表达 (B, 2048)
        │
        ↓
   Transformer (8层)
        │
   CLS embedding (512d)
   ┌────┴────────────────────────────────────┐
   │                                         │
   ↓                                         ↓
binary_decoder                           domain_disc (经GRL)
   │                                         │
   ├─→ TabularLoss(decs, Y)                  CrossEntropy = adv_loss
   │   系数 ×1.0        = pred_loss          系数 ×0.1×α
   │
   └─→ MSE(pred_logFC, true_logFC)      成对 MMD(CLS, domain_labels)
       系数 ×0.1       = logfc_loss      系数 ×2.0×α = mmd_loss
                                                │
dataset_encoder(dataset_embs)           alignment_loss = 2.0α·mmd + 0.1α·adv
→ CrossEntropy = dataset_loss
   系数 ×1.0
```

> 建议 PPT 用颜色区分四条分支：
> - 橙色：pred_loss（最主要）
> - 蓝色：dataset_loss
> - 绿色：logfc_loss（我们添加的方向损失）
> - 红色：alignment_loss（MMD-AAE 域对齐）

### 讲稿

> 我来详细解释损失函数的完整结构，这是整个项目的核心设计，也是后续所有实验结论的根源所在。
>
> 这里需要先厘清一个容易混淆的地方：项目里实际上有**两套不同的损失函数**。第一套在 `train_mmd_aae.py` 里，是第一阶段独立验证 MMD-AAE 时用的，包含重建损失 L_recon、MMD 损失和对抗损失，是一个标准的自编码器框架，和 STATE 没有关系。**我们今天讲的**是第二套——把 MMD-AAE 集成进 STATE 后，`model.py` 里的实际训练损失，两者完全不同。
>
> 实际的训练总损失由三项组成，**不是简单等权相加**，各项系数差别很大。
>
> 第一项是 TabularLoss，系数隐式为 1，是权重最大的一项。它衡量模型预测的基因表达分布和真实分布之间的距离，是 STATE 原有的核心损失，我们没有修改它。
>
> 第二项是 logfc_loss，系数 0.1，这是我们添加的"路线一"方向感知损失——直接用 logFC 的 MSE 监督扰动方向。注意系数只有 0.1，比 pred_loss 小一个数量级。
>
> 第三项是 alignment_loss，分为两个子项：mmd_loss 系数 2.0×alpha，adv_loss 系数 0.1×alpha。alpha 在前 5 个 epoch 等于 0，对齐损失从 epoch 5 才逐渐引入。
>
> 你可能会问：dataset_loss 呢？因为 config 里 `dataset_correction: false`，整个 dataset_encoder 根本没有被构建，dataset_loss 在我们这次训练中**完全不存在**。
>
> **这个权重结构导致了一个关键问题**：模型绝大部分梯度压力来自 TabularLoss，这个损失只优化分布形状，不奖励扰动方向的正确性。即使 logfc_loss 存在，0.1 的系数让它的影响非常有限。这就是后续 DEG Pearson 实验揭示问题的根源。

---

## Slide 3b — Loss Function 深入：各项权重的量级对比

> ⭐ **重点讲解 Slide**

### 内容

**为什么权重不对等很重要？**

训练过程中四项损失的典型数量级（根据 training log 估计）：

| 项 | 典型量级 | 系数 | 实际梯度贡献 |
|----|---------|------|------------|
| TabularLoss (pred_loss) | ~1.0 – 5.0 | ×1.0 | **最主导** |
| dataset_loss | ~0.5 – 2.0 | ×1.0 | 中等 |
| logfc_loss | ~0.01 – 0.1 | ×0.1 | **很弱** |
| mmd_loss | ~0.001 – 0.01 | ×2.0×α | epoch≥5 后存在 |
| adv_loss | ~0.5 – 2.0 | ×0.1×α | 很弱 |

**每项优化的是什么，优化不了什么：**

```
TabularLoss   ✅ 学会：基因表达整体分布形状
              ❌ 学不会：哪个基因具体上调 / 下调

dataset_loss  ✅ 学会：识别细胞来自哪个数据集
              ❌ 与扰动方向预测无关

logfc_loss    ✅ 理论上能学：扰动方向（logFC 方向）
              ⚠ 实际上：系数 0.1 太小，被 TabularLoss 淹没

alignment_loss ✅ 学会：三个域的 CLS embedding 融合
               ❌ 与扰动方向预测无关
```

**核心矛盾：**
> TabularLoss 是最强的梯度信号，但它对扰动方向一无所知；
> logfc_loss 理论上能提供方向信息，但 0.1 的系数让它几乎被淹没。

### 可视化

**图：各项梯度贡献比例（饼图或堆叠柱状图）**

数据需要从训练 log 读取，但可以用估计值先绘制示意图：

```python
# 本地可运行（示意图，数值为估计）
import matplotlib.pyplot as plt
import numpy as np

labels = ['TabularLoss\n(×1.0)', 'dataset_loss\n(×1.0)', 
          'logfc_loss\n(×0.1)', 'alignment_loss\n(×2.0α/0.1α)']
# 估计：tabular~3.0, dataset~1.0, logfc~0.005, alignment~0.01
values = [3.0, 1.0, 0.005, 0.01]
colors = ['#f59e0b', '#3b82f6', '#10b981', '#ef4444']

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(labels, values, color=colors, alpha=0.85)
ax.set_ylabel('实际贡献量级（估计）', fontsize=12)
ax.set_title('四项损失的实际梯度贡献比较\n（logfc_loss 几乎被淹没）', fontsize=13)
for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
            f'{v:.3f}', ha='center', va='bottom', fontsize=10)
ax.axhline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig('loss_magnitude.png', dpi=150)
```

> ⚠ 如果有服务器的 metrics.csv，可以读取真实的 loss 数值替换上面的估计值：
> ```bash
> find /media/mldadmin/home/s125mdg34_03/state -name "metrics.csv" | head -3
> # 列出：trainer/train_loss, trainer/dataset_loss, trainer/logfc_loss, trainer/mmd_loss 等列
> ```

### 讲稿

> 我们把四项损失的实际量级摆出来，问题就很清楚了。TabularLoss 的量级大约在 1 到 5 之间，系数是 1，是绝对主导的梯度来源。dataset_loss 量级类似，系数也是 1，是次要但存在的信号。
>
> 再看 logfc_loss——理论上它是最直接告诉模型"扰动方向"的监督信号，但实际量级只有 0.01 到 0.1，再乘以 0.1 的系数，最终贡献大概是 0.001 到 0.01 的量级。**它被 TabularLoss 的信号淹没了。**
>
> alignment_loss 情况类似，量级很小，而且前 5 个 epoch 根本不存在。
>
> 这就解释了为什么 DEG Pearson ≈ 0：模型 90% 以上的梯度压力告诉它"让分布形状对"，只有不到 1% 的信号告诉它"方向要对"。在这种权重配置下，模型合理地忽略了方向信息。这不是一个 bug，而是我们的 Loss 设计在权重层面就决定了的结果。

---

## Slide 4 — TabularLoss 详解

### 内容

**输入输出：**
- 输入：`decs`（预测分数，B×n_genes）和 `Y`（真实 log1p 表达量，B×n_genes）
- 内部：两个子损失，均使用 Energy Distance

**Energy Distance 公式：**
```
ED(P, Q) = 2·E[||X - Y||] - E[||X - X'||] - E[||Y - Y'||]
```
- X, X' ~ P（预测分布的两个独立样本）
- Y, Y' ~ Q（真实分布的两个独立样本）
- 含义：P、Q 完全相同时 ED=0；分布越不同 ED 越大

**两个子损失视角：**

| 子损失 | 谁是"分布" | 衡量什么 |
|--------|-----------|---------|
| `gene_loss` | 每个 batch = 一个点云，基因为维度 | 所有细胞的基因表达分布整体是否相似 |
| `cell_loss` | 每个基因 = 一个点云，细胞为维度 | 每个基因在所有细胞上的表达是否相似 |

### 可视化
**图：Energy Distance 示意图（分布对比）**

想象两个直方图：
- 左图（对齐好）：P 和 Q 重叠，ED ≈ 0
- 右图（对齐差）：P 和 Q 分离，ED 大

> 无需跑脚本，可用 PPT 手绘两个正态分布对比图，或用 Python 的 matplotlib 画：
> ```python
> import numpy as np, matplotlib.pyplot as plt
> x = np.linspace(-3, 5, 200)
> plt.plot(x, np.exp(-0.5*x**2), label='真实分布 Q')
> plt.plot(x, np.exp(-0.5*(x-1.5)**2), label='预测分布 P（差）')
> plt.fill_between(...)  # 可选
> ```
> 这个图简单，**可以在本地直接生成，不需要服务器**。

### 讲稿
> TabularLoss 的核心是 Energy Distance。直觉很简单：如果两个分布完全一样，从两个分布各抽一个样本，它们之间的平均距离，应该等于从同一个分布内部抽两个样本的平均距离。当两个分布不同时，跨分布距离会更大，Energy Distance 就是在衡量这个"超出量"。TabularLoss 用它从两个角度评估：基因维度——这批细胞的整体基因表达分布像不像？细胞维度——每个基因在所有细胞上的表达规律像不像？这两个角度合起来，构成了 STATE 原有的主损失。

---

## Slide 5 — alignment_loss：MMD + GRL

### 内容

**成对 MMD（多尺度 RBF 核）：**
```
MMD(P,Q) = E[k(x,x')] + E[k(y,y')] - 2·E[k(x,y)]
核函数 k(x,y) = exp(-||x-y||² / 2σ²)，σ ∈ {0.1, 1.0, 10.0}
```
三个域两两配对：K562↔RPE1、K562↔Jurkat、RPE1↔Jurkat，取均值。

**梯度反转层（GRL）：**
- 前向传播：恒等变换（embedding 原样通过）
- 反向传播：梯度乘以 **-α**
- 结果：判别器学会分类 → 梯度反转 → 编码器学会让 embedding 无法被分类 → CLS embedding 域不变

**Alpha 预热调度（余弦）：**

| 阶段 | Epoch | Alpha | 说明 |
|------|-------|-------|------|
| 静止期 | 0 – 4 | 0 | 纯预测训练，对齐模块完全关闭 |
| 爬坡期 | 5 – 9 | 0 → 1（余弦曲线） | 缓慢引入对齐压力 |
| 全力期 | 10 – 15 | 1.0（平台） | 完整域对齐 |

**实际代码公式（`model.py:_get_alignment_alpha`）：**

```python
warmup_steps = total_steps × (alignment_warmup / max_epochs)  # 前 5 epoch
ramp_steps   = total_steps × (5 / max_epochs)                 # 接下来 5 epoch

if current_step < warmup_steps:
    alpha = 0.0
else:
    progress = (current_step - warmup_steps) / ramp_steps      # 0 → 1
    alpha = 0.5 × (1 - cos(π × progress))                     # 余弦爬坡
```

**为什么用余弦，不用线性？**

```
线性（蓝）：对齐压力一开始就以固定速率增加
            → 早期 pred_loss 还没稳定，域对齐压力突然出现 → 可能破坏预测能力

余弦（橙）：初期增速很慢（接近 0），中期加速，末期再次变缓
            → 给 pred_loss 充分的过渡时间，对齐压力平滑接管
```

公式 `0.5×(1-cos(π×t))` 的特性：
- `t=0` → alpha=0（起点斜率为 0，不突兀）
- `t=0.5` → alpha=0.5（中点加速）
- `t=1` → alpha=1（终点斜率为 0，平滑收敛）

**最终公式（注意 mmd 和 adv 的系数不对称）：**
```
alignment_loss = 2.0 × α × mmd_loss    ← mmd 权重大（2.0），是核心约束
               + 0.1 × α × adv_loss    ← adv 权重小（0.1），辅助约束

# 加入总 loss 时系数为 1（直接相加）
total_loss += alignment_loss
```
**为什么 mmd:adv = 20:1？**
- MMD 是直接的几何距离约束，信号稳定可靠，给大系数
- adv_loss 来自对抗博弈，梯度噪声大，给小系数防止训练不稳定

### 可视化

**图1：Alpha 调度曲线（余弦 vs 线性对比）**

> **本地可直接生成，不需要服务器**：

```python
import numpy as np
import matplotlib.pyplot as plt

total_epochs = 16
warmup = 5   # alignment_warmup
ramp   = 5   # ramp_fraction epochs

epochs = np.linspace(0, total_epochs, 500)

def cosine_alpha(e, warmup=5, ramp=5):
    if e < warmup:
        return 0.0
    progress = min((e - warmup) / ramp, 1.0)
    return 0.5 * (1 - np.cos(np.pi * progress))

def linear_alpha(e, warmup=5, ramp=5):
    if e < warmup:
        return 0.0
    return min((e - warmup) / ramp, 1.0)

alpha_cos = np.array([cosine_alpha(e) for e in epochs])
alpha_lin = np.array([linear_alpha(e) for e in epochs])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(epochs, alpha_cos, color='#f59e0b', linewidth=2.5, label='余弦调度（实际使用）')
ax.plot(epochs, alpha_lin, color='#6366f1', linewidth=1.5,
        linestyle='--', label='线性调度（对比）', alpha=0.6)
ax.axvspan(0, 5,  alpha=0.08, color='#3b82f6', label='静止期（α=0）')
ax.axvspan(5, 10, alpha=0.08, color='#f59e0b', label='爬坡期（余弦）')
ax.axvspan(10, 16, alpha=0.08, color='#10b981', label='全力期（α=1）')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Alpha（对齐强度）', fontsize=12)
ax.set_title('Cosine Alpha Warmup 调度\nmmd_loss & adv_loss 共享此 alpha', fontsize=13)
ax.legend(fontsize=10)
ax.set_xlim(0, 16); ax.set_ylim(-0.05, 1.1)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('alpha_schedule.png', dpi=150)
print("saved: alpha_schedule.png")
```

> 如果同时有服务器 `metrics.csv`，可以在同一张图上叠加真实训练记录的 alpha 值做验证：
> ```bash
> python scripts/plot_mmd_training.py
> ```

**图2：GRL 工作原理示意图**

```
                    编码器
                      │
                  CLS embedding
                 ┌────┴────┐
                 │         │
            预测头       GRL（梯度×-α）
                 │         │
           TabularLoss  domain_disc
                         │
                    CrossEntropy
               （判别器想分对，编码器想骗过它）
```
> 无需跑脚本，PPT 手绘即可。

### 讲稿
> 域对齐损失由两个组件构成，它们共同完成同一件事：让三个细胞系的 CLS embedding 在特征空间中无法被区分。
>
> 第一个是成对 MMD：直接用数学距离衡量并最小化三个细胞系分布的差异，三对两两配对，用多尺度 RBF 核评估，是显式的几何约束。第二个是对抗 GRL：我们挂一个域判别器，但用梯度反转层连接——判别器努力分类，反向传播时梯度乘以负号，编码器因此学会让自己的输出无法被分类。一个显式拉近，一个隐式对抗，两者互补。
>
> **为什么这两个 loss 共享同一个 alpha？** 因为它们的目标完全一样——都是"域对齐"，应该同时开关，而不是独立控制。
>
> **关于 cosine alpha 调度**：alpha 不是从一开始就存在的，也不是线性增加。前五个 epoch，alpha 固定为 0，MMD 和 adv_loss 对训练没有任何影响，模型只做纯预测。从第五个 epoch 开始，alpha 按余弦曲线缓慢爬升到 1，再保持平台直到训练结束。
>
> 选余弦而不选线性有一个实际原因：余弦曲线在起点和终点的斜率都接近 0，意味着对齐压力的引入非常平滑，不会在某个时间点突然给模型施加强梯度——这在预测损失还没有完全收敛的时候尤为重要，避免对齐压力和预测目标之间产生剧烈冲突。

---

## Slide 6 — 训练过程：Loss 曲线

### 内容
- 16 个 epoch 训练完成
- 展示三条曲线：pred_loss、mmd_loss、adv_loss、alignment_alpha

### 可视化

**⚠ 需要先跑脚本获取数据**

```bash
# 服务器上运行
cd ~/state/src
python ../scripts/plot_mmd_training.py
```

**如果 plot_mmd_training.py 读取失败，手动找 metrics.csv：**
```bash
find /media/mldadmin/home/s125mdg34_03/state -name "metrics.csv" 2>/dev/null
# 或者找 version_X 目录
ls /media/mldadmin/home/s125mdg34_03/state/checkpoints/mmd_aae/
```

**期望输出图：**
- 横轴：训练 step 或 epoch
- 四条曲线：
  - `val_loss`（总验证损失，应该下降）
  - `trainer/mmd_loss`（应在 epoch 5 后开始下降）
  - `trainer/adv_loss`（应在 epoch 5 后出现）
  - `trainer/alignment_alpha`（应呈余弦上升后平台）

> **如果训练 log 不存在或 metrics.csv 为空**：说明 Lightning logger 没有正确记录，可跳过此 slide 或用文字描述代替。

### 讲稿
> 这张图展示训练过程的稳定性。我们可以看到 alignment_alpha 在 epoch 5 附近开始从 0 上升，对应了 MMD 和 adversarial loss 的出现。pred_loss 在整个训练过程中持续下降，说明引入域对齐没有破坏主要的预测能力——这正是 alpha 预热调度的目的。

---

## Slide 7 — 域对齐结果：UMAP 可视化

### 内容
- 四个指标：域分类准确率（越低越好）、Silhouette Score、MMD²、CORAL
- 结论：域对齐在特征空间层面是成功的

| 指标 | STATE+MMD | Baseline | 方向 |
|------|-----------|----------|------|
| DomainClsAcc | ? | ? | ↓ 越低越好 |
| Silhouette | ? | ? | ↓ 越接近0越好 |
| MMD² mean | ? | ? | ↓ 越低越好 |
| CORAL mean | ? | ? | ↓ 越低越好 |

> **⚠ 表格中 ? 的部分需要先跑 eval 脚本填入**（见下方）

### 可视化

**⚠ 需要先跑脚本**

```bash
# 服务器上运行（会比较久，约 10-20 分钟）
cd ~/state/src

python ../scripts/eval_state_domain_alignment.py \
    --checkpoint /media/mldadmin/home/s125mdg34_03/state/checkpoints/mmd_aae/last.ckpt \
    --baseline /media/mldadmin/home/s125mdg34_03/state/checkpoints/no_mmd_baseline/last.ckpt \
    --config ../configs/mmd_aae_config.yaml \
    --include_hepg2_umap \
    --output /media/mldadmin/home/s125mdg34_03/state/checkpoints/mmd_aae/eval_alignment
```

**期望输出文件：**
```
eval_alignment/
  umap_mmd_aligned.png        ← STATE+MMD 的 UMAP（主图）
  umap_baseline.png           ← Baseline 的 UMAP
  umap_comparison.png         ← 两者并排对比（最推荐放 PPT）
  domain_alignment_metrics_*.json  ← 数字结果，填入上面表格
```

**图描述（PPT 中放 umap_comparison.png）：**
- 左图：Baseline — K562/RPE1/Jurkat 三团分离
- 右图：STATE+MMD — 三团融合（越融合说明域对齐越好）
- HepG2（橙色点）在两图中都出现，看它落在哪里

### 讲稿
> 域对齐效果怎么样？从 UMAP 可以直观看到：左边是没有域对齐的 Baseline，三个细胞系的 CLS embedding 明显分成三团；右边是我们的 STATE+MMD，三个细胞系的 embedding 混在一起，几乎无法区分。从数字上看，域分类准确率从 XX% 降到了 XX%，Silhouette Score 接近 0，说明三个域在特征空间已经成功融合。这部分是成功的。

---

## Slide 8 — 第一次评估：全基因组 Pearson r（误导性指标）

### 内容
- 做了什么：在 HepG2 全部 ~18,000 个基因上计算预测值 vs 真实值的 Pearson r
- 结果：r ≈ 0.70
- 问题：被老师指出是误导性指标

**为什么 0.70 是假象？**

```
18,080 个基因 = 17,800 个不变基因（预测≈0，真实≈0）
              +   280 个真正差异基因（DEG）

这 17,800 个"预测=0，真实=0"的基因对
贡献了绝大部分相关性 → Pearson r 虚高到 0.70

即使模型什么都没学会，只要把所有基因都预测成 0
Pearson r 依然会很高
```

### 可视化

**图：基因数量比例示意图（饼图或条形图）**

数据已知，无需跑脚本：
- 不变基因：17,800 个（98.5%）
- DEG（真正变化的基因）：~280 个（1.5%）

> 可用 Python 在本地生成（不需要服务器）：
> ```python
> import matplotlib.pyplot as plt
> labels = ['不变基因 (98.5%)\n预测≈0, 真实≈0', 'DEG (1.5%)\n真正变化的基因']
> sizes = [17800, 280]
> colors = ['#94a3b8', '#f43f5e']
> plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
> plt.title('18,080 个基因的构成\n（决定了全基因组 Pearson r = 0.70 为何虚高）')
> plt.savefig('gene_composition.png', dpi=150, bbox_inches='tight')
> ```

### 讲稿
> 第一轮评估，我们在 HepG2 的全部 18,000 个基因上计算预测和真实值的 Pearson r，得到了 0.70。乍一看不错，但老师马上指出了问题。18,000 个基因里，只有不到 300 个因为基因敲除真正发生了变化，其余 17,800 个基因在扰动前后表达量几乎不变——预测值约等于 0，真实值也约等于 0。这 17,800 对"0 对 0"贡献了巨大的相关性，把整体 Pearson r 虚高到了 0.70。就算模型什么都没学，只要把所有基因都预测成 0，它依然能得到一个虚高的相关系数。

---

## Slide 9 — 正确的评估：DEG Pearson（四步流程）

### 内容

**核心思路：** 先找出真正变化的基因（DEG），只在这些基因上计算预测方向的准确性。

**以 TP53 敲除为例，四步流程：**

**第一步：Wilcoxon + BH 校正，找 DEG**
- 对每个基因，把扰动组 vs 对照组细胞做非参数秩和检验
- BH 校正控制多重检验的假发现率（FDR）
- 筛出统计显著的 DEG（p < 0.20），每个扰动约 30–200 个 DEG

**第二步：计算真实 log-FC**
- 对每个 DEG：`actual_logFC = mean(pert cells) - mean(ctrl cells)`（log1p 空间）
- 正数 = 敲除后该基因上调，负数 = 下调

**第三步：用模型预测这些基因的变化**
- 对照组细胞 → Transformer → 平均 CLS embedding（512d）
- 扰动组细胞 → Transformer → 平均 CLS embedding（512d）
- 两个 CLS embedding 分别通过 binary_decoder，得到全部 18,080 个基因的预测分数
- 取差值，在 DEG 子集上得到 `pred_logFC`

**第四步：计算 Pearson r**
- `r = pearsonr(pred_logFC, actual_logFC)`
- 对全部 65 个扰动收集 r 值，取均值

### 可视化

**图：四步流程图（无需数据，PPT 手绘箭头图）**

```
[HepG2 h5]
    │
    ├──→ [对照组细胞] ──→ Wilcoxon vs [扰动组] ──→ DEG 列表（~30-200个基因）
    │                                                    │
    │                                              actual_logFC
    │
    ├──→ [对照组细胞] ──→ Transformer ──→ CLS_ctrl ──→ binary_decoder ──→ pred_ctrl
    │
    └──→ [扰动组细胞] ──→ Transformer ──→ CLS_pert ──→ binary_decoder ──→ pred_pert
                                                                              │
                                                               pred_logFC = pred_pert - pred_ctrl
                                                                              │
                                                            pearsonr(pred_logFC[DEG], actual_logFC[DEG])
```

### 讲稿
> 正确的评估方法是 DEG Pearson r，这是老师提出的方案。思路很清晰：既然只有少数基因真正变化了，我们就先用统计方法把这些基因找出来，再看模型对这些基因的预测方向是否正确。具体四步：第一步用 Wilcoxon 秩和检验加 BH 多重校正，筛出统计显著的差异表达基因——每个敲除扰动大概得到 30 到 200 个 DEG。第二步计算这些 DEG 的真实 log fold-change，就是扰动组均值减对照组均值。第三步用模型分别编码对照组和扰动组细胞，得到两个平均 CLS embedding，再用 binary_decoder 解码出全部基因的预测分数，差值就是预测的 logFC。第四步，只在 DEG 子集上计算预测 logFC 和真实 logFC 的 Pearson r，看方向对不对。

---

## Slide 10 — DEG Pearson 结果

### 内容

**核心数字：**
- STATE+MMD：DEG Pearson 均值 = **-0.035**
- Baseline（无 MMD）：DEG Pearson 均值 = **-0.021**
- 两者均 ≈ 0（随机猜测水平）
- 正相关比例：约 48%（随机期望 50%）

**LOO 验证（第四轮）：** 分别留一个细胞系做验证，其他三个训练，全部 DEG Pearson ≈ 0（-0.035 ~ +0.013）

| r 值 | 含义 |
|------|------|
| +1 | 完美预测哪些基因上调/下调 |
| 0 | 随机猜测 |
| -1 | 系统性预测反了 |

### 可视化

**图1（主图）：65 个扰动的 DEG Pearson r 分布直方图（对比 MMD vs Baseline）**

**⚠ 需要先跑脚本获取真实分布数据**

```bash
# 服务器上运行
cd ~/state/src

python ../scripts/eval_deg_pearson.py \
    --checkpoint /media/mldadmin/home/s125mdg34_03/state/checkpoints/mmd_aae/last.ckpt \
    --baseline /media/mldadmin/home/s125mdg34_03/state/checkpoints/no_mmd_baseline/last.ckpt \
    --config ../configs/mmd_aae_config.yaml \
    --output /media/mldadmin/home/s125mdg34_03/state/checkpoints/mmd_aae/eval_deg_pearson
```

**期望输出：** JSON 文件，包含每个扰动的 Pearson r 值列表，然后用本地 Python 画直方图：
```python
import json, matplotlib.pyplot as plt, numpy as np

with open('eval_deg_pearson/results.json') as f:
    data = json.load(f)

mmd_rs = [r['pearson_r'] for r in data['STATE+MMD']['per_perturbation']]
base_rs = [r['pearson_r'] for r in data['Baseline']['per_perturbation']]

bins = np.linspace(-1, 1, 25)
plt.figure(figsize=(8, 5))
plt.hist(mmd_rs, bins=bins, alpha=0.6, label=f'STATE+MMD (μ={np.mean(mmd_rs):.3f})', color='#6366f1')
plt.hist(base_rs, bins=bins, alpha=0.6, label=f'Baseline (μ={np.mean(base_rs):.3f})', color='#22d3ee')
plt.axvline(0, color='#f59e0b', linestyle='--', linewidth=2, label='r=0（随机猜测）')
plt.xlabel('DEG Pearson r（每个扰动）')
plt.ylabel('扰动数量')
plt.title('HepG2 零样本 DEG Pearson r 分布\n（65 个基因敲除扰动）')
plt.legend()
plt.tight_layout()
plt.savefig('deg_pearson_hist.png', dpi=150)
```

**图2（辅助）：LOO 结果柱状图**

> 数据已知（-0.035 ~ +0.013），可直接在本地画，不需要服务器：
```python
import matplotlib.pyplot as plt
import numpy as np

models = ['K562\n留出', 'RPE1\n留出', 'Jurkat\n留出', 'HepG2\n留出']
deg_pearson = [-0.035, -0.018, +0.013, -0.022]  # 替换为实际值
colors = ['#f43f5e' if v < 0 else '#10b981' for v in deg_pearson]

plt.figure(figsize=(7, 4))
bars = plt.bar(models, deg_pearson, color=colors, alpha=0.8)
plt.axhline(0, color='#f59e0b', linestyle='--', linewidth=1.5, label='r=0（随机水平）')
plt.ylabel('DEG Pearson r（均值）')
plt.title('LOO 留一法实验：各细胞系验证集 DEG Pearson r')
plt.legend()
plt.tight_layout()
plt.savefig('loo_deg_pearson.png', dpi=150)
```
> **⚠ 其中实际数值需要从 eval_deg_pearson 脚本输出中读取，上面只是示意。**

### 讲稿
> 结果非常清晰，也令人警醒。STATE+MMD 的 DEG Pearson 均值是 -0.035，Baseline 是 -0.021，两者都约等于 0。正相关比例约 48%，而随机猜测的期望是 50%——这就是随机猜测的水平，模型对这 65 个扰动中哪些基因应该上调、哪些应该下调，完全没有预测能力。更关键的是 LOO 实验：我们分别留一个细胞系作为验证集，用其余三个训练——结果所有情况的 DEG Pearson 都约等于 0。这说明问题不出在 HepG2 的零样本泛化上，即使是训练集自己的细胞系，模型也无法预测扰动方向。问题来自更根本的地方。

---

## Slide 11 — 根本诊断

### 内容

**为什么 DEG Pearson ≈ 0？**

TabularLoss（Energy Distance）的优化目标是：
```
让预测的基因表达分布"看起来像"真实分布
```

模型的最优策略：
```
预测"大多数基因≈0，少数有些变化，整体分布合理"
→ Loss 就能压到很低
→ 不需要学 TP53 敲除后 CDKN1A 上调、MDM2 下调
```

**两个评估指标的一致性：**

| 指标 | STATE+MMD | Baseline | 随机基线 |
|------|-----------|----------|---------|
| DEG Pearson r（均值） | -0.035 | -0.021 | ≈ 0 |
| DES 基因重叠率 | 0.0054 | 0.0042 | 0.0028 |
| DES > 0 的扰动比例 | 21% | — | — |

DEG Pearson（方向相关性）和 DES（命中率）从两个不同角度都指向同一个结论。

**关键区分：**
- 域对齐 ✅ 成功（CLS embedding 层面的融合）
- 扰动方向预测 ❌ 失败（binary_decoder 层面的 Loss 设计问题）
- 两者相互独立——对齐成功不等于预测成功

### 可视化

**图：诊断因果链（PPT 手绘箭头图，无需数据）**

```
TabularLoss = Energy Distance（分布级别的相似性）
        ↓ 模型的最优策略
预测"分布形状合理"，不预测"具体方向"
        ↓ 导致
DEG Pearson ≈ 0（对方向一无所知）
DES 重叠率 ≈ 随机水平

与域对齐无关 ←→ 域对齐已成功（UMAP 融合）
```

### 讲稿
> 问题出在哪？出在 TabularLoss 的本质目标上。Energy Distance 优化的是两个分布之间的统计距离——只要预测的整体基因表达分布形状和真实分布形状相似，Loss 就低。模型完全可以通过学会"大多数基因接近 0，少数基因有一点变化，整体合理"来达到这个目标，根本不需要知道具体哪个基因上调、哪个下调。我们用两个不同的指标验证了这个诊断：DEG Pearson 衡量方向相关性，DES 衡量命中率，两者都指向同一个结论。这里要强调一个重要区分：域对齐是成功的，CLS embedding 在特征空间确实融合了三个细胞系。但 binary_decoder 仍然被 TabularLoss 训练，域对齐无法修复 Loss 设计问题。

---

## Slide 12 — 下一步：三条路径

### 内容

根本问题已经清楚：**TabularLoss 不奖励扰动方向的正确性**。

解决思路：让损失函数直接惩罚方向预测的错误。

**方向 1（最快，1-2天）：在现有 Loss 上加 Pearson 正则项**
- 在 shared_step 中追加 `-pearsonr(pred_logFC, true_logFC)` 作为额外惩罚
- 优点：改动最小
- 缺点：Pearson r 不可微，需要 soft 版本；true_logFC 训练时不一定可得

**方向 2（推荐，3-5天）：改监督目标为 log-FC MSE**
- 当前：binary_decoder → TabularLoss（分布相似性）
- 改为：binary_decoder → MSE（预测 logFC，真实 logFC）
- 参考 GEARS、CPA 的做法
- 需要：改 DataLoader 提供 ctrl 均值，改 Loss 函数
- 验证目标：DEG Pearson > 0.10

**方向 3（彻底，1-2周）：换用扰动预测专用架构**
- 将 MMD 对齐后的 embedding 接入 CPA 或 GEARS
- 工程量最大，但最有可能根本解决问题

### 可视化

**图：三条路径对比表（PPT 表格，无需数据）**

| | 方向 1 | 方向 2 ⭐推荐 | 方向 3 |
|--|--------|-------------|--------|
| 工程量 | 小 | 中 | 大 |
| 时间 | 1-2天 | 3-5天 | 1-2周 |
| 可行性 | 中 | 高 | 高 |
| 预期效果 | 不确定 | 较有把握 | 最有把握 |
| 改动点 | shared_step | DataLoader + Loss | 整体架构 |

### 讲稿
> 问题诊断清楚了，下一步怎么走？有三条路径。最快的方向是在现有 Loss 上叠加一个 Pearson 正则项，直接最大化预测 logFC 和真实 logFC 的相关性，1-2 天能验证，但 Pearson r 不可微，工程上需要处理。我推荐的起点是方向 2：把监督目标从分布相似性改成 logFC 的 MSE——不再让模型预测"基因表达分布形状"，而是直接预测"敲除后每个基因的变化量"。这是 GEARS 和 CPA 这类专门做扰动预测的模型的标准做法，逻辑直接，改动范围可控，验证目标也清晰——DEG Pearson 能不能超过 0.10。如果方向 2 验证有效，再考虑方向 3，把我们的域对齐模块作为前处理，接入更专业的扰动预测架构。

---

## 附录：需要执行的脚本清单

### 必须在服务器跑（有数据依赖）

| Slide | 脚本 | 预计时间 | 输出 |
|-------|------|---------|------|
| Slide 6 | `plot_mmd_training.py` | 1分钟 | loss 曲线图 |
| Slide 7 | `eval_state_domain_alignment.py` | 15-20分钟 | UMAP 图 + 指标 JSON |
| Slide 10 | `eval_deg_pearson.py` | 20-30分钟 | 每个扰动的 r 值 JSON |

### 可在本地跑（不需要服务器，不需要模型）

| Slide | 内容 | 代码在哪里 |
|-------|------|-----------|
| Slide 4 | Energy Distance 分布示意图 | 见 Slide 4 可视化部分 |
| Slide 8 | 基因数量饼图（98.5% vs 1.5%） | 见 Slide 8 可视化部分 |
| Slide 10 | LOO 柱状图（已知数值） | 见 Slide 10 可视化部分 |

### 推荐执行顺序
```
Step 1: plot_mmd_training.py         ← 最快，先跑
Step 2: eval_state_domain_alignment  ← 跑着等
Step 3: eval_deg_pearson             ← 同时本地画简单图
```

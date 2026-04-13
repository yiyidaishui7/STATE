# Differential Gene Expression (DGE) 分析详解

> 参考资料：
> - [sc-best-practices DGE 章节](https://www.sc-best-practices.org/conditions/differential_gene_expression.html)
> - [GGE 评估框架论文 arXiv:2603.11244](https://arxiv.org/abs/2603.11244)（ICLR 2026 Gen² Workshop）

---

## 第一部分：从零理解——DGE 是什么，为什么要做

### 用一句话说

**DGE（Differential Gene Expression，差异基因表达分析）就是：找出在两组细胞之间，哪些基因的表达量有显著差异。**

### 生物学背景（类比解释）

想象一个工厂，有 18,080 条流水线，每条流水线生产一种蛋白质（对应一个基因）。

- **正常工厂（对照组，non-targeting）**：每条流水线按正常速度运转
- **敲除 TP53 后的工厂（扰动组）**：TP53 这条流水线被关掉了，但这会导致一连串连锁反应——有些流水线因此加速（上调），有些减速（下调）

DGE 的任务就是：**在 18,080 条流水线里，找出哪些流水线的速度在两种状态下有显著变化。**

### 在我们项目里的具体位置

```
真实数据（HepG2 / K562 / RPE1 / Jurkat）
         ↓
  每个扰动 vs non-targeting
         ↓ DGE 分析（Wilcoxon + BH）
   DEG 列表（"参考答案"：真正变化的基因）
         ↓
  ┌──────────────────┬──────────────────┐
  │   DEG Pearson r  │      DES         │
  │  模型预测方向    │  top-k 命中率    │
  │  是否与真实一致？│  有多少基因猜对？│
  └──────────────────┴──────────────────┘
```

DGE 是评估的"地基"——它告诉我们答案是什么，然后我们才能评估模型有多接近答案。

---

## 第二部分：数据结构

以 HepG2 为例（K562、RPE1、Jurkat 结构完全相同）：

```
adata.X             : (n_cells, 18080)  — 每个细胞在每个基因上的表达量（log1p 归一化）
adata.obs['gene']   : 每个细胞的身份标签，例如：
    "non-targeting"  → 正常对照细胞（~5000 个）
    "TP53"           → TP53 被敲除的细胞（约 50~200 个）
    "CDKN1A"         → CDKN1A 被敲除的细胞
    ...              → HepG2 共 68 种敲除扰动
adata.var_names     : 18080 个基因的名称列表
```

**HepG2 的数据可以理解为一张大表：**

```
            基因1   基因2   基因3  ...  基因18080
细胞1(ctrl)  0.0     2.3     0.0  ...    1.2
细胞2(ctrl)  0.0     2.1     0.1  ...    0.9
细胞3(TP53)  0.0     3.5     0.0  ...    0.8    ← TP53 被敲除
细胞4(TP53)  0.1     3.8     0.0  ...    0.7    ← TP53 被敲除
...
```

每种扰动和对照组之间做一次 DGE，HepG2 共需 **68 次独立分析**。

---

## 第三部分：为什么需要统计检验？

### 直接比较均值不行吗？

直觉上，我们可以直接算：

```
真实变化 = 扰动组平均值 - 对照组平均值
```

但这有个问题：**细胞本身就有很大的随机噪声。**

类比：你想知道"喝咖啡的人"比"不喝咖啡的人"打字速度快不快，于是各测了 5 个人。结果喝咖啡的人平均比不喝咖啡的人快 2 字/分钟。但这个差异是真实的效果，还是纯属运气选到了 5 个快手？

统计检验就是用数学回答这个问题：**这个差异有多大概率是随机噪声造成的？**

### p-value 是什么

p-value = "假设两组根本没有区别，观察到当前这么大差异的概率"

- p = 0.001：只有 0.1% 的概率是随机，说明差异很可能是真实的
- p = 0.8：有 80% 可能是随机噪声，不可信
- 通常 p < 0.05 认为显著

---

## 第四部分：Wilcoxon Rank-Sum Test

### 为什么不用 t 检验？

t 检验要求数据近似正态分布。但单细胞 RNA-seq 数据不满足：

| scRNA-seq 的特点 | 导致的问题 |
|----------------|-----------|
| 大量基因表达量为 0（dropout） | 分布严重偏斜，不是正态 |
| 每个扰动细胞数很少（50~200） | 小样本 t 检验不准 |
| 不同基因方差差异极大 | t 检验假设方差相等 |

### Wilcoxon 的核心思想（类比解释）

Wilcoxon 不直接比数值，只比**名次（秩）**。

**举例**：比较 TP53 敲除 vs 对照，某基因的表达量：

```
对照组:  0.0, 0.0, 0.1, 0.2, 1.5
扰动组:  0.0, 0.3, 0.8, 1.2, 2.0
```

把所有数值混在一起排名（0.0=1, 0.0=1, 0.0=1, 0.1=4, 0.2=5, 0.3=6, 0.8=7, 1.2=8, 1.5=9, 2.0=10）：

- 对照组的名次总和 = 1+1+4+5+9 = 20
- 扰动组的名次总和 = 1+6+7+8+10 = 32

扰动组的名次普遍更靠后（值更大），说明该基因在敲除后上调。

**Wilcoxon 检验问题**：如果两组来自同一分布，名次总和应该差不多。观察到的差异有多大概率是随机的？

**优点**：
- 不关心具体数值，只关心大小关系 → 对 dropout 的 0 值鲁棒
- 不假设分布形态 → 适合任何形状的数据

---

## 第五部分：Benjamini-Hochberg 多重检验校正

### 为什么需要校正？

我们同时对 **18,080 个基因**做检验。

假设没有任何基因真的差异表达，用 p < 0.05 做标准：
```
18,080 × 0.05 = 904 个"显著"基因（全是假阳性！）
```

这就像买彩票——一个人买 1 张中奖概率是 5%，买 18,080 张几乎必然中一张，但这不代表你"运气好"。

### BH 校正怎么做？（简化版）

1. 把所有 18,080 个 p-value 从小到大排序
2. 对排名第 i 的基因，计算调整后的 q-value = p(i) × 18080 / i
3. 只保留 q-value < 0.05 的基因

**效果**：在所有筛选出的 DEG 里，期望最多 5% 是假阳性。

```
原始 p-value  → BH 校正后 adj p-value（q-value）
p = 0.0001   → q ≈ 0.0001 × 18080 / 1 = 1.808  (如果只有它显著，q > 0.05，被过滤)
p = 0.0001   → 如果有 1000 个基因都显著，q = 0.0001 × 18080/1000 = 0.0018  (保留)
```

---

## 第六部分：完整代码——单个扰动的 DGE

以比较 TP53 敲除 vs 对照为例：

```python
import anndata as ad
import scanpy as sc
import numpy as np

# ────────────────────────────────────────────
# 步骤 1：分开对照组和扰动组，贴上标签
# ────────────────────────────────────────────
pert_gene = "TP53"

ctrl_mask = adata.obs['gene'] == 'non-targeting'
pert_mask = adata.obs['gene'] == pert_gene

adata_control   = adata[ctrl_mask].copy()
adata_predicted = adata[pert_mask].copy()

adata_control.obs['group']   = 'control'    # 告诉 scanpy 这是对照
adata_predicted.obs['group'] = 'perturbed'  # 告诉 scanpy 这是扰动

print(f"对照组细胞数: {ctrl_mask.sum()}")   # 约 5000
print(f"扰动组细胞数: {pert_mask.sum()}")   # 约 50~200

# ────────────────────────────────────────────
# 步骤 2：合并，做 Wilcoxon 检验
# ────────────────────────────────────────────
adata_combined = ad.concat([adata_control, adata_predicted])

sc.tl.rank_genes_groups(
    adata_combined,
    groupby="group",                   # 按 group 列区分两组
    groups=["perturbed"],              # 找 perturbed 里显著的基因
    reference="control",               # 以 control 为参照
    method="wilcoxon",                 # 用 Wilcoxon 秩和检验
    corr_method="benjamini-hochberg",  # BH 校正
    tie_correct=True,                  # 处理大量 0 值造成的同秩问题（重要！）
    use_raw=False,                     # 用 adata.X（已 log1p）
)

# ────────────────────────────────────────────
# 步骤 3：提取显著 DEG
# ────────────────────────────────────────────
de_genes = sc.get.rank_genes_groups_df(
    adata_combined,
    group='perturbed',
    pval_cutoff=0.05,    # 只保留 adj p < 0.05
)

print(f"\n显著 DEG 数量: {len(de_genes)}")
print(de_genes.head(10))
```

**输出的 DataFrame 长这样：**

| names | scores | pvals | pvals_adj | logfoldchanges |
|-------|--------|-------|-----------|----------------|
| CDKN1A | 8.23 | 2e-15 | 1e-12 | 1.42 |（上调）|
| MDM2 | 6.71 | 4e-11 | 3e-8 | 0.87 |（上调）|
| BAX | 5.12 | 8e-8 | 0.002 | 0.63 |（上调）|
| ... | ... | ... | ... | ... |

其中 `logfoldchanges` 是 log2 fold-change：
- 正数 = 敲除后该基因表达量上升（上调）
- 负数 = 敲除后该基因表达量下降（下调）

---

## 第七部分：批量处理所有扰动

实际评估中对每个扰动（HepG2 共 68 个）逐一做 DGE：

```python
import pandas as pd

def run_dge_for_all_perturbations(
    adata,
    pert_col="gene",
    ctrl_label="non-targeting",
    pval_cutoff=0.05,
    min_pert_cells=5,
    max_ctrl_cells=500,
):
    pert_labels = adata.obs[pert_col].astype(str).values
    all_perts   = sorted(set(pert_labels) - {ctrl_label})

    # 对照组（取最多 500 个，避免计算太慢）
    ctrl_indices = np.where(pert_labels == ctrl_label)[0]
    if len(ctrl_indices) > max_ctrl_cells:
        ctrl_indices = np.random.choice(ctrl_indices, max_ctrl_cells, replace=False)

    adata_ctrl = adata[ctrl_indices].copy()
    adata_ctrl.obs['group'] = 'control'

    results = {}
    summary_rows = []

    for pert in all_perts:
        pert_indices = np.where(pert_labels == pert)[0]
        if len(pert_indices) < min_pert_cells:
            continue   # 细胞太少，跳过

        adata_pert = adata[pert_indices].copy()
        adata_pert.obs['group'] = 'perturbed'
        adata_combined = ad.concat([adata_ctrl, adata_pert])

        sc.tl.rank_genes_groups(
            adata_combined, groupby="group", groups=["perturbed"],
            reference="control", method="wilcoxon",
            corr_method="benjamini-hochberg", tie_correct=True, use_raw=False,
        )
        de_df = sc.get.rank_genes_groups_df(
            adata_combined, group="perturbed", pval_cutoff=pval_cutoff,
        )

        n_up   = (de_df['logfoldchanges'] > 0).sum()
        n_down = (de_df['logfoldchanges'] < 0).sum()
        results[pert] = de_df

        summary_rows.append({
            "perturbation": pert,
            "n_pert_cells": len(pert_indices),
            "n_deg_total":  len(de_df),
            "n_deg_up":     n_up,
            "n_deg_down":   n_down,
        })
        print(f"  {pert:20s}  cells={len(pert_indices):4d}  DEG={len(de_df):4d}  up={n_up}  down={n_down}")

    return results, pd.DataFrame(summary_rows)

# 使用：
results, summary = run_dge_for_all_perturbations(adata)
```

---

## 第八部分：DEG 如何用于评估模型

### 8.1 为什么不直接用全基因组 Pearson r？

**这是老师否定第一轮实验（r=0.70）的核心原因。**

类比：你让一个算法预测一栋楼里哪些房间的温度异常。楼里有 18,080 个房间，只有 50 个房间（DEG）真的异常，其余 18,030 个都是正常的（≈0）。

- 算法说："我预测所有房间都正常（全输出 ≈ 0）"
- 真实情况：18,030 个正常房间预测对了，50 个异常的完全没有预测对
- 全体 Pearson r ≈ 0.70（因为 18,030 个正常的贡献了虚高的相关性）

这个 0.70 **没有任何意义**，因为算法根本没有学到"哪些房间异常"。

**解决方案**：只在那 50 个异常房间（DEG）上评估，这才是真正检验算法能力的地方。

---

### 8.2 DEG Pearson r（正确的评估方式）

**GGE 论文（arXiv:2603.11244）给出的标准公式（Equation 1）：**

```
μ_effect = corr(μ_real - μ_ctrl,   μ_gen - μ_ctrl)
           ↑                         ↑
    真实变化量（真实扰动组 - 对照）  预测变化量（预测扰动组 - 对照）
```

关键点：**correlate 的是变化量，而不是绝对表达量。**

在我们项目的实际代码（`eval_deg_pearson.py`）里：

```python
# 真实变化量（只在 DEG 子集上）
actual_logfc = adata.X[pert_indices][:, de_gene_indices].mean(axis=0) \
             - actual_ctrl_mean_all[de_gene_indices]    # μ_real - μ_ctrl

# 预测变化量（STATE 模型解码器输出）
pred_logfc = pred_pert_all[de_gene_indices] \
           - pred_ctrl_all[de_gene_indices]             # μ_gen - μ_ctrl

# Pearson r
r, pval = pearsonr(pred_logfc, actual_logfc)
```

**r 的含义：**

| r 值 | 含义 | 类比 |
|------|------|------|
| r = +1 | 完美预测：该上调的预测上调，该下调的预测下调 | 满分答卷 |
| r = +0.6 | 好的模型（GEARS、scGen 能达到的水平） | 大部分答对 |
| r = 0 | 随机猜测：预测方向与真实无关 | 乱写答案 |
| r = -1 | 系统性预测反了 | 全部反着写 |

**我们项目的结果（HepG2 零样本）：**
- STATE+MMD：mean r = **-0.035 ≈ 0**
- Baseline：mean r = **-0.021 ≈ 0**
- 正相关扰动比例 ≈ 48%（随机猜测期望值 50%）

**诊断**：TabularLoss 只学了"分布的形状"，没有学哪个基因该涨哪个该跌。

---

### 8.3 DES：top-k 基因重叠率

另一个维度的评估：预测变化最大的 top-50 基因里，有多少是真正的 DEG？

```python
k = 50
# 真实：DEG 里变化最大的 top-50
top_k_real = set(de_df.nlargest(k, 'logfoldchanges', keep='first')['names'])

# 预测：模型认为变化最大的 top-50
pred_logfc_all = pred_pert_all - pred_ctrl_all           # 全部 18080 基因
top_k_pred     = set(adata.var_names[np.argsort(np.abs(pred_logfc_all))[-k:]])

DES = len(top_k_real & top_k_pred) / k
# 随机基线 = 50 / 18080 ≈ 0.0028（随机挑 50 个，平均能命中 2.8 个真实 DEG）
```

**我们项目的结果：**
- STATE+MMD：mean DES = **0.0054**（100 个中命中 0.54 个）
- 随机基线 = **0.0028**
- median = 0（大多数扰动完全没命中）

---

## 第九部分：sc-best-practices 对 Wilcoxon 的警告

sc-best-practices 官方**不推荐 Wilcoxon** 作为首选方法，推荐 **pseudobulk 方法（edgeR、DESeq2）**。

### 为什么？伪重复问题（Pseudoreplication Problem）

想象一个临床试验：

```
❌ 错误做法（伪重复）：
  3 个病人各取 500 个细胞 → 1500 个"独立观测"
  → Wilcoxon → p 极小 → 虚假的强显著性

✅ 正确做法（pseudobulk）：
  3 个病人各聚合成 1 个样本 → 3 个真实独立重复
  → edgeR/DESeq2 → 统计上正确
```

同一个人的 500 个细胞不是"500 个独立实验"，它们来自同一个生物个体，天然相关。把它们当作独立样本，会严重膨胀样本量，导致假阳性爆炸。

### 但为什么我们用 Wilcoxon 还是合理的？

我们做的是 **CRISPR perturbation screen**，不是临床试验：

```
CRISPR 实验设计：
  一个培养皿里有 10,000 个 HepG2 细胞
  用 CRISPR 把其中一些细胞的 TP53 敲掉
  → "TP53 敲除细胞" 和 "non-targeting 细胞" 在同一个实验里
```

这里没有"多个独立的生物学样本"——细胞本身就是"重复单元"。这种场景下，整个领域（GEARS、scGen、CPA、STATE 的原始论文）都用 Wilcoxon，是公认的做法。

**一句话总结**：pseudobulk 适合"多个病人/多个小鼠"的设计；Wilcoxon 适合"一个实验里的 CRISPR 筛选"。我们属于后者。

---

## 第十部分：GGE 论文的核心发现——评估标准混乱问题

### 10.1 调查结论：12 篇论文，没有两篇用同一套评估

GGE 论文作者调查了 12 个主流单细胞生成模型，发现一个严重问题：

> **同一个指标名字（如"Wasserstein distance"）在不同论文里的计算方法完全不同。**

**实验结果（表 2）**：完全相同的数据，W₂ 距离根据计算空间不同相差 **6 倍**：

| 计算空间 | W₂ 值 |
|---------|------|
| Raw 基因空间（18000维） | 104.3 |
| PCA-100 空间 | 53.8 |
| PCA-50 空间 | 33.6 |
| PCA-25 空间 | 17.2 |

同一份数据，A 论文报 "W₂ = 17.2"，B 论文报 "W₂ = 104.3"，都叫 "Wasserstein distance"，**根本无法比较**。

### 10.2 三种计算空间的含义

| 空间 | 怎么做 | 适合什么 | 缺点 |
|------|-------|---------|------|
| **Raw** | 直接用 18000 维原始表达矩阵 | 想要基因级别可解释性 | 高维噪声主导；dropout 干扰 |
| **PCA-50** | 先降维到 50 维再计算 | 主要分布评估（推荐） | 可能低估罕见但重要的扰动基因 |
| **DEG-restricted** | 只在差异表达基因上算 | 生物学验证 | 依赖 DEG 选择；小样本不稳定 |

**GGE 的建议**：三种空间都用，PCA-50 作为主要指标，DEG-restricted 作为生物学验证。

### 10.3 DEG 选择方式的影响（表 3，Norman 数据集，138 个扰动）

不同的 DEG 选法，Pearson r 数值差异显著：

| 选择方式 | 平均 DEG 数 | Pearson r |
|---------|-----------|----------|
| Top-20（GEARS 的做法） | 20 | 0.614 ± 0.066 |
| Top-100（scGen 的做法） | 100 | 0.594 ± 0.024 |
| 严格阈值：lfc>1, p<0.01 | 15.3 ± 5.1 | **0.506 ± 0.217**（方差最大！）|
| 宽松阈值：lfc>0.25, p<0.1 | 71.7 ± 6.9 | 0.622 ± 0.079 |

**我们项目用 adj p<0.05 属于"严格阈值"类**，方差最大（±0.217）——在细胞数少的扰动条件下，DEG 数量不稳定（有时 0 个，有时很多），导致评估数字波动很大。

**参考值**：好的模型（GEARS、scGen、CPA）在 Norman 数据集上 DEG Pearson ≈ 0.6。**我们目前是 ≈ 0**，确认是 TabularLoss 根本问题，而非评估方式问题。

### 10.4 Top-N 选法 vs 阈值选法的取舍

```
Top-N 选法（如 top-20）：
  ✅ 每个扰动始终有固定数量的 DEG → 跨条件可比
  ✅ 方差小，结果稳定
  ❌ 即使一个基因没怎么变，也会被选进来

阈值选法（如 adj p<0.05）：
  ✅ 自适应——效果强的扰动（DEG 多）和弱的（DEG 少）都能反映
  ✅ 生物学上更有意义
  ❌ 方差大——细胞少时可能一个 DEG 都没有，无法算 Pearson r
```

**可以改进的方向**：在现有 adj p<0.05 基础上，同时报告 top-20 和 top-100 的结果，与 GEARS、scGen 等模型做可比性对比。

---

## 第十一部分：项目代码中的实际实现

项目里的 DGE 调用在 [scripts/eval_deg_pearson.py:363-388](scripts/eval_deg_pearson.py#L363-L388)，与上述代码的对应关系：

```python
# 实际代码（eval_deg_pearson.py）
adata_ctrl.obs["_group"] = "control"    # 注意：用 "_group"（加下划线）
adata_pert.obs["_group"] = "perturbed"
adata_combined = ad.concat([adata_ctrl, adata_pert])

sc.tl.rank_genes_groups(
    adata_combined,
    groupby="_group",           # ← 下划线版本
    groups=["perturbed"],
    reference="control",
    method="wilcoxon",
    corr_method="benjamini-hochberg",
    tie_correct=True,
    use_raw=False,
)
de_df = sc.get.rank_genes_groups_df(
    adata_combined,
    group="perturbed",
    pval_cutoff=args.pval_cutoff,   # 默认 0.05
)
```

**两个工程优化（循环外提前计算，只算一次）：**

```python
# 对照组 CLS embedding 均值（所有扰动共用同一个对照）
ctrl_cls = encode_cells_to_cls(model, cfg, h5_path, ctrl_indices, args, device)
ctrl_cls_mean = torch.tensor(ctrl_cls.mean(axis=0), ...).unsqueeze(0)

# 对照组全基因预测分数（也只算一次）
pred_ctrl_all = predict_from_emb(model, ctrl_cls_mean, gene_embs, ...)
# pred_ctrl_all: (18080,)，循环里每个扰动都用这个减
```

然后对每个扰动，只需要额外算扰动组的 embedding 和预测，再取差值在 DEG 子集上算 Pearson r。

---

## 总结

| 步骤 | 做了什么 | 为什么 |
|------|---------|--------|
| 1 | 把细胞分成 control 和 perturbed 两组 | 明确比较对象 |
| 2 | Wilcoxon 秩和检验（非参数） | scRNA-seq 数据非正态，dropout 多 |
| 3 | BH 校正 | 18,080 次同时检验会产生大量假阳性 |
| 4 | 筛 adj p < 0.05 → DEG 列表 | 只关注真正变化的基因 |
| 5 | 在 DEG 上算 Pearson r（变化量 vs 变化量） | 检验模型预测方向是否正确 |
| 6 | 在 DEG 上算 DES（top-k 命中率） | 检验模型是否找到了对的基因 |

**核心结论**：
- DGE 是"参考答案"，只依赖真实数据，与模型无关
- 全基因组 Pearson r 是误导性指标，被大量不变基因虚高（sc-best-practices 和 GGE 论文都指出了这点）
- DEG Pearson r 才是真正检验模型能力的指标（好模型目标是 r ≈ 0.6，我们目前 ≈ 0）
- 我们的 r ≈ 0 是 TabularLoss 的根本问题，不是 DGE 分析方法的问题

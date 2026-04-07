# STATE + MMD-AAE 实验总结

**更新日期：** 2026-04-07  
**硬件：** A100 GPU (24GB VRAM), 服务器 gpu34  
**代码仓库：** yiyidaishui7/STATE (main branch)

---

## 一、实验目的

在 STATE 单细胞预训练模型基础上，加入 MMD-AAE 域对齐模块，目标：

1. **消除细胞系特异性偏差**：让 K562/RPE1/Jurkat 的 CLS embedding 分布对齐
2. **提升零样本泛化**：对未见过的 HepG2 细胞系做零样本预测时表现更好
3. **验证 cosine α schedule**：替代旧版 epoch 级阶梯 α，实现平滑连续的对齐权重变化

---

## 二、训练了哪些模型

### 2.1 三个主要模型

| 模型名 | Version | Config | Checkpoint | 状态 |
|--------|---------|--------|------------|------|
| No-MMD Baseline | version_11 | `no_mmd_baseline_config.yaml` | `checkpoints/no_mmd_baseline/last.ckpt` | 完成 |
| STATE+MMD (staircase α) | version_12 | `mmd_aae_config.yaml` | `checkpoints/mmd_aae/last.ckpt` | 完成 |
| STATE+MMD (cosine α) | version_14 | `mmd_aae_cosine_config.yaml` | `checkpoints/mmd_aae_cosine/last.ckpt` | 完成 |

所有模型：`max_epochs=16`，`batch_size=48`，在 K562/RPE1/Jurkat 三域训练，HepG2 作零样本验证域。

### 2.2 训练命令（服务器 ~/state/src 目录）

```bash
# Baseline（无 MMD）
CUDA_VISIBLE_DEVICES=1 python train.py --config ../configs/no_mmd_baseline_config.yaml \
    > ~/state/logs/no_mmd_baseline.log 2>&1 &

# MMD staircase
CUDA_VISIBLE_DEVICES=0 python train.py --config ../configs/mmd_aae_config.yaml \
    > ~/state/logs/mmd_aae.log 2>&1 &

# MMD cosine
CUDA_VISIBLE_DEVICES=1 python train.py --config ../configs/mmd_aae_cosine_config.yaml \
    > ~/state/logs/mmd_aae_cosine.log 2>&1 &
```

### 2.3 LOO Leave-One-Out 实验（全部 error，暂搁置）

设计了 3 个 LOO 配置，各留出一个细胞系做验证：

| Config | Train 域 | Val 域 | Port |
|--------|----------|--------|------|
| `loo_jurkat_config.yaml` | K562, RPE1, HepG2 | jurkat.h5 | 12403 |
| `loo_rpe1_config.yaml` | K562, Jurkat, HepG2 | rpe1.h5 | 12404 |
| `loo_k562_config.yaml` | RPE1, Jurkat, HepG2 | k562.h5 | 12405 |

当前状态：全部训练报错（具体错误待查）。

---

## 三、总损失函数设计

```
total_loss = pred_loss + α × (2.0 × mmd_loss + 0.1 × adv_loss)
```

| 组件 | 含义 | 权重 |
|------|------|------|
| `pred_loss` | TabularLoss：预测基因表达 vs 真实表达的能量距离 | 1.0（固定） |
| `mmd_loss` | 多尺度 RBF-MMD：K562/RPE1/Jurkat CLS embedding 成对分布差异 | 2.0 × α |
| `adv_loss` | GRL 对抗判别器：交叉熵损失，反向传播时梯度取反 | 0.1 × α |
| `α` | 对齐权重调度器，训练初期为 0，逐步升至 1.0 | 动态变化 |

### TabularLoss 详解

```python
# 两个能量距离之和
gene_mmd = EnergyDistance(pred_全基因组_分布, target_全基因组_分布)
cell_mmd = EnergyDistance(pred_shared基因^T, target_shared基因^T)
total = gene_mmd + cell_mmd
```

- **`gene_mmd`**：把每个细胞的全基因预测视为一个点，衡量预测分布 vs 真实分布的差异
- **`cell_mmd`**：转置后，以每个基因在不同细胞上的分布来衡量差异（只用最后 128 个 shared 基因）
- 用 `SamplesLoss("energy")`（能量距离），是 Wasserstein 距离的近似，比 MSE 更鲁棒

**关键点**：TabularLoss 优化的是分布层面的相似性，不是 per-cell Pearson r，因此 Pearson r 不是直接优化目标。

### α Schedule 对比

**旧版（staircase）**：
```python
# epoch 级离散跳变
if epoch < 5: α = 0.0          # warmup 期
elif epoch < 10: α = (epoch-5)/5  # 线性 ramp（每个 epoch 突变一次）
else: α = 1.0
```

**新版（cosine）**：
```python
# global_step 级连续 S 型曲线
warmup_steps = total_steps * 5/16
ramp_steps   = total_steps * 5/16
if step < warmup_steps: α = 0.0
else:
    progress = min((step - warmup_steps) / ramp_steps, 1.0)
    α = 0.5 * (1 - cos(π × progress))   # S 型 cosine
```

为什么用 cosine：staircase 在每个 epoch 边界有突变（梯度冲击），cosine 提供平滑连续的变化。

---

## 四、评估脚本

### 4.1 域对齐评估

**脚本**：`scripts/eval_state_domain_alignment.py`

**运行方式**：
```bash
python ../scripts/eval_state_domain_alignment.py \
    --checkpoint /media/.../checkpoints/<model>/last.ckpt \
    --config ../configs/<config>.yaml \
    --output /media/.../src/lightning_logs/<version>/eval_alignment
```

**输出**：
- `domain_alignment_metrics_<timestamp>.json`：含 4 个指标
- `umap_mmd_aligned.png`：UMAP 可视化图

**指标含义**：

| 指标 | 计算方式 | 解读 |
|------|----------|------|
| `domain_cls_acc` | 线性探针分类 K562/RPE1/Jurkat | 越低 = 域信息越少保留（对齐越好） |
| `silhouette` | CLS embedding 的 Silhouette Score | 越低 = 各域 cluster 越混在一起（对齐越好） |
| `mmd_mean` | 三对域之间 MMD² 的均值 | 越低 = 分布越相似（对齐越好） |
| `coral` | 二阶矩（协方差）差异 | 越低越好 |

### 4.2 HepG2 Pearson r 评估（旧版全基因）

**脚本**：`scripts/eval_hepg2_pearson.py`

**运行方式**：
```bash
python ../scripts/eval_hepg2_pearson.py \
    --checkpoint /media/.../checkpoints/<model>/last.ckpt \
    --config ../configs/<config>.yaml \
    --hepg2 /media/.../competition_support_set/hepg2.h5 \
    --output /media/.../src/lightning_logs/<version>/eval_pearson
```

**输出**：`pearson_results_<timestamp>.json`

**Mode A（细胞级）**：对每个 HepG2 细胞，encode → decode，预测分数 vs 实际 log1p(counts) 的 Pearson r。反映零样本重建质量。

### 4.3 DEG Pearson r 评估（谢老师新要求）

**脚本**：`scripts/eval_deg_pearson.py`

**运行方式**：
```bash
python ../scripts/eval_deg_pearson.py \
    --checkpoint /media/.../checkpoints/mmd_aae_cosine/last.ckpt \
    --baseline   /media/.../checkpoints/no_mmd_baseline/last.ckpt \
    --config     ../configs/mmd_aae_cosine_config.yaml \
    --h5ad       /media/.../competition_support_set/hepg2.h5 \
    --pert_col   gene \
    --ctrl_label non-targeting
```

**方法**（谢老师指定）：
```python
# 对每个扰动条件（基因敲除），vs non-targeting 对照
sc.tl.rank_genes_groups(adata_combined, groupby="_group",
    method="wilcoxon", corr_method="benjamini-hochberg")
de_df = sc.get.rank_genes_groups_df(..., pval_cutoff=0.05)
# 只在 adj-p < 0.05 的 DE 基因上计算 Pearson r（预测 vs 实际 log-FC）
```

**与旧 Mode B 的区别**：旧版用 |log_fc| 排序取 top-50，无统计显著性。新版用 Wilcoxon + BH 校正，只保留真正显著的 DE 基因，避免全基因组中大量不变基因稀释信号。

**技术细节**：encode 细胞 → 平均 CLS embedding → 用蛋白质 embedding decode 全部 18080 个基因 → 在 DE 基因子集上计算 Pearson r。（不是从 batch 直接取预测，因为每 batch 只随机采样 1536 个基因）

**输出**：
- `deg_pearson_<timestamp>.json`
- `deg_pearson_comparison.png`：各模型对比柱状图
- `deg_pearson_distribution.png`：每个扰动的 Pearson r 分布

### 4.4 可视化出图

**脚本**：`scripts/plot_results_summary.py`

**运行方式**（需先跑完 4.1 和 4.2）：
```bash
python ../scripts/plot_results_summary.py \
    --base_csv      .../version_11/metrics.csv \
    --mmd_csv       .../version_12/metrics.csv \
    --cosine_csv    .../version_14/metrics.csv \
    --base_align    ".../version_11/eval_alignment/domain_alignment*.json" \
    --mmd_align     ".../version_12/eval_alignment/domain_alignment*.json" \
    --cosine_align  ".../version_14/eval_alignment/domain_alignment*.json" \
    --base_pearson  ".../version_11/eval_pearson/pearson_results*.json" \
    --mmd_pearson   ".../version_12/eval_pearson/pearson_results*.json" \
    --cosine_pearson ".../version_14/eval_pearson/pearson_results*.json" \
    --base_umap     .../version_11/eval_alignment/umap_mmd_aligned.png \
    --mmd_umap      .../version_12/eval_alignment/umap_mmd_aligned.png \
    --cosine_umap   .../version_14/eval_alignment/umap_mmd_aligned.png \
    --output        /media/.../state/figures_three_way
```

**生成 6 张图**：

| 图 | 文件名 | 内容 |
|----|--------|------|
| fig1 | `fig1_pearson_bar.png` | 三模型 Pearson r 柱状图对比 |
| fig2 | `fig2_training_curves.png` | 训练/验证损失曲线三路对比 |
| fig3 | `fig3_alpha_schedule.png` | staircase vs cosine α 曲线形状对比（新） |
| fig4 | `fig4_alignment_metrics.png` | DomainClsAcc / Silhouette / MMD 三路对比 |
| fig5 | `fig5_umap_comparison.png` | UMAP 三图并排 |
| fig6 | `fig6_summary_panel.png` | 综合总结面板 |

---

## 五、当前结果

### 5.1 Pearson r（Mode A 细胞级，HepG2 零样本）

| 模型 | Mean ± Std | Median |
|------|-----------|--------|
| No-MMD Baseline | 0.700 ± 0.046 | 0.703 |
| STATE+MMD (staircase) | 0.699 ± 0.046 | 0.703 |
| STATE+MMD (cosine) | （已跑，见 version_14/eval_pearson） |

**结论**：三者基本持平，MMD 对齐分布但未提升 Pearson r。

### 5.2 域对齐指标

| 指标 | Baseline | MMD-staircase | MMD-cosine |
|------|----------|---------------|------------|
| DomainClsAcc | ~0.99 | ~0.99 | 待看 |
| Silhouette | ~0.40 | ~0.04 | 待看 |
| MMD mean | ~0.188 | ~0.001 | 待看 |

**结论**：
- MMD 成功对齐低阶统计量（Silhouette ↓，MMD ↓）
- DomainClsAcc 仍 99%（GRL 悖论）：encoder 只骗了训练时的判别器，新 MLP 仍能 99% 区分域
- 为什么 Pearson r 没提升：多域预训练本身已提供泛化，对齐分布不等于对齐高层语义

### 5.3 DEG Pearson r

尚未完成（脚本已修复 IndexError bug，需重跑）。

---

## 六、代码修改记录

| 文件 | 修改内容 |
|------|----------|
| `src/state/emb/train/trainer.py` | 1) CumulativeFLOPSCallback 改为条件触发（修复 OOM）2) train_domains 改为 config 可配置 |
| `src/state/emb/nn/model.py` | `_get_alignment_alpha()` 改为 cosine S 型曲线（原为 epoch 级阶梯） |
| `configs/no_mmd_baseline_config.yaml` | 新建：无 MMD 基线配置 |
| `configs/mmd_aae_cosine_config.yaml` | 新建：cosine α 版 MMD 配置 |
| `configs/loo_*.yaml` | 新建：3 个 LOO 配置（当前报错） |
| `scripts/eval_hepg2_pearson.py` | 现有脚本，Mode A/B Pearson 评估 |
| `scripts/eval_state_domain_alignment.py` | 现有脚本，域对齐 4 指标 + UMAP |
| `scripts/eval_deg_pearson.py` | 新建：谢老师 DEG 方法 Pearson r 评估 |
| `scripts/plot_results_summary.py` | 扩展为三路对比（原为两路），新增 fig3 α 曲线图 |

---

## 七、下一步（按优先级）

### 7.1 立即执行：DEG Pearson r 评估（谢老师最高要求）

```bash
cd /media/mldadmin/home/s125mdg34_03/state/src
git pull origin main   # 拉取修复后的脚本

python ../scripts/eval_deg_pearson.py \
    --checkpoint /media/mldadmin/home/s125mdg34_03/state/checkpoints/mmd_aae_cosine/last.ckpt \
    --baseline   /media/mldadmin/home/s125mdg34_03/state/checkpoints/no_mmd_baseline/last.ckpt \
    --config     ../configs/mmd_aae_cosine_config.yaml \
    --h5ad       /media/mldadmin/home/s125mdg34_03/state/competition_support_set/hepg2.h5 \
    --pert_col   gene \
    --ctrl_label non-targeting
```

### 7.2 重新出图（补全三路数据后）

所有评估跑完后用 `plot_results_summary.py` 生成完整 6 张图。

### 7.3 LOO 实验（次要，看错误决定）

把 LOO 错误日志贴出来，判断能否快速修复。

### 7.4 老师可能追问的问题与回答

| 问题 | 答案 |
|------|------|
| TabularLoss 是什么？ | 两个能量距离之和（gene_mmd + cell_mmd），优化预测分布 vs 真实分布的相似性，不是 MSE |
| 为什么 Pearson r 不变？ | 多域预训练本身已有泛化；MMD 对齐低阶统计量但高阶特征（DomainClsAcc=99%）未消除；pred_loss 与 Pearson r 不等价 |
| α 控制什么？ | α 是对齐损失（MMD+ADV）相对于预测损失的权重；α=0 时纯预测，α=1 时全力对齐；cosine 版比 staircase 更平滑 |
| adv_weight 0.1 是否太弱？ | 是潜在原因之一；ADV 损失收敛到 ln(3)（判别器被欺骗），但新 MLP 仍能分 99%，说明 encoder 没有真正消除域信息 |
| 为什么用 DEG Pearson 而不是全基因 Pearson？ | 全基因中大量基因在扰动后不变，稀释了真正的预测信号；只看 Wilcoxon 显著 DE 基因（adj p<0.05）才能评估模型是否正确预测了真实的扰动效应 |

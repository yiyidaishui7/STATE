# STATE + MMD-AAE 集成进展文档

## 最终目标

将 MMD-AAE 域对齐模块集成到 STATE 模型中，使 STATE 的 CLS token embedding 具备域不变性（跨 K562/RPE1/Jurkat），从而提升零样本泛化能力（在 HepG2 等未见细胞系上的表现）。

```
[K562 cells]  ─┐
[RPE1 cells]  ─┤→ STATE Transformer → CLS embedding ─→ Prediction loss
[Jurkat cells]─┘                              │
                                              ├→ MMD loss (域间距离最小化)
                                              └→ ADV loss (域判别器对抗)
```

---

## ✅ 已实现

### Phase 1: 独立 MMD-AAE 模块（已完成）
- 实现了独立的 MMD-AAE autoencoder（`src/state/emb/nn/mmd_aae.py`）
- 完成超参数调优实验（最优 `ratio=5`, `lambda_mmd≈2.12`）
- 验证了 MMD loss 可有效减小域间分布距离

### Phase 2: 集成到 STATE（代码已完成，正在调试启动）

| 文件 | 状态 | 内容 |
|------|------|------|
| `src/state/emb/nn/model.py` | ✅ 已修改 | GRL、域判别器、pairwise MMD、warmup schedule、`shared_step` 集成 |
| `src/state/emb/train/trainer.py` | ✅ 已修改 | `ParallelZipLoader` 合并三域 batch + 域标签 `batch[9]`、val dataset h5 直接读 |
| `src/state/emb/data/loader.py` | ✅ 已修改 | null `ds_emb_mapping` guard、KeyError fallback |
| `configs/mmd_aae_config.yaml` | ✅ 已修改 | 域对齐参数、mapping 路径、val_check_interval |
| `scripts/generate_ds_emb_mapping.py` | ✅ 已新建 | 生成基因→embedding 映射（已在服务器成功运行，98.3% 匹配） |

### 域对齐训练逻辑
- **Epoch 0-4**：只有 prediction loss（warmup）
- **Epoch 5-9**：alpha 从 0 线性增长到 1，逐渐加入 MMD + ADV loss
- **Epoch 10+**：全力对齐，`loss = pred + 2.0×mmd + 0.1×adv`

---

## 🔥 当前阻塞：OOM

训练进入 Transformer forward pass 时显存不足（24GB GPU 用尽）。

**直接原因**：合并 batch = 96（3×32），配合 `pad_length=2048`、8 层 Transformer。

**解决方案**（服务器上直接改 config）：
```bash
sed -i 's/batch_size: 96/batch_size: 48/' ~/state/configs/mmd_aae_config.yaml
sed -i 's/gradient_accumulation_steps: 8/gradient_accumulation_steps: 16/' ~/state/configs/mmd_aae_config.yaml
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -m state.emb.train --conf configs/mmd_aae_config.yaml
```

---

## ❌ 尚未完成

### 训练运行
- [x] 解决 OOM，让训练成功启动并稳定运行
- [x] 跑完 16 个 epoch，保存 checkpoint

### 评估与对比
- [x] 脚本已就绪：`scripts/eval_state_domain_alignment.py`（域对齐指标 + UMAP）
- [x] 脚本已就绪：`scripts/eval_hepg2_pearson.py`（HepG2 零样本 Pearson r）
- [x] 脚本已就绪：`scripts/plot_mmd_training.py`（MMD/ADV/alpha 曲线）
- [ ] **运行评估**：在服务器上执行上述脚本，获取实验数据

### 可选改进
- [ ] 使用预训练 STATE checkpoint 微调（而非从头训练）
- [ ] 调参：`mmd_weight`、`adv_weight` 在集成场景下的最优值
- [ ] 将 HepG2 也纳入三路训练（四域对齐）

---

## 关键文件索引

| 文件 | 用途 |
|------|------|
| `configs/mmd_aae_config.yaml` | 训练配置（batch size、对齐参数等） |
| `src/state/emb/nn/model.py` | STATE 模型（含域对齐逻辑） |
| `src/state/emb/train/trainer.py` | 训练入口（多域 DataLoader） |
| `src/state/emb/data/loader.py` | 数据加载与 collation |
| `scripts/generate_ds_emb_mapping.py` | 生成基因→embedding 映射（服务器运行一次） |
| `scripts/eval_state_domain_alignment.py` | 域对齐评估：DomainClsAcc/Silhouette/MMD/CORAL + UMAP |
| `scripts/eval_hepg2_pearson.py` | HepG2 零样本 Pearson r（细胞级 & 扰动级） |
| `scripts/plot_mmd_training.py` | Lightning CSV log → MMD/ADV/alpha 训练曲线 |

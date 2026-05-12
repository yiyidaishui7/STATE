#!/usr/bin/env python3
"""
eval_deg_pearson.py — DEG-based Pearson r evaluation (谢老师方法)

方法：
  对每个扰动条件（基因敲除），将扰动细胞 vs 对照细胞（non-targeting）
  做 Wilcoxon rank-sum test (BH校正)，筛选显著 DE 基因 (adj p < 0.05)。
  只在这些关键 DE 基因上计算 Pearson r（预测 vs 实际），
  避免全基因组中大量不变基因稀释信号。

使用方法（服务器 ~/state/src 目录）：

    python ../scripts/eval_deg_pearson.py \\
        --checkpoint /path/to/ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --h5ad /path/to/hepg2.h5 \\
        --pert_col gene \\
        --ctrl_label non-targeting

    # 对比两个 checkpoint
    python ../scripts/eval_deg_pearson.py \\
        --checkpoint /path/to/mmd_ckpt.ckpt \\
        --baseline /path/to/baseline_ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --h5ad /path/to/hepg2.h5

    # 对 LOO checkpoint 在留出域上评估
    python ../scripts/eval_deg_pearson.py \\
        --checkpoint /path/to/loo_ckpt.ckpt \\
        --config ../configs/loo_jurkat_config.yaml \\
        --h5ad /path/to/jurkat.h5
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
DEFAULT_H5AD = f"{BASE_DIR}/competition_support_set/hepg2.h5"
FIGURES_DIR = f"{BASE_DIR}/figures"


def parse_args():
    p = argparse.ArgumentParser(description="DEG-based Pearson r evaluation")
    p.add_argument("--checkpoint", required=True, help="主 checkpoint (.ckpt)")
    p.add_argument("--baseline", default=None, help="基线 checkpoint（对比用）")
    p.add_argument("--config", required=True, help="config yaml 路径")
    p.add_argument("--h5ad", default=DEFAULT_H5AD, help="h5ad 文件路径（带扰动标签）")
    p.add_argument("--pert_col", default="gene", help="obs 列名（扰动标签）")
    p.add_argument("--ctrl_label", default="non-targeting", help="对照组标签")
    p.add_argument("--pval_cutoff", type=float, default=0.05,
                   help="adj p-value 阈值（--top_k_max 未设时生效）")
    p.add_argument("--top_k_max", type=int, default=10,
                   help="按 |Wilcoxon scores| 依次取 top-1..top_k_max 基因各算一次 PCC（默认 10）")
    p.add_argument("--min_de_genes", type=int, default=5, help="最少 DE 基因数才计算 Pearson r")
    p.add_argument("--max_cells_per_group", type=int, default=200,
                   help="每组最多取多少细胞（加速）")
    p.add_argument("--batch_size", type=int, default=16, help="推理 batch size")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--output", default=None, help="输出目录")
    p.add_argument("--read_depth", type=float, default=4.0, help="RDA read depth")
    return p.parse_args()


# ============================================================================
# 数据加载
# ============================================================================

def load_adata(h5_path):
    """加载 h5ad 文件，返回 anndata.AnnData，X 为 log1p counts (dense)。"""
    import anndata as ad
    from scipy.sparse import issparse

    try:
        adata = ad.read_h5ad(h5_path)
    except Exception:
        # 尝试 h5 格式 fallback
        import h5py
        from scipy.sparse import csr_matrix
        with h5py.File(h5_path, "r") as f:
            if "indices" in f["X"]:
                data = f["X"]["data"][:]
                indices = f["X"]["indices"][:]
                indptr = f["X"]["indptr"][:]
                attrs = dict(f["X"].attrs)
                n_cells = indptr.shape[0] - 1
                n_genes = int(attrs["shape"][1])
                X = csr_matrix((data, indices, indptr), shape=(n_cells, n_genes)).toarray()
            else:
                X = f["X"][:]
        adata = ad.AnnData(X=X)
        print("  警告：以 fallback h5 模式读取，无 obs 元数据")

    if issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = adata.X.astype(np.float32)

    # 若是 raw counts（最大值 > 100），做 log1p 归一化
    if adata.X.max() > 100:
        print("  检测到 raw counts，执行 log1p 归一化")
        adata.X = np.log1p(adata.X)

    return adata


# ============================================================================
# 模型加载
# ============================================================================

def load_state_model(checkpoint_path, cfg):
    from state.emb.nn.model import StateEmbeddingModel
    from state.emb.train.trainer import get_embeddings
    from state.emb.utils import get_embedding_cfg

    emb_cfg = get_embedding_cfg(cfg)

    print(f"  加载 checkpoint: {checkpoint_path}")
    model = StateEmbeddingModel.load_from_checkpoint(
        checkpoint_path,
        token_dim=emb_cfg.size,
        d_model=cfg.model.emsize,
        nhead=cfg.model.nhead,
        d_hid=cfg.model.d_hid,
        nlayers=cfg.model.nlayers,
        output_dim=cfg.model.output_dim,
        dropout=0.0,
        cfg=cfg,
        strict=False,
        map_location="cpu",
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "protein_embeds_dict" in ckpt:
        pe_dict = ckpt["protein_embeds_dict"]
        all_pe = torch.vstack(list(pe_dict.values())).to(device)
    else:
        all_pe = get_embeddings(cfg).to(device)

    all_pe.requires_grad_(False)
    model.pe_embedding = nn.Embedding.from_pretrained(all_pe)
    model.pe_embedding = model.pe_embedding.to(device)

    # 将 protein_embeds_dict 挂到 model.protein_embeds，使 get_gene_embedding() 可按基因名查询
    if "protein_embeds_dict" in ckpt:
        model.protein_embeds = ckpt["protein_embeds_dict"]

    model.eval()
    return model, device


# ============================================================================
# 模型推理：encode 细胞 → CLS embedding，再 decode 全部基因
# ============================================================================

def encode_cells_to_cls(model, cfg, h5_path, cell_indices, args, device):
    """
    对指定细胞编码，返回 CLS embedding np.ndarray (n_cells, 512)。
    不受每 batch 采样基因数限制。
    """
    from state.emb.data import H5adSentenceDataset, VCIDatasetSentenceCollator
    from torch.utils.data import Subset
    import h5py

    domain_name = Path(h5_path).stem
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["X"].attrs)
        if "shape" in attrs:
            n_cells_total, n_genes = int(attrs["shape"][0]), int(attrs["shape"][1])
        elif hasattr(f["X"], "shape") and len(f["X"].shape) == 2:
            n_cells_total, n_genes = f["X"].shape[0], f["X"].shape[1]
        else:
            indptr = f["X"]["indptr"]
            n_cells_total = indptr.shape[0] - 1
            n_genes = int(attrs.get("shape", [0, 18080])[1])

    dataset = H5adSentenceDataset(
        cfg, test=True,
        datasets=[domain_name],
        shape_dict={domain_name: (n_cells_total, n_genes)},
    )
    dataset.dataset_path_map = {domain_name: h5_path}

    subset = Subset(dataset, cell_indices)
    collator = VCIDatasetSentenceCollator(cfg, is_train=False)
    collator.cfg = cfg

    # num_workers=0 避免在循环中多次创建 worker 进程导致文件描述符泄漏 (Errno 9)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
    )

    all_cls = []
    with torch.no_grad():
        for batch in loader:
            _, _, _, embs, _ = model._compute_embedding_for_batch(batch)
            all_cls.append(embs.detach().cpu().float().numpy())

    return np.concatenate(all_cls, axis=0)  # (n_cells, 512)


def predict_from_emb(model, cell_emb, gene_embs, read_depth, device):
    """
    给定单个 CLS embedding (output_dim,) 和所有基因 embedding (n_genes, d_model)，
    返回全基因预测分数 np.ndarray (n_genes,)。
    分批计算避免 OOM。
    """
    batch_size = 512
    n_genes = gene_embs.size(0)
    all_scores = []

    # 兼容 (output_dim,) 和 (1, output_dim) 两种输入形状
    cell_emb = cell_emb.reshape(-1)              # 统一为 (output_dim,)
    z_base = cell_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, output_dim)

    for start in range(0, n_genes, batch_size):
        end = min(start + batch_size, n_genes)
        gene_batch = gene_embs[start:end].to(device)
        n_batch = gene_batch.size(0)

        z = z_base.expand(1, n_batch, -1)   # (1, n_batch, output_dim)
        g = gene_batch.unsqueeze(0)          # (1, n_batch, d_model)

        if model.z_dim_rd == 1:
            rd = torch.full((1, n_batch, 1), read_depth, device=device)
            combine = torch.cat([g, z, rd], dim=2)
        else:
            combine = torch.cat([g, z], dim=2)

        with torch.no_grad():
            # squeeze(-1) 避免 n_batch==1 时把整个 tensor 压成标量
            scores = model.binary_decoder(combine).squeeze(-1).squeeze(0)
            all_scores.append(scores.cpu().float().numpy())

    return np.concatenate(all_scores)  # (n_genes,)


def _build_gene_embs(model, h5_path, device):
    """
    构建全基因的 d_model 维 embedding。
    假设 h5 文件中第 i 列基因对应 pe_embedding 第 i 行
    （与训练时 H5adSentenceDataset 的列顺序一致）。
    """
    import h5py
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["X"].attrs)
        if "shape" in attrs:
            n_genes = int(attrs["shape"][1])
        elif hasattr(f["X"], "shape") and len(f["X"].shape) == 2:
            n_genes = f["X"].shape[1]
        else:
            n_genes = int(attrs.get("shape", [0, 18080])[1])

    with torch.no_grad():
        gene_indices = torch.arange(n_genes, device=device)
        raw_embs = model.pe_embedding(gene_indices)       # (n_genes, token_dim)
        gene_embs = model.gene_embedding_layer(raw_embs)  # (n_genes, d_model)
    return gene_embs


def get_predictions_for_cells(model, cfg, h5_path, cell_indices, args, device,
                               gene_embs=None):
    """
    对指定细胞编码并解码到全基因预测空间。
    返回 np.ndarray (n_cells, n_genes)。

    gene_embs: 可选，预先计算好的基因 embedding (n_genes, d_model)。
               跨扰动复用时传入以避免重复计算。
    """
    # Step 1: CLS embeddings (n_cells, output_dim)
    cls_embs = encode_cells_to_cls(model, cfg, h5_path, cell_indices, args, device)

    # Step 2: 获取基因 embedding（若未传入则现算）
    if gene_embs is None:
        gene_embs = _build_gene_embs(model, h5_path, device)

    # Step 3: 对每个细胞解码全基因
    all_preds = []
    for i in range(len(cls_embs)):
        cell_tensor = torch.tensor(cls_embs[i], dtype=torch.float32, device=device)
        pred = predict_from_emb(model, cell_tensor, gene_embs, args.read_depth, device)
        all_preds.append(pred)

    return np.stack(all_preds, axis=0)  # (n_cells, n_genes)


# ============================================================================
# 核心评估：DEG-based Pearson r
# ============================================================================

def eval_deg_pearson(model, cfg, h5_path, args, device, label="model"):
    """
    对每个扰动，按 |Wilcoxon scores| 降序排列全部基因，
    依次取 top-1, top-2, ..., top_k_max 个基因各算一次 Pearson r，
    输出 PCC 随基因数 k 变化的曲线。

    推理：encode 细胞 → 平均 CLS embedding → decode 全部基因（循环外只算一次）
    """
    import scanpy as sc
    import anndata as ad

    adata = load_adata(h5_path)

    if args.pert_col not in adata.obs.columns:
        print(f"  列 '{args.pert_col}' 不存在。可用列: {list(adata.obs.columns)}")
        return None

    pert_labels = adata.obs[args.pert_col].astype(str).values
    ctrl_mask = pert_labels == args.ctrl_label
    n_ctrl = ctrl_mask.sum()

    if n_ctrl == 0:
        print(f"  未找到对照组（'{args.ctrl_label}'）。可用标签: {np.unique(pert_labels)[:10]}")
        return None

    print(f"\n  对照细胞: {n_ctrl}")
    pert_unique = sorted([p for p in np.unique(pert_labels) if p != args.ctrl_label])
    print(f"  扰动类型数: {len(pert_unique)}")

    # 对照组细胞索引（取前 max_cells_per_group 个）
    ctrl_indices = np.where(ctrl_mask)[0]
    if len(ctrl_indices) > args.max_cells_per_group:
        ctrl_indices = ctrl_indices[:args.max_cells_per_group]

    # ---- 预计算全基因 embedding（循环外，只算一次）----
    gene_names = list(adata.var_names)
    n_total_genes = len(gene_names)
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    print(f"  获取全基因 embedding ({n_total_genes} genes)...")
    with torch.no_grad():
        gene_embs = model.get_gene_embedding(gene_names).detach().cpu()  # (n_genes, d_model)

    # ---- 预计算 ctrl CLS embedding 均值（循环外，只算一次）----
    print(f"  编码对照组细胞 ({len(ctrl_indices)} cells)...")
    ctrl_cls = encode_cells_to_cls(model, cfg, h5_path, list(ctrl_indices), args, device)
    ctrl_cls_mean = torch.tensor(ctrl_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)

    # ctrl 全基因预测（循环外）
    pred_ctrl_all = predict_from_emb(model, ctrl_cls_mean, gene_embs, args.read_depth, device)
    # pred_ctrl_all: (n_total_genes,)

    actual_ctrl_mean_all = np.asarray(adata.X[ctrl_indices].mean(axis=0)).ravel()  # (n_total_genes,)

    k_values = list(range(1, args.top_k_max + 1))
    pearson_at_k = {k: [] for k in k_values}   # PCC 列表，按 k 分组
    results_per_pert = []
    n_skipped = 0

    for pert in tqdm(pert_unique, desc=f"  [{label}] DEG Pearson curve"):
        pert_mask = pert_labels == pert
        pert_indices = np.where(pert_mask)[0]

        if len(pert_indices) < 5:
            n_skipped += 1
            continue

        if len(pert_indices) > args.max_cells_per_group:
            pert_indices = pert_indices[:args.max_cells_per_group]

        # ---- Step 1: Wilcoxon，取全部基因按 |scores| 排序 ----
        adata_ctrl = adata[ctrl_indices].copy()
        adata_pert = adata[pert_indices].copy()
        adata_ctrl.obs["_group"] = "control"
        adata_pert.obs["_group"] = "perturbed"
        adata_combined = ad.concat([adata_ctrl, adata_pert])

        try:
            sc.tl.rank_genes_groups(
                adata_combined,
                groupby="_group",
                groups=["perturbed"],
                reference="control",
                method="wilcoxon",
                corr_method="benjamini-hochberg",
                tie_correct=True,
                use_raw=False,
            )
            de_df = sc.get.rank_genes_groups_df(adata_combined, group="perturbed")
            de_df = de_df.copy()
            de_df["abs_scores"] = de_df["scores"].abs()
            de_df = de_df.sort_values("abs_scores", ascending=False).reset_index(drop=True)
        except Exception:
            n_skipped += 1
            continue

        # 取前 top_k_max 个基因名（后续按 k 截取）
        ranked_genes = de_df["names"].iloc[: args.top_k_max].tolist()

        # ---- Step 2: 编码扰动组 → 预测全基因 ----
        try:
            pert_cls = encode_cells_to_cls(model, cfg, h5_path, list(pert_indices), args, device)
        except Exception as e:
            print(f"  编码失败 ({pert}): {e}")
            n_skipped += 1
            continue

        pert_cls_mean = torch.tensor(pert_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
        pred_pert_all = predict_from_emb(model, pert_cls_mean, gene_embs, args.read_depth, device)

        # 全基因 logFC（真实 & 预测）
        actual_pert_mean_all = np.asarray(adata.X[pert_indices].mean(axis=0)).ravel()
        actual_logfc_all = actual_pert_mean_all - actual_ctrl_mean_all   # (n_total_genes,)
        pred_logfc_all   = pred_pert_all - pred_ctrl_all                  # (n_total_genes,)

        # ---- Step 3: 对 k=1..top_k_max 各算一次 Pearson r ----
        pert_row = {"perturbation": pert, "n_pert_cells": len(pert_indices), "pearson_at_k": {}}
        for k in k_values:
            names_k = ranked_genes[:k]
            idx_k = np.array([gene_name_to_idx[g] for g in names_k if g in gene_name_to_idx])

            if len(idx_k) < 2:
                # k=1 时只有 1 个点，Pearson r 无意义
                val = float("nan")
            else:
                a = actual_logfc_all[idx_k]
                p = pred_logfc_all[idx_k]
                valid = ~(np.isnan(a) | np.isnan(p))
                if valid.sum() < 2:
                    val = float("nan")
                else:
                    try:
                        r, _ = pearsonr(p[valid], a[valid])
                        val = float(r) if not np.isnan(r) else float("nan")
                    except Exception:
                        val = float("nan")

            pearson_at_k[k].append(val)
            pert_row["pearson_at_k"][k] = val

        results_per_pert.append(pert_row)

    n_valid = len(results_per_pert) - n_skipped
    print(f"\n  完成 {len(results_per_pert)} 个扰动（跳过 {n_skipped} 个）")

    if not results_per_pert:
        return {"label": label, "error": "no valid perturbations"}

    # ---- 汇总：每个 k 的 mean/median/std（忽略 NaN）----
    pearson_curve = {}
    print(f"\n  {'k':>4}  {'mean PCC':>10}  {'median PCC':>12}  {'std':>8}  {'N valid':>8}")
    print(f"  {'-'*50}")
    for k in k_values:
        vals = [v for v in pearson_at_k[k] if not np.isnan(v)]
        if vals:
            m, med, s = float(np.mean(vals)), float(np.median(vals)), float(np.std(vals))
        else:
            m = med = s = float("nan")
        pearson_curve[k] = {"mean": m, "median": med, "std": s, "n_valid": len(vals)}
        print(f"  {k:>4}  {m:>10.4f}  {med:>12.4f}  {s:>8.4f}  {len(vals):>8}")

    return {
        "label": label,
        "n_perturbations": len(results_per_pert),
        "pearson_curve": pearson_curve,
        "per_perturbation": results_per_pert,
    }


def eval_deg_pearson_pval(model, cfg, h5_path, args, device, label="model",
                          pval_cutoff=0.05):
    """
    传统 p-value 阈值方法：筛 adj p < pval_cutoff 的 DEG，在这些基因上算单个 PCC。
    与 eval_deg_pearson（top-k）输出可直接对比。
    """
    import scanpy as sc
    import anndata as ad

    adata = load_adata(h5_path)
    pert_labels = adata.obs[args.pert_col].astype(str).values
    ctrl_mask = pert_labels == args.ctrl_label
    ctrl_indices = np.where(ctrl_mask)[0]
    if len(ctrl_indices) > args.max_cells_per_group:
        ctrl_indices = ctrl_indices[:args.max_cells_per_group]

    pert_unique = sorted([p for p in np.unique(pert_labels) if p != args.ctrl_label])

    # 预计算（循环外只算一次）
    gene_names = list(adata.var_names)
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    with torch.no_grad():
        gene_embs = model.get_gene_embedding(gene_names).detach().cpu()
    ctrl_cls = encode_cells_to_cls(model, cfg, h5_path, list(ctrl_indices), args, device)
    ctrl_cls_mean = torch.tensor(ctrl_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    pred_ctrl_all = predict_from_emb(model, ctrl_cls_mean, gene_embs, args.read_depth, device)
    actual_ctrl_mean_all = np.asarray(adata.X[ctrl_indices].mean(axis=0)).ravel()

    pearson_list = []
    results_per_pert = []
    n_skipped = 0

    for pert in tqdm(pert_unique, desc=f"  [{label}] pval<{pval_cutoff}"):
        pert_mask = pert_labels == pert
        pert_indices = np.where(pert_mask)[0]
        if len(pert_indices) < 5:
            n_skipped += 1
            continue
        if len(pert_indices) > args.max_cells_per_group:
            pert_indices = pert_indices[:args.max_cells_per_group]

        # Wilcoxon + BH 校正，筛 adj p < pval_cutoff
        adata_ctrl = adata[ctrl_indices].copy()
        adata_pert = adata[pert_indices].copy()
        adata_ctrl.obs["_group"] = "control"
        adata_pert.obs["_group"] = "perturbed"
        adata_combined = ad.concat([adata_ctrl, adata_pert])
        try:
            sc.tl.rank_genes_groups(
                adata_combined, groupby="_group", groups=["perturbed"],
                reference="control", method="wilcoxon",
                corr_method="benjamini-hochberg", tie_correct=True, use_raw=False,
            )
            de_df = sc.get.rank_genes_groups_df(
                adata_combined, group="perturbed", pval_cutoff=pval_cutoff
            )
        except Exception:
            n_skipped += 1
            continue

        if de_df is None or len(de_df) < args.min_de_genes:
            n_skipped += 1
            continue

        de_genes = de_df["names"].tolist()
        idx_de = np.array([gene_name_to_idx[g] for g in de_genes if g in gene_name_to_idx])
        if len(idx_de) < 2:
            n_skipped += 1
            continue

        # 模型预测
        try:
            pert_cls = encode_cells_to_cls(model, cfg, h5_path, list(pert_indices), args, device)
        except Exception:
            n_skipped += 1
            continue
        pert_cls_mean = torch.tensor(pert_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
        pred_pert_all = predict_from_emb(model, pert_cls_mean, gene_embs, args.read_depth, device)

        actual_pert_mean_all = np.asarray(adata.X[pert_indices].mean(axis=0)).ravel()
        actual_logfc = actual_pert_mean_all[idx_de] - actual_ctrl_mean_all[idx_de]
        pred_logfc   = pred_pert_all[idx_de]        - pred_ctrl_all[idx_de]

        valid = ~(np.isnan(actual_logfc) | np.isnan(pred_logfc))
        if valid.sum() < 2:
            n_skipped += 1
            continue
        try:
            r, _ = pearsonr(pred_logfc[valid], actual_logfc[valid])
            r = float(r) if not np.isnan(r) else float("nan")
        except Exception:
            r = float("nan")

        pearson_list.append(r)
        results_per_pert.append({
            "perturbation": pert,
            "n_de_genes": len(idx_de),
            "pearson_r": r,
        })

    print(f"\n  完成 {len(results_per_pert)} 个扰动（跳过 {n_skipped} 个）")

    vals = [v for v in pearson_list if not np.isnan(v)]
    if not vals:
        return {"label": label, "method": f"pval<{pval_cutoff}", "error": "no valid perturbations"}

    m, med, s = float(np.mean(vals)), float(np.median(vals)), float(np.std(vals))
    frac_pos = float(np.mean(np.array(vals) > 0))
    print(f"  Mean={m:.4f}  Median={med:.4f}  Std={s:.4f}  "
          f"N={len(vals)}  Positive={frac_pos:.1%}")

    return {
        "label": label,
        "method": f"pval<{pval_cutoff}",
        "pearson_mean": m,
        "pearson_median": med,
        "pearson_std": s,
        "frac_positive": frac_pos,
        "n_perturbations": len(vals),
        "per_perturbation": results_per_pert,
    }


# ============================================================================
# 绘图
# ============================================================================

def plot_pcc_curve(results_list, output_dir):
    """双子图：上=mean PCC 折线 + std 阴影，下=正相关比例折线。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results_list if "pearson_curve" in r]
    if not valid:
        return

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)

    for r, color in zip(valid, colors):
        curve = r["pearson_curve"]
        ks = sorted(curve.keys())
        means  = np.array([curve[k]["mean"] for k in ks], dtype=float)
        stds   = np.array([curve[k]["std"]  for k in ks], dtype=float)

        # 上图：mean PCC + std 阴影
        ax1.plot(ks, means, marker="o", color=color, label=r["label"], linewidth=2, markersize=5)
        ax1.fill_between(ks, means - stds, means + stds, color=color, alpha=0.15)

        # 下图：每个 k 下 PCC > 0 的扰动比例
        frac_pos = []
        for k in ks:
            vals = [x["pearson_at_k"].get(k, float("nan"))
                    for x in r.get("per_perturbation", [])
                    if not np.isnan(x["pearson_at_k"].get(k, float("nan")))]
            frac_pos.append(np.mean(np.array(vals) > 0) if vals else float("nan"))
        ax2.plot(ks, frac_pos, marker="s", color=color, label=r["label"], linewidth=2, markersize=5)

    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax1.set_ylabel("Mean Pearson r", fontsize=12)
    ax1.set_title("DEG Pearson r vs. Number of Top Genes (ranked by |Wilcoxon scores|)", fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="random (50%)")
    ax2.set_xlabel("Top-k genes", fontsize=12)
    ax2.set_ylabel("Fraction PCC > 0", fontsize=12)
    ax2.set_title("Fraction of Perturbations with PCC > 0", fontsize=12)
    ax2.set_xticks(ks)
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "deg_pearson_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {path}")


def plot_per_pert_distribution(results_list, output_dir, k):
    """在 top-k 固定时，各扰动 PCC 的直方图分布。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]
    fig, ax = plt.subplots(figsize=(7, 5))

    for res, color in zip(results_list, colors):
        if "per_perturbation" not in res:
            continue
        rs = [x["pearson_at_k"].get(k, float("nan")) for x in res["per_perturbation"]]
        rs = [v for v in rs if not np.isnan(v)]
        if not rs:
            continue
        ax.hist(rs, bins=20, alpha=0.6, color=color,
                label=f"{res['label']} (mean={np.mean(rs):.4f})",
                edgecolor="white", linewidth=0.5)

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel(f"Pearson r at top-{k} genes (per perturbation)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Per-Perturbation PCC Distribution (top-{k} by |scores|)", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, f"deg_pearson_dist_top{k}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {path}")


def plot_pcc_heatmap(results_list, output_dir):
    """热图：行=扰动，列=k，颜色=PCC；每个模型一张图。行按 mean PCC 排序。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    for res in results_list:
        if "per_perturbation" not in res or "pearson_curve" not in res:
            continue

        ks = sorted(res["pearson_curve"].keys())
        rows = res["per_perturbation"]

        # 构建矩阵 (n_perts × n_k)
        pert_names = [r["perturbation"] for r in rows]
        matrix = np.array([
            [r["pearson_at_k"].get(k, float("nan")) for k in ks]
            for r in rows
        ], dtype=float)

        # 按每行均值（忽略 NaN）降序排列
        row_means = np.nanmean(matrix, axis=1)
        order = np.argsort(row_means)[::-1]
        matrix = matrix[order]
        pert_names = [pert_names[i] for i in order]

        vmax = max(abs(np.nanmax(matrix)), abs(np.nanmin(matrix)), 0.01)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        fig_h = max(6, len(pert_names) * 0.25)
        fig, ax = plt.subplots(figsize=(max(6, len(ks) * 0.7), fig_h))
        im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", norm=norm)

        ax.set_xticks(range(len(ks)))
        ax.set_xticklabels([f"top-{k}" for k in ks], fontsize=9)
        ax.set_yticks(range(len(pert_names)))
        ax.set_yticklabels(pert_names, fontsize=7)
        ax.set_xlabel("Number of top genes (by |Wilcoxon scores|)", fontsize=11)
        ax.set_title(f"Per-Perturbation PCC Heatmap — {res['label']}\n(sorted by mean PCC)", fontsize=11)

        plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.6)
        plt.tight_layout()
        label_safe = res["label"].replace(" ", "_").replace("+", "plus")
        path = os.path.join(output_dir, f"deg_pearson_heatmap_{label_safe}.png")
        plt.savefig(path, dpi=200, bbox_inches="tight")
        plt.close()
        print(f"  图已保存: {path}")


def plot_method_comparison(topk_results, pval_results, output_dir, pval_cutoff=0.05):
    """
    对比图：左=top-k PCC 曲线，右=p-value 阈值法各模型 mean PCC 柱状图。
    直观展示两种评估方法的差异。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ---- 左图：top-k PCC 曲线 ----
    valid_topk = [r for r in topk_results if "pearson_curve" in r]
    for r, color in zip(valid_topk, colors):
        curve = r["pearson_curve"]
        ks = sorted(curve.keys())
        means = np.array([curve[k]["mean"] for k in ks], dtype=float)
        stds  = np.array([curve[k]["std"]  for k in ks], dtype=float)
        ax1.plot(ks, means, marker="o", color=color, label=r["label"], linewidth=2, markersize=5)
        ax1.fill_between(ks, means - stds, means + stds, color=color, alpha=0.12)

    ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    ax1.set_xlabel("Top-k genes (by |Wilcoxon scores|)", fontsize=11)
    ax1.set_ylabel("Mean Pearson r", fontsize=11)
    ax1.set_title("Method A: Top-k PCC Curve\n(genes ranked by |Wilcoxon scores|)", fontsize=11)
    if valid_topk:
        ax1.set_xticks(sorted(valid_topk[0]["pearson_curve"].keys()))
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # ---- 右图：p-value 阈值法柱状图 ----
    valid_pval = [r for r in pval_results if "pearson_mean" in r]
    labels  = [r["label"] for r in valid_pval]
    means   = [r["pearson_mean"] for r in valid_pval]
    stds    = [r["pearson_std"]  for r in valid_pval]
    bar_colors = colors[:len(labels)]

    bars = ax2.bar(labels, means, color=bar_colors, alpha=0.85,
                   yerr=stds, capsize=5, error_kw={"linewidth": 1.2})
    ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 m + (0.01 if m >= 0 else -0.03),
                 f"{m:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Pearson r", fontsize=11)
    ax2.set_title(f"Method B: p-value Threshold\n(adj p < {pval_cutoff}, all DEGs)", fontsize=11)
    ax2.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Two Evaluation Methods: Top-k PCC vs. p-value Threshold",
                 fontsize=12, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, f"method_comparison_pval{pval_cutoff}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {path}")


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    output_dir = args.output
    if output_dir is None:
        # 默认保存到 state/figures/deg_pearson/<数据集名>/
        # 不同数据集（hepg2/jurkat/k562/rpe1）自动分文件夹
        dataset_name = Path(args.h5ad).stem  # e.g. "hepg2"
        output_dir = str(Path(FIGURES_DIR) / "deg_pearson" / dataset_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    print(f"数据文件: {args.h5ad}")
    print(f"扰动列: {args.pert_col}  对照标签: {args.ctrl_label}")
    print(f"DEG 选择: top-1..{args.top_k_max} by |Wilcoxon scores|  最少 DE 基因: {args.min_de_genes}")

    topk_results = []   # top-k PCC 曲线
    pval_results  = []  # p-value 阈值方法

    # ---- 主 checkpoint：两种方法都跑 ----
    print(f"\n{'='*60}\n[1] STATE+MMD — top-k PCC\n{'='*60}")
    model, device = load_state_model(args.checkpoint, cfg)
    res_main = eval_deg_pearson(model, cfg, args.h5ad, args, device, label="STATE+MMD")
    if res_main:
        topk_results.append(res_main)

    print(f"\n{'='*60}\n[1] STATE+MMD — pval<{args.pval_cutoff}\n{'='*60}")
    res_main_pval = eval_deg_pearson_pval(
        model, cfg, args.h5ad, args, device,
        label="STATE+MMD", pval_cutoff=args.pval_cutoff,
    )
    if res_main_pval:
        pval_results.append(res_main_pval)
    del model
    torch.cuda.empty_cache()

    # ---- 基线 checkpoint：两种方法都跑 ----
    if args.baseline:
        print(f"\n{'='*60}\n[2] Baseline — top-k PCC\n{'='*60}")
        model_base, device = load_state_model(args.baseline, cfg)
        res_base = eval_deg_pearson(model_base, cfg, args.h5ad, args, device, label="Baseline")
        if res_base:
            topk_results.append(res_base)

        print(f"\n{'='*60}\n[2] Baseline — pval<{args.pval_cutoff}\n{'='*60}")
        res_base_pval = eval_deg_pearson_pval(
            model_base, cfg, args.h5ad, args, device,
            label="Baseline", pval_cutoff=args.pval_cutoff,
        )
        if res_base_pval:
            pval_results.append(res_base_pval)
        del model_base
        torch.cuda.empty_cache()

    # ---- 汇总打印 ----
    print("\n" + "=" * 70)
    print("方法 A：Top-k PCC 曲线（|Wilcoxon scores| 排序）")
    print("=" * 70)
    for res in topk_results:
        if "pearson_curve" not in res:
            continue
        print(f"\n  [{res['label']}]  N={res['n_perturbations']} 扰动")
        print(f"  {'k':>4}  {'mean PCC':>10}  {'median':>10}  {'std':>8}")
        for k, stat in sorted(res["pearson_curve"].items()):
            print(f"  {k:>4}  {stat['mean']:>10.4f}  {stat['median']:>10.4f}  {stat['std']:>8.4f}")

    print("\n" + "=" * 70)
    print(f"方法 B：p-value 阈值法（adj p < {args.pval_cutoff}，全部 DEG）")
    print("=" * 70)
    for res in pval_results:
        if "pearson_mean" not in res:
            continue
        print(f"  [{res['label']}]  Mean={res['pearson_mean']:.4f}  "
              f"Std={res['pearson_std']:.4f}  N={res['n_perturbations']}  "
              f"Positive={res['frac_positive']:.1%}")
    print("=" * 70)

    # ---- 绘图 ----
    all_topk = topk_results
    if all_topk:
        plot_pcc_curve(all_topk, output_dir)
        plot_per_pert_distribution(all_topk, output_dir, k=args.top_k_max)
        plot_pcc_heatmap(all_topk, output_dir)
    if all_topk and pval_results:
        plot_method_comparison(all_topk, pval_results, output_dir, args.pval_cutoff)

    # ---- 保存 JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_combined = {"topk": topk_results, "pval": pval_results}
    json_path = os.path.join(output_dir, f"deg_pearson_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(all_combined, f, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")
    print("评估完成")


if __name__ == "__main__":
    main()

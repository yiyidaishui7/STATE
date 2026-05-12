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
    """双子图：上=mean PCC（SEM阴影+三阶段标注），下=正相关比例。从 k=3 开始绘制。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    valid = [r for r in results_list if "pearson_curve" in r]
    if not valid:
        return

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    markers = ["o", "s", "^", "D"]

    # 确定 x 范围（从 k=3 开始，跳过 k=1 NaN 和 k=2 artifact）
    all_ks_raw = sorted(valid[0]["pearson_curve"].keys())
    ks = [k for k in all_ks_raw if k >= 3]
    max_k = max(ks)

    fig_w = max(10, min(max_k * 0.35, 22))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_w, 9), sharex=True)

    # ---- 三阶段背景色带 ----
    # 信号峰 k=3~5：浅绿；衰减区 k=6~15：浅黄；平台区 k=16+：浅蓝
    stage_defs = []
    if max_k >= 3:
        stage_defs.append((3, min(5, max_k),   "#d5f5e3", "Signal\nPeak"))
    if max_k >= 6:
        stage_defs.append((6, min(15, max_k),  "#fef9e7", "Decay"))
    if max_k >= 16:
        stage_defs.append((16, max_k,           "#eaf4fb", "Plateau"))

    for (x0, x1, fc, label) in stage_defs:
        for ax in (ax1, ax2):
            ax.axvspan(x0 - 0.5, x1 + 0.5, color=fc, alpha=0.6, zorder=0)
        # 在上图顶部标注阶段名
        ax1.text((x0 + x1) / 2, 1.01, label,
                 ha="center", va="bottom", fontsize=8, color="#555555",
                 transform=ax1.get_xaxis_transform())

    for r, color, mkr in zip(valid, colors, markers):
        curve = r["pearson_curve"]
        means = np.array([curve[k]["mean"] for k in ks], dtype=float)
        stds  = np.array([curve[k]["std"]  for k in ks], dtype=float)
        n_valid = np.array([curve[k].get("n_valid", 67) for k in ks], dtype=float)
        sem   = stds / np.sqrt(np.maximum(n_valid, 1))

        ax1.plot(ks, means, marker=mkr, color=color, label=r["label"],
                 linewidth=2.5, markersize=6, zorder=3)
        ax1.fill_between(ks, means - sem, means + sem, color=color, alpha=0.25, zorder=2)

        # 标注 k=3 和平台区均值
        if len(means) > 0 and not np.isnan(means[0]):
            ax1.annotate(f"{means[0]:+.3f}",
                         xy=(ks[0], means[0]),
                         xytext=(ks[0] + 0.3, means[0] + 0.01),
                         fontsize=8, color=color, fontweight="bold")
        if max_k >= 16:
            plateau_idx = [i for i, k in enumerate(ks) if k >= 16]
            if plateau_idx:
                plateau_mean = float(np.nanmean(means[plateau_idx]))
                ax1.axhline(plateau_mean, color=color, linestyle=":",
                            linewidth=1.2, alpha=0.7)
                ax1.text(max_k + 0.3, plateau_mean, f"{plateau_mean:+.3f}",
                         color=color, fontsize=8, va="center")

        # 下图：正相关比例
        frac_pos = []
        for k in ks:
            vals = [x["pearson_at_k"].get(k, float("nan"))
                    for x in r.get("per_perturbation", [])
                    if not np.isnan(x["pearson_at_k"].get(k, float("nan")))]
            frac_pos.append(np.mean(np.array(vals) > 0) if vals else float("nan"))
        ax2.plot(ks, frac_pos, marker=mkr, color=color, label=r["label"],
                 linewidth=2.5, markersize=6, zorder=3)

    ax1.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
    ax1.set_ylabel("Mean Pearson r", fontsize=12)
    ax1.set_title("Top-k PCC Curve  (genes ranked by |Wilcoxon scores|, SEM shading)",
                  fontsize=12, fontweight="bold")
    ax1.legend(fontsize=10, loc="upper right")
    ax1.grid(True, alpha=0.25)

    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=1.0,
                alpha=0.7, label="random (50%)", zorder=1)
    ax2.set_xlabel("Top-k genes", fontsize=12)
    ax2.set_ylabel("Fraction PCC > 0", fontsize=12)
    ax2.set_title("Fraction of Perturbations with Positive PCC", fontsize=12)
    stride = 1 if max_k <= 20 else (5 if max_k <= 50 else 10)
    ax2.set_xticks([k for k in ks if k % stride == 0 or k == min(ks)])
    ax2.set_ylim(0.2, 0.85)
    ax2.legend(fontsize=10, loc="upper right")
    ax2.grid(True, alpha=0.25)

    # 添加 k=2 artifact 说明
    fig.text(0.01, 0.01,
             "Note: k=1 (undefined) and k=2 (mathematical artifact, pearsonr≡±1) omitted.",
             fontsize=8, color="#888888")

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    path = os.path.join(output_dir, "deg_pearson_curve.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {path}")


def plot_per_pert_distribution(results_list, output_dir, k):
    """多 k violin 图：在信号峰/衰减/平台各取代表性 k，展示跨扰动 PCC 分布演变。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results_list if "per_perturbation" in r and "pearson_curve" in r]
    if not valid:
        return

    # 根据 k_max 选代表性 k 值
    all_ks = sorted(valid[0]["pearson_curve"].keys())
    max_k = max(all_ks)
    candidates = [3, 5, 10, 15, 20, 25, 30, 40, 50]
    show_ks = [kk for kk in candidates if kk <= max_k and kk >= 3]
    if not show_ks:
        show_ks = [kk for kk in all_ks if kk >= 3][:6]

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"]
    n_models = len(valid)
    n_ks = len(show_ks)

    fig, axes = plt.subplots(1, n_ks, figsize=(max(10, n_ks * 2.2), 5.5), sharey=True)
    if n_ks == 1:
        axes = [axes]

    # 阶段背景色
    def stage_color(kk):
        if kk <= 5:   return "#d5f5e3"
        if kk <= 15:  return "#fef9e7"
        return "#eaf4fb"

    def stage_label(kk):
        if kk <= 5:   return "Peak"
        if kk <= 15:  return "Decay"
        return "Plateau"

    for col, kk in enumerate(show_ks):
        ax = axes[col]
        ax.set_facecolor(stage_color(kk))

        data_per_model = []
        for res in valid:
            rs = [x["pearson_at_k"].get(kk, float("nan")) for x in res["per_perturbation"]]
            rs = [v for v in rs if not np.isnan(v)]
            data_per_model.append(rs)

        # violin + strip
        positions = list(range(n_models))
        parts = ax.violinplot(data_per_model, positions=positions,
                              showmedians=True, showextrema=False, widths=0.7)
        for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
            pc.set_facecolor(color)
            pc.set_alpha(0.55)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(2)

        for i, (rs, color) in enumerate(zip(data_per_model, colors)):
            jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(rs))
            ax.scatter([i + j for j in jitter], rs, color=color,
                       alpha=0.4, s=12, zorder=3)
            mean_val = np.mean(rs) if rs else float("nan")
            ax.text(i, ax.get_ylim()[0] if col == 0 else -1.05,
                    f"μ={mean_val:+.3f}", ha="center", fontsize=7.5,
                    color=color, fontweight="bold",
                    transform=ax.get_xaxis_transform() if col == 0 else ax.transData)

        ax.axhline(0, color="gray", linestyle="--", linewidth=0.9, alpha=0.7)
        ax.set_xticks(positions)
        ax.set_xticklabels([r["label"].replace(" ", "\n") for r in valid], fontsize=8)
        ax.set_title(f"k = {kk}\n({stage_label(kk)})", fontsize=9, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")

    axes[0].set_ylabel("Pearson r (per perturbation)", fontsize=11)

    # 在每个 violin 下方统一标注 mean（用 fig.text 避开 transform 问题）
    for col, (ax, kk) in enumerate(zip(axes, show_ks)):
        for i, (res, color) in enumerate(zip(valid, colors)):
            rs = [x["pearson_at_k"].get(kk, float("nan")) for x in res["per_perturbation"]]
            rs = [v for v in rs if not np.isnan(v)]
            mean_val = np.mean(rs) if rs else float("nan")
            ax.text(i, -1.12, f"μ={mean_val:+.3f}",
                    ha="center", va="top", fontsize=7.5,
                    color=color, fontweight="bold",
                    transform=ax.get_xaxis_transform())

    fig.suptitle("Per-Perturbation PCC Distribution at Selected k Values\n"
                 "(green=Signal Peak, yellow=Decay, blue=Plateau)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout(rect=[0, 0.06, 1, 1])
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

        max_k = max(ks)

        # k 较多时只保留代表性列，避免热图过窄
        if max_k > 20:
            key_ks = sorted(set(
                [k for k in ks if k <= 10] +
                [k for k in ks if k > 10 and k % 5 == 0]
            ))
            col_idx = [ks.index(k) for k in key_ks]
            display_matrix = matrix[:, col_idx]
            display_ks = key_ks
        else:
            display_matrix = matrix
            display_ks = ks

        vmax = max(abs(np.nanmax(display_matrix)), abs(np.nanmin(display_matrix)), 0.01)
        # 固定 vmax 上限到 1.0（PCC 范围），让颜色对应真实 PCC 量级
        vmax = min(vmax, 1.0)
        norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

        n_cols = len(display_ks)
        fig_h = max(6, len(pert_names) * 0.22)
        fig_w = max(6, min(n_cols * 0.55 + 2, 18))
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        im = ax.imshow(display_matrix, aspect="auto", cmap="RdBu_r", norm=norm)

        # 绘制阶段分割竖线
        stage_boundaries = [5.5, 15.5]  # 列值（x坐标）的分割点
        for boundary in stage_boundaries:
            col_positions = [i for i, k in enumerate(display_ks) if k <= boundary]
            if col_positions and col_positions[-1] < n_cols - 1:
                ax.axvline(col_positions[-1] + 0.5, color="black",
                           linewidth=1.5, linestyle="--", alpha=0.5)

        ax.set_xticks(range(n_cols))
        ax.set_xticklabels([f"k={k}" for k in display_ks], fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(pert_names)))
        ax.set_yticklabels(pert_names, fontsize=7)
        ax.set_xlabel("Number of top genes (by |Wilcoxon scores|)", fontsize=11)
        ax.set_title(
            f"Per-Perturbation PCC Heatmap — {res['label']}\n"
            f"(rows sorted by mean PCC  |  dashed lines = stage boundaries)",
            fontsize=11, fontweight="bold"
        )

        cbar = plt.colorbar(im, ax=ax, label="Pearson r", shrink=0.6)
        cbar.set_label("Pearson r", fontsize=10)

        # 顶部标注阶段名
        stage_spans = [(3, 5, "Peak"), (6, 15, "Decay"), (16, max_k, "Plateau")]
        for (s_start, s_end, s_name) in stage_spans:
            cols_in = [i for i, k in enumerate(display_ks) if s_start <= k <= s_end]
            if cols_in:
                mid = (cols_in[0] + cols_in[-1]) / 2
                ax.text(mid, -1.5, s_name, ha="center", va="bottom",
                        fontsize=8, color="#444444",
                        transform=ax.get_xaxis_transform())

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
    markers = ["o", "s", "^", "D"]

    valid_topk = [r for r in topk_results if "pearson_curve" in r]
    valid_pval = [r for r in pval_results if "pearson_mean" in r]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

    # ---- 左图：top-k PCC 曲线（SEM 阴影，从 k=3 开始） ----
    if valid_topk:
        all_ks_raw = sorted(valid_topk[0]["pearson_curve"].keys())
        ks = [k for k in all_ks_raw if k >= 3]
        max_k = max(ks)

        # 阶段背景
        for (x0, x1, fc) in [(3, min(5, max_k), "#d5f5e3"),
                              (6, min(15, max_k), "#fef9e7"),
                              (16, max_k, "#eaf4fb")]:
            if x0 <= max_k:
                ax1.axvspan(x0 - 0.5, x1 + 0.5, color=fc, alpha=0.6, zorder=0)

        for r, color, mkr in zip(valid_topk, colors, markers):
            curve = r["pearson_curve"]
            means = np.array([curve[k]["mean"] for k in ks], dtype=float)
            stds  = np.array([curve[k]["std"]  for k in ks], dtype=float)
            n_v   = np.array([curve[k].get("n_valid", 67) for k in ks], dtype=float)
            sem   = stds / np.sqrt(np.maximum(n_v, 1))
            ax1.plot(ks, means, marker=mkr, color=color, label=r["label"],
                     linewidth=2.5, markersize=5, zorder=3)
            ax1.fill_between(ks, means - sem, means + sem, color=color, alpha=0.2, zorder=2)
            # 标注 k=3 值
            if not np.isnan(means[0]):
                ax1.annotate(f"{means[0]:+.3f}", xy=(ks[0], means[0]),
                             xytext=(ks[0] + 0.5, means[0] + 0.012),
                             fontsize=8, color=color, fontweight="bold")

        ax1.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7, zorder=1)
        stride = 1 if max_k <= 20 else (5 if max_k <= 50 else 10)
        ax1.set_xticks([k for k in ks if k % stride == 0 or k == min(ks)])
        ax1.set_xlabel("Top-k genes (by |Wilcoxon scores|)", fontsize=11)
        ax1.set_ylabel("Mean Pearson r", fontsize=11)
        ax1.set_title("Method A: Top-k PCC Curve\n(SEM shading; Peak / Decay / Plateau zones)",
                      fontsize=11, fontweight="bold")
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.25)

    # ---- 右图：p-value 阈值法柱状图（SEM 误差棒而非 std）----
    if valid_pval:
        labels     = [r["label"] for r in valid_pval]
        means_pval = np.array([r["pearson_mean"] for r in valid_pval])
        stds_pval  = np.array([r["pearson_std"]  for r in valid_pval])
        ns_pval    = np.array([r.get("n_perturbations", 39) for r in valid_pval], dtype=float)
        sem_pval   = stds_pval / np.sqrt(np.maximum(ns_pval, 1))
        bar_colors = colors[:len(labels)]

        bars = ax2.bar(labels, means_pval, color=bar_colors, alpha=0.80,
                       yerr=sem_pval, capsize=6,
                       error_kw={"linewidth": 1.5, "capthick": 1.5})
        ax2.axhline(0, color="gray", linestyle="--", linewidth=1.0, alpha=0.7)
        for bar, m, n in zip(bars, means_pval, ns_pval):
            offset = 0.004 if m >= 0 else -0.006
            ax2.text(bar.get_x() + bar.get_width() / 2, m + offset,
                     f"{m:.4f}\n(N={int(n)})",
                     ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax2.set_ylabel("Mean Pearson r", fontsize=11)
        ax2.set_title(f"Method B: p-value Threshold  (adj p < {pval_cutoff})\n"
                      f"Error bars = SEM  |  Skips {68 - int(ns_pval.max())}/68 perturbations",
                      fontsize=11, fontweight="bold")
        ax2.grid(True, alpha=0.25, axis="y")

    plt.suptitle("Evaluation Method Comparison: Top-k PCC vs. p-value Threshold",
                 fontsize=13, fontweight="bold", y=1.02)
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

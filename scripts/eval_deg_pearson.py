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


def parse_args():
    p = argparse.ArgumentParser(description="DEG-based Pearson r evaluation")
    p.add_argument("--checkpoint", required=True, help="主 checkpoint (.ckpt)")
    p.add_argument("--baseline", default=None, help="基线 checkpoint（对比用）")
    p.add_argument("--config", required=True, help="config yaml 路径")
    p.add_argument("--h5ad", default=DEFAULT_H5AD, help="h5ad 文件路径（带扰动标签）")
    p.add_argument("--pert_col", default="gene", help="obs 列名（扰动标签）")
    p.add_argument("--ctrl_label", default="non-targeting", help="对照组标签")
    p.add_argument("--pval_cutoff", type=float, default=0.05, help="adj p-value 阈值")
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

    print(f"  加载 checkpoint: {checkpoint_path}")
    model = StateEmbeddingModel.load_from_checkpoint(
        checkpoint_path, dropout=0.0, strict=False, cfg=cfg, map_location="cpu"
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
    对每个扰动：
      1. 找出 DE 基因（Wilcoxon test，adj p < pval_cutoff）
      2. 在 DE 基因上计算 Pearson r（预测 vs 实际）

    推理方式：encode 细胞 → 平均 CLS embedding → decode 全部基因
    避免 H5adSentenceDataset 每 batch 只采样部分基因导致的索引越界。
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
    print(f"  获取全基因 embedding ({n_total_genes} genes)...")
    with torch.no_grad():
        gene_embs = model.get_gene_embedding(gene_names).detach().cpu()  # (n_genes, d_model)

    # ---- 预计算 ctrl CLS embedding 均值（循环外，只算一次）----
    print(f"  编码对照组细胞 ({len(ctrl_indices)} cells)...")
    ctrl_cls = encode_cells_to_cls(model, cfg, h5_path, list(ctrl_indices), args, device)
    ctrl_cls_mean = torch.tensor(ctrl_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)  # (1, 512)

    # ctrl 全基因预测（循环外）
    pred_ctrl_all = predict_from_emb(model, ctrl_cls_mean, gene_embs, args.read_depth, device)
    # pred_ctrl_all: (n_total_genes,)

    actual_ctrl_mean_all = adata.X[ctrl_indices].mean(axis=0)  # (n_total_genes,)

    pearson_list = []
    results_per_pert = []
    n_skipped = 0

    for pert in tqdm(pert_unique, desc=f"  [{label}] DEG Pearson"):
        pert_mask = pert_labels == pert
        pert_indices = np.where(pert_mask)[0]

        if len(pert_indices) < 5:
            n_skipped += 1
            continue

        if len(pert_indices) > args.max_cells_per_group:
            pert_indices = pert_indices[:args.max_cells_per_group]

        # ---- Step 1: 找 DE 基因（Wilcoxon, 用真实数据） ----
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
            de_df = sc.get.rank_genes_groups_df(
                adata_combined,
                group="perturbed",
                pval_cutoff=args.pval_cutoff,
            )
        except Exception:
            n_skipped += 1
            continue

        if len(de_df) < args.min_de_genes:
            n_skipped += 1
            continue

        # DE 基因名 → 索引（在 adata 全基因中的位置）
        de_gene_names = set(de_df["names"].tolist())
        de_gene_indices = np.array([i for i, g in enumerate(gene_names) if g in de_gene_names])

        if len(de_gene_indices) < args.min_de_genes:
            n_skipped += 1
            continue

        # ---- Step 2: 编码扰动组细胞 → 平均 CLS → decode 全基因 ----
        try:
            pert_cls = encode_cells_to_cls(model, cfg, h5_path, list(pert_indices), args, device)
        except Exception as e:
            print(f"  编码失败 ({pert}): {e}")
            n_skipped += 1
            continue

        pert_cls_mean = torch.tensor(pert_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
        pred_pert_all = predict_from_emb(model, pert_cls_mean, gene_embs, args.read_depth, device)
        # pred_pert_all: (n_total_genes,)

        # ---- Step 3: 在 DE 基因上计算 Pearson r ----
        actual_ctrl_mean = actual_ctrl_mean_all[de_gene_indices]
        actual_pert_mean = adata.X[pert_indices][:, de_gene_indices].mean(axis=0)
        actual_logfc = actual_pert_mean - actual_ctrl_mean  # (n_de_genes,)

        # predicted log-FC on DE genes（用全基因预测结果取 DE 子集）
        pred_logfc = pred_pert_all[de_gene_indices] - pred_ctrl_all[de_gene_indices]

        valid = ~(np.isnan(actual_logfc) | np.isnan(pred_logfc))
        if valid.sum() < args.min_de_genes:
            n_skipped += 1
            continue

        try:
            r, pval = pearsonr(pred_logfc[valid], actual_logfc[valid])
        except Exception:
            n_skipped += 1
            continue

        if np.isnan(r):
            n_skipped += 1
            continue

        pearson_list.append(float(r))
        results_per_pert.append({
            "perturbation": pert,
            "n_de_genes": len(de_gene_indices),
            "n_pert_cells": len(pert_indices),
            "pearson_r": float(r),
            "pval": float(pval),
        })

    print(f"\n  完成 {len(pearson_list)} 个扰动（跳过 {n_skipped} 个）")

    if not pearson_list:
        return {"label": label, "error": "no valid perturbations"}

    mean_r = float(np.mean(pearson_list))
    median_r = float(np.median(pearson_list))
    std_r = float(np.std(pearson_list))

    print(f"  DEG Pearson r: mean={mean_r:.4f}, median={median_r:.4f}, std={std_r:.4f}")
    print(f"  正相关扰动比例: {sum(r > 0 for r in pearson_list) / len(pearson_list):.1%}")

    return {
        "label": label,
        "n_perturbations": len(pearson_list),
        "pearson_mean": mean_r,
        "pearson_median": median_r,
        "pearson_std": std_r,
        "frac_positive": float(sum(r > 0 for r in pearson_list) / len(pearson_list)),
        "per_perturbation": results_per_pert,
    }


# ============================================================================
# 绘图
# ============================================================================

def plot_comparison(results_dict, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [r["label"] for r in results_dict if "pearson_mean" in r]
    means = [r["pearson_mean"] for r in results_dict if "pearson_mean" in r]
    stds = [r["pearson_std"] for r in results_dict if "pearson_mean" in r]

    if len(names) == 0:
        return

    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"][:len(names)]
    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2.5), 5))
    bars = ax.bar(names, means, yerr=stds, color=colors,
                  capsize=6, alpha=0.85, edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_ylabel("DEG Pearson r (mean ± std)", fontsize=12)
    ax.set_title("HepG2 Zero-Shot DEG Pearson r\n(Wilcoxon adj p < 0.05 genes only)", fontsize=12)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.005 + (0.01 if v >= 0 else -0.03),
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "deg_pearson_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {path}")


def plot_per_pert_scatter(result, output_dir, other_result=None):
    """扰动级 Pearson r 分布（箱线图或散点图）。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))

    def add_data(res, color, label):
        rs = [x["pearson_r"] for x in res["per_perturbation"]]
        ax.hist(rs, bins=30, alpha=0.6, color=color, label=f"{label} (mean={np.mean(rs):.4f})",
                edgecolor="white", linewidth=0.5)

    add_data(result, "#3498DB", result["label"])
    if other_result and "per_perturbation" in other_result:
        add_data(other_result, "#E74C3C", other_result["label"])

    ax.axvline(0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    ax.set_xlabel("Pearson r (per perturbation)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Per-Perturbation DEG Pearson r Distribution", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "deg_pearson_distribution.png")
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
        output_dir = str(Path(args.checkpoint).parent / "eval_deg_pearson")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    print(f"数据文件: {args.h5ad}")
    print(f"扰动列: {args.pert_col}  对照标签: {args.ctrl_label}")
    print(f"adj p 阈值: {args.pval_cutoff}  最少 DE 基因: {args.min_de_genes}")

    all_results = []

    # ---- 主 checkpoint ----
    print(f"\n{'='*60}\n[1] STATE+MMD\n{'='*60}")
    model, device = load_state_model(args.checkpoint, cfg)
    res_main = eval_deg_pearson(model, cfg, args.h5ad, args, device, label="STATE+MMD")
    if res_main:
        all_results.append(res_main)
    del model
    torch.cuda.empty_cache()

    # ---- 基线 checkpoint ----
    if args.baseline:
        print(f"\n{'='*60}\n[2] Baseline\n{'='*60}")
        model_base, device = load_state_model(args.baseline, cfg)
        res_base = eval_deg_pearson(model_base, cfg, args.h5ad, args, device, label="Baseline")
        if res_base:
            all_results.append(res_base)
        del model_base
        torch.cuda.empty_cache()

    # ---- 汇总 ----
    print("\n" + "=" * 70)
    print("DEG Pearson r 汇总（谢老师方法）")
    print("=" * 70)
    for res in all_results:
        if "pearson_mean" in res:
            print(f"  {res['label']:20s}  "
                  f"Mean={res['pearson_mean']:.4f} ± {res['pearson_std']:.4f}  "
                  f"Median={res['pearson_median']:.4f}  "
                  f"N={res['n_perturbations']}  "
                  f"Positive={res['frac_positive']:.1%}")
    print("=" * 70)

    # ---- 绘图 ----
    if len(all_results) >= 1:
        plot_comparison(all_results, output_dir)
    if len(all_results) >= 1 and "per_perturbation" in all_results[0]:
        other = all_results[1] if len(all_results) > 1 else None
        plot_per_pert_scatter(all_results[0], output_dir, other)

    # ---- 保存 JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"deg_pearson_{ts}.json")
    with open(json_path, "w") as f:
        # per_perturbation 可能很长，但保留供后续分析
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")
    print("评估完成")


if __name__ == "__main__":
    main()

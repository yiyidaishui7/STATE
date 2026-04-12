#!/usr/bin/env python3
"""
eval_des.py — DES (Drug Effect Signature) gene overlap evaluation

STATE 原生 DES 指标：
  对每个扰动条件，预测 top-k 差异表达基因，与 Wilcoxon 检验得到的真实 top-k DEG 求交集。
  DES = |predicted_top_k ∩ true_top_k| / k

预测方式：
  对照组 CLS embedding → binary_decoder → ctrl_scores (n_genes,)
  扰动组 CLS embedding → binary_decoder → pert_scores (n_genes,)
  delta = |pert_scores - ctrl_scores|，取 top-k 作为预测 DEG

真实 DEG：
  Wilcoxon rank-sum test (BH 校正)，按 |log-FC| 排序取前 top-k 显著基因

使用方法（服务器 ~/state/src 目录）：

    # HepG2 零样本（MMD 模型 vs Baseline）
    python ../scripts/eval_des.py \\
        --checkpoint /path/to/mmd_aae_pretrain/last.ckpt \\
        --baseline /path/to/baseline/last.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --h5ad /path/to/hepg2.h5 \\
        --pert_col gene \\
        --ctrl_label non-targeting

    # hepg2_val_with_controls.h5ad（target_gene 列）
    python ../scripts/eval_des.py \\
        --checkpoint /path/to/mmd_aae_pretrain/last.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --h5ad /path/to/hepg2_val_with_controls.h5ad \\
        --pert_col target_gene \\
        --ctrl_label non-targeting

    # LOO 实验
    python ../scripts/eval_des.py \\
        --checkpoint /path/to/loo_jurkat/last.ckpt \\
        --config ../configs/loo_jurkat_config.yaml \\
        --h5ad /path/to/jurkat.h5 \\
        --pert_col gene
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
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
DEFAULT_H5AD = f"{BASE_DIR}/competition_support_set/hepg2.h5"
FIGURES_DIR = f"{BASE_DIR}/figures"


def parse_args():
    p = argparse.ArgumentParser(description="DES gene overlap evaluation")
    p.add_argument("--checkpoint", required=True, help="主 checkpoint (.ckpt)")
    p.add_argument("--baseline", default=None, help="基线 checkpoint（对比用）")
    p.add_argument("--config", required=True, help="config yaml 路径")
    p.add_argument("--h5ad", default=DEFAULT_H5AD, help="h5/h5ad 文件路径（带扰动标签）")
    p.add_argument("--pert_col", default="gene", help="obs 列名（扰动标签）")
    p.add_argument("--ctrl_label", default="non-targeting", help="对照组标签")
    p.add_argument("--top_k", type=int, default=50,
                   help="top-k 基因数（DES 计算窗口，默认 50）")
    p.add_argument("--pval_cutoff", type=float, default=0.20,
                   help="Wilcoxon adj p 阈值（筛选真实 DEG，默认 0.20）")
    p.add_argument("--min_de_genes", type=int, default=3,
                   help="最少真实 DEG 数才评估（默认 3）")
    p.add_argument("--max_cells_per_group", type=int, default=200,
                   help="每组最多取多少细胞（加速，默认 200）")
    p.add_argument("--batch_size", type=int, default=16, help="推理 batch size")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output", default=None, help="输出目录")
    p.add_argument("--read_depth", type=float, default=4.0, help="RDA read depth")
    return p.parse_args()


# ============================================================================
# 数据加载
# ============================================================================

def load_adata(h5_path):
    """加载 h5/h5ad 文件，返回 anndata.AnnData，X 为 log1p counts (dense float32)。"""
    import anndata as ad
    from scipy.sparse import issparse

    try:
        adata = ad.read_h5ad(h5_path)
    except Exception:
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

    if "protein_embeds_dict" in ckpt:
        model.protein_embeds = ckpt["protein_embeds_dict"]

    model.eval()
    return model, device


# ============================================================================
# 推理工具（与 eval_deg_pearson.py 相同，复用避免依赖）
# ============================================================================

def encode_cells_to_cls(model, cfg, h5_path, cell_indices, args, device):
    """对指定细胞编码，返回 CLS embedding np.ndarray (n_cells, 512)。"""
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

    return np.concatenate(all_cls, axis=0)


def predict_from_emb(model, cell_emb, gene_embs, read_depth, device):
    """给定单个 CLS embedding 和基因 embedding，返回全基因预测分数 (n_genes,)。"""
    batch_size = 512
    n_genes = gene_embs.size(0)
    all_scores = []

    cell_emb = cell_emb.reshape(-1)
    z_base = cell_emb.unsqueeze(0).unsqueeze(0)

    for start in range(0, n_genes, batch_size):
        end = min(start + batch_size, n_genes)
        gene_batch = gene_embs[start:end].to(device)
        n_batch = gene_batch.size(0)

        z = z_base.expand(1, n_batch, -1)
        g = gene_batch.unsqueeze(0)

        if model.z_dim_rd == 1:
            rd = torch.full((1, n_batch, 1), read_depth, device=device)
            combine = torch.cat([g, z, rd], dim=2)
        else:
            combine = torch.cat([g, z], dim=2)

        with torch.no_grad():
            scores = model.binary_decoder(combine).squeeze(-1).squeeze(0)
            all_scores.append(scores.cpu().float().numpy())

    return np.concatenate(all_scores)


# ============================================================================
# 核心评估：DES gene overlap
# ============================================================================

def eval_des(model, cfg, h5_path, args, device, label="model"):
    """
    对每个扰动：
      1. 获取真实 top-k DEG（Wilcoxon + BH 校正，按 |log-FC| 排序）
      2. 获取预测 top-k 基因（|pert_score - ctrl_score| 最大的 k 个）
      3. DES = |predicted ∩ true| / k

    返回：
      {
        "label": str,
        "mean_des": float,          # 所有扰动 DES 均值
        "median_des": float,
        "std_des": float,
        "n_perturbations": int,     # 有效扰动数
        "k": int,
        "per_perturbation": [{"perturbation", "n_de_genes", "n_pert_cells", "des"}]
      }
    """
    import scanpy as sc
    import anndata as ad

    k = args.top_k
    adata = load_adata(h5_path)

    if args.pert_col not in adata.obs.columns:
        print(f"  列 '{args.pert_col}' 不存在。可用列: {list(adata.obs.columns)}")
        return None

    pert_labels = adata.obs[args.pert_col].astype(str).values
    ctrl_mask = pert_labels == args.ctrl_label
    n_ctrl = ctrl_mask.sum()

    if n_ctrl == 0:
        print(f"  未找到对照组 '{args.ctrl_label}'。可用标签: {np.unique(pert_labels)[:10]}")
        return None

    print(f"\n  对照细胞: {n_ctrl}")
    pert_unique = sorted([p for p in np.unique(pert_labels) if p != args.ctrl_label])
    print(f"  扰动类型数: {len(pert_unique)}")

    ctrl_indices = np.where(ctrl_mask)[0]
    if len(ctrl_indices) > args.max_cells_per_group:
        ctrl_indices = ctrl_indices[:args.max_cells_per_group]

    gene_names = list(adata.var_names)
    n_total_genes = len(gene_names)
    print(f"  总基因数: {n_total_genes}")

    # 基因 embedding（全局计算一次）
    print(f"  计算基因 embedding ({n_total_genes} genes)...")
    with torch.no_grad():
        gene_embs = model.get_gene_embedding(gene_names).detach().cpu()

    # 对照组 CLS embedding 均值 → 对照预测分数
    print(f"  编码对照组细胞 ({len(ctrl_indices)} cells)...")
    ctrl_cls = encode_cells_to_cls(model, cfg, h5_path, list(ctrl_indices), args, device)
    ctrl_cls_mean = torch.tensor(ctrl_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
    pred_ctrl_scores = predict_from_emb(model, ctrl_cls_mean, gene_embs, args.read_depth, device)
    # pred_ctrl_scores: (n_total_genes,)

    des_list = []
    results_per_pert = []
    n_skipped = 0

    for pert in tqdm(pert_unique, desc=f"  [{label}] DES"):
        pert_mask = pert_labels == pert
        pert_indices = np.where(pert_mask)[0]

        if len(pert_indices) < 5:
            n_skipped += 1
            continue

        if len(pert_indices) > args.max_cells_per_group:
            pert_indices = pert_indices[:args.max_cells_per_group]

        # ---- Step 1: 真实 top-k DEG（Wilcoxon） ----
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

        # 按 |log-FC| 降序取 top-k 真实 DEG
        de_df = de_df.copy()
        de_df["abs_logfc"] = de_df["logfoldchanges"].abs()
        de_df = de_df.sort_values("abs_logfc", ascending=False)
        true_top_k = set(de_df["names"].iloc[:k].tolist())

        # ---- Step 2: 预测 top-k 基因（|pert_score - ctrl_score| 最大） ----
        try:
            pert_cls = encode_cells_to_cls(model, cfg, h5_path, list(pert_indices), args, device)
        except Exception as e:
            print(f"  编码失败 ({pert}): {e}")
            n_skipped += 1
            continue

        pert_cls_mean = torch.tensor(pert_cls.mean(axis=0), dtype=torch.float32).unsqueeze(0).to(device)
        pred_pert_scores = predict_from_emb(model, pert_cls_mean, gene_embs, args.read_depth, device)

        delta = np.abs(pred_pert_scores - pred_ctrl_scores)
        pred_top_k_indices = np.argsort(delta)[::-1][:k]
        pred_top_k = set(gene_names[i] for i in pred_top_k_indices)

        # ---- Step 3: DES = 交集 / k ----
        overlap = len(true_top_k & pred_top_k)
        des = overlap / k

        des_list.append(des)
        results_per_pert.append({
            "perturbation": pert,
            "n_de_genes": len(de_df),
            "n_pert_cells": len(pert_indices),
            "true_top_k_size": len(true_top_k),
            "overlap": overlap,
            "des": des,
        })

    print(f"\n  完成 {len(des_list)} 个扰动（跳过 {n_skipped} 个）")

    if not des_list:
        return {"label": label, "error": "no valid perturbations"}

    mean_des = float(np.mean(des_list))
    median_des = float(np.median(des_list))
    std_des = float(np.std(des_list))

    print(f"  DES (top-{k} overlap):  mean={mean_des:.4f},  median={median_des:.4f},  std={std_des:.4f}")
    print(f"  DES > 0 比例: {sum(d > 0 for d in des_list) / len(des_list):.1%}")

    return {
        "label": label,
        "k": k,
        "n_perturbations": len(des_list),
        "mean_des": mean_des,
        "median_des": median_des,
        "std_des": std_des,
        "frac_positive": float(sum(d > 0 for d in des_list) / len(des_list)),
        "per_perturbation": results_per_pert,
    }


# ============================================================================
# 绘图
# ============================================================================

def plot_comparison(results_list, output_dir, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results_list if "mean_des" in r]
    if not valid:
        return

    names = [r["label"] for r in valid]
    means = [r["mean_des"] for r in valid]
    stds = [r["std_des"] for r in valid]
    colors = ["#E74C3C", "#3498DB", "#2ECC71", "#F39C12"][:len(names)]

    fig, ax = plt.subplots(figsize=(max(6, len(names) * 2.5), 5))
    bars = ax.bar(names, means, yerr=stds, color=colors,
                  capsize=6, alpha=0.85, edgecolor="white", linewidth=0.5)
    # 随机基线：随机选 k 个基因，期望 overlap = k² / n_genes ≈ 0
    random_baseline = k / len(valid[0].get("per_perturbation", [{}])[0].get("n_de_genes", 18000) or 18000)
    ax.axhline(random_baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Random baseline (~{random_baseline:.4f})", alpha=0.7)
    ax.set_ylabel(f"DES (top-{k} gene overlap)", fontsize=12)
    ax.set_title(f"DES Evaluation (top-{k})\n(Wilcoxon adj p < 0.20 true DEGs)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.001,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(output_dir, "des_comparison.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  图已保存: {path}")


def plot_per_pert_distribution(results_list, output_dir, k):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [r for r in results_list if "per_perturbation" in r]
    if not valid:
        return

    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#F39C12"]
    fig, ax = plt.subplots(figsize=(8, 5))

    for res, color in zip(valid, colors):
        des_vals = [x["des"] for x in res["per_perturbation"]]
        ax.hist(des_vals, bins=20, alpha=0.6, color=color,
                label=f"{res['label']} (mean={res['mean_des']:.4f})",
                edgecolor="white", linewidth=0.5)

    ax.set_xlabel(f"DES (top-{k} overlap per perturbation)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title(f"Per-Perturbation DES Distribution (top-{k})", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, "des_distribution.png")
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
        dataset_name = Path(args.h5ad).stem
        output_dir = str(Path(FIGURES_DIR) / "des" / dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n输出目录: {output_dir}")
    print(f"数据文件: {args.h5ad}")
    print(f"扰动列: {args.pert_col}  对照标签: {args.ctrl_label}")
    print(f"top-k: {args.top_k}  adj p 阈值: {args.pval_cutoff}  最少 DE 基因: {args.min_de_genes}")

    all_results = []

    # ---- 主 checkpoint ----
    print(f"\n{'='*60}\n处理: STATE+MMD\n{'='*60}")
    model, device = load_state_model(args.checkpoint, cfg)
    res_main = eval_des(model, cfg, args.h5ad, args, device, label="STATE+MMD")
    if res_main:
        all_results.append(res_main)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- 基线 checkpoint（可选）----
    if args.baseline:
        print(f"\n{'='*60}\n处理: Baseline\n{'='*60}")
        model_b, device_b = load_state_model(args.baseline, cfg)
        res_base = eval_des(model_b, cfg, args.h5ad, args, device_b, label="Baseline")
        if res_base:
            all_results.append(res_base)
        del model_b
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 汇总打印 ----
    print("\n" + "=" * 70)
    print(f"DES (top-{args.top_k} gene overlap) 汇总")
    print("=" * 70)
    for r in all_results:
        if "mean_des" in r:
            print(f"  {r['label']:20s}  mean={r['mean_des']:.4f} ± {r['std_des']:.4f}  "
                  f"median={r['median_des']:.4f}  N={r['n_perturbations']}  "
                  f"DES>0={r['frac_positive']:.1%}")
    print("=" * 70)

    # ---- 绘图 ----
    if all_results:
        plot_comparison(all_results, output_dir, args.top_k)
        plot_per_pert_distribution(all_results, output_dir, args.top_k)

    # ---- 保存 JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"des_results_{ts}.json")
    with open(json_path, "w") as f:
        # per_perturbation 列表可能较大，保留但截断到前 200 条
        save_results = []
        for r in all_results:
            r_save = {k: v for k, v in r.items() if k != "per_perturbation"}
            if "per_perturbation" in r:
                r_save["per_perturbation"] = r["per_perturbation"][:200]
            save_results.append(r_save)
        json.dump(save_results, f, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")

    # ---- CSV 汇总 ----
    import pandas as pd
    summary_rows = []
    for r in all_results:
        if "mean_des" in r:
            summary_rows.append({
                "label": r["label"],
                "k": r["k"],
                "mean_des": r["mean_des"],
                "median_des": r["median_des"],
                "std_des": r["std_des"],
                "n_perturbations": r["n_perturbations"],
                "frac_positive": r["frac_positive"],
            })
    if summary_rows:
        csv_path = os.path.join(output_dir, f"des_summary_{ts}.csv")
        pd.DataFrame(summary_rows).to_csv(csv_path, index=False)
        print(f"CSV 已保存: {csv_path}")

    print("\n✅ DES 评估完成！")


if __name__ == "__main__":
    main()

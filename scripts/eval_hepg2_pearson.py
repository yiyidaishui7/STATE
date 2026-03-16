#!/usr/bin/env python3
"""
eval_hepg2_pearson.py — HepG2 零样本 Pearson 相关系数评估

评估方法（两种模式）：
  模式 A: 细胞级 Pearson（per-cell reconstruction quality）
    - 对每个 HepG2 细胞：encode → CLS embedding → decode 全部基因的预测分数
    - 预测分数 vs 实际 log1p(counts) 的 Pearson r
    - 反映模型对零样本域的基因表达重建能力

  模式 B: 扰动级 Pearson（perturbation effect prediction, 若有扰动标签）
    - 对每种基因敲除：计算平均扰动表达 - 平均对照表达（真实 log-FC）
    - 用 CLS embedding 解码预测分数，与真实 log-FC 的 Pearson r
    - 反映模型对零样本扰动效应的预测能力

使用方法（在服务器 ~/state/src 目录）：

    # 模式 A（细胞级）
    python ../scripts/eval_hepg2_pearson.py \\
        --checkpoint /path/to/mmd_ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --hepg2 /path/to/hepg2.h5

    # 对比两个 checkpoint
    python ../scripts/eval_hepg2_pearson.py \\
        --checkpoint /path/to/mmd_ckpt.ckpt \\
        --baseline /path/to/baseline_ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --hepg2 /path/to/hepg2.h5

    # 模式 B（需要扰动标签，指定 obs 列名）
    python ../scripts/eval_hepg2_pearson.py \\
        --checkpoint /path/to/mmd_ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --hepg2 /path/to/hepg2.h5 \\
        --pert_col gene \\
        --ctrl_label ctrl
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
import h5py
from torch import nn
from torch.utils.data import DataLoader
from scipy.stats import pearsonr
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
HEPG2_H5 = f"{BASE_DIR}/competition_support_set/hepg2.h5"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="STATE+MMD checkpoint (.ckpt)")
    p.add_argument("--baseline", default=None, help="基线 checkpoint（对比用）")
    p.add_argument("--config", required=True, help="mmd_aae_config.yaml 路径")
    p.add_argument("--hepg2", default=HEPG2_H5, help="HepG2 h5/h5ad 文件路径")
    p.add_argument("--output", default=None, help="输出目录")
    p.add_argument("--max_cells", type=int, default=2000, help="最大细胞数（模式A加速）")
    p.add_argument("--batch_size", type=int, default=16, help="推理 batch size（解码开销大）")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--read_depth", type=float, default=4.0,
                   help="RDA 模式下的 read depth（默认 4.0）")
    # 模式 B 参数
    p.add_argument("--pert_col", default=None,
                   help="obs 列名（扰动标签）。若不指定，仅运行模式A")
    p.add_argument("--ctrl_label", default="ctrl",
                   help="对照组标签（在 pert_col 中）")
    p.add_argument("--top_genes", type=int, default=50,
                   help="模式B中用于 Pearson r 的差异基因数（top DE genes）")
    return p.parse_args()


# ============================================================================
# 模型加载（与 eval_state_domain_alignment.py 相同）
# ============================================================================

def read_h5_shape(h5_path):
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["X"].attrs)
        if "shape" in attrs:
            n_cells, n_genes = int(attrs["shape"][0]), int(attrs["shape"][1])
        elif attrs.get("encoding-type") in ("csr_matrix", "csc_matrix"):
            n_cells = f["X"]["indptr"].shape[0] - 1
            n_genes = int(attrs["shape"][1]) if "shape" in attrs else 18080
        elif hasattr(f["X"], "shape") and len(f["X"].shape) == 2:
            n_cells, n_genes = f["X"].shape[0], f["X"].shape[1]
        else:
            raise ValueError(f"Cannot determine shape of {h5_path}")
    return int(n_cells), int(n_genes)


def load_state_model(checkpoint_path, cfg):
    from state.emb.nn.model import StateEmbeddingModel
    from state.emb.train.trainer import get_embeddings

    print(f"  Loading: {checkpoint_path}")
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
    model.eval()
    return model, device


# ============================================================================
# 模式 A: 细胞级 Pearson（重建质量）
# ============================================================================

def eval_mode_a_cell_pearson(model, cfg, h5_path, args, device):
    """
    对每个细胞: 通过 shared_step 内的 forward 计算预测分数 (decs) 和目标 (Y)。
    返回每个细胞的 Pearson r 列表以及平均值。

    注意：这里直接在 DataLoader 的 batch 上运行推理，不走完整的 encode-decode 分两步，
    因为 binary_decoder 的输入结构和 shared_step 中一致。
    """
    from state.emb.data import H5adSentenceDataset, VCIDatasetSentenceCollator

    n_cells_total, n_genes = read_h5_shape(h5_path)
    n_cells = min(n_cells_total, args.max_cells)
    domain_name = Path(h5_path).stem  # e.g. "hepg2"
    print(f"  HepG2: ({n_cells}/{n_cells_total} cells, {n_genes} genes)")

    dataset = H5adSentenceDataset(
        cfg, test=True,
        datasets=[domain_name],
        shape_dict={domain_name: (n_cells, n_genes)},
    )
    dataset.dataset_path_map = {domain_name: h5_path}

    collator = VCIDatasetSentenceCollator(cfg, is_train=False)
    collator.cfg = cfg

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=args.num_workers,
        persistent_workers=(args.num_workers > 0),
    )

    pearson_per_cell = []
    total_processed = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Mode A inference"):
            # 与 shared_step 对齐：获取 X (gene embs), Y (targets), embs (CLS)
            X, Y, batch_weights, embs, dataset_emb = model._compute_embedding_for_batch(batch)

            # 构造 combine（与 shared_step 中相同逻辑）
            z = embs.unsqueeze(1).repeat(1, X.shape[1], 1)

            if model.z_dim_rd == 1:  # rda=True
                # mu = mean non-zero expression of Y（对于 HepG2 零样本，用实际 Y 来计算 mu）
                Y_float = Y.float()
                mu = torch.nan_to_num(
                    torch.nanmean(
                        Y_float.masked_fill(Y_float == 0, float("nan")), dim=1
                    ), nan=0.0
                )
                reshaped_counts = mu.unsqueeze(1).unsqueeze(2).repeat(1, X.shape[1], 1)
                combine = torch.cat((X, z, reshaped_counts), dim=2)
            else:
                combine = torch.cat((X, z), dim=2)

            decs = model.binary_decoder(combine).squeeze(-1)  # (B, n_genes)

            # Pearson r per cell
            decs_np = decs.detach().cpu().float().numpy()  # (B, n_genes)
            Y_np = Y.detach().cpu().float().numpy()          # (B, n_genes)

            for i in range(len(decs_np)):
                pred = decs_np[i]
                gt = Y_np[i]
                # 只用有值的基因（Y != 0 or pred != 0）
                valid = ~(np.isnan(pred) | np.isnan(gt))
                if valid.sum() < 10:
                    continue
                r, _ = pearsonr(pred[valid], gt[valid])
                if not np.isnan(r):
                    pearson_per_cell.append(float(r))

            total_processed += len(decs_np)
            if total_processed >= args.max_cells:
                break

    mean_r = float(np.mean(pearson_per_cell)) if pearson_per_cell else float("nan")
    median_r = float(np.median(pearson_per_cell)) if pearson_per_cell else float("nan")
    return {
        "mode": "A_cell_level",
        "n_cells": len(pearson_per_cell),
        "pearson_mean": mean_r,
        "pearson_median": median_r,
        "pearson_std": float(np.std(pearson_per_cell)) if pearson_per_cell else float("nan"),
    }


# ============================================================================
# 模式 B: 扰动级 Pearson（零样本扰动预测）
# ============================================================================

def load_hepg2_adata(h5_path):
    """加载 HepG2 h5ad 文件，返回 anndata.AnnData 对象。"""
    import anndata
    try:
        adata = anndata.read_h5ad(h5_path)
        # 确保 X 为 dense numpy array（log1p counts）
        from scipy.sparse import issparse
        if issparse(adata.X):
            adata.X = adata.X.toarray()
        return adata
    except Exception as e:
        print(f"  无法以 h5ad 格式读取 ({e})，尝试直接读 X...")
        with h5py.File(h5_path, "r") as f:
            from scipy.sparse import issparse, csr_matrix
            if "indices" in f["X"]:
                # sparse CSR
                data = f["X"]["data"][:]
                indices = f["X"]["indices"][:]
                indptr = f["X"]["indptr"][:]
                attrs = dict(f["X"].attrs)
                n_cells = indptr.shape[0] - 1
                n_genes = int(attrs["shape"][1]) if "shape" in attrs else indices.max() + 1
                X = csr_matrix((data, indices, indptr), shape=(n_cells, n_genes)).toarray()
            else:
                X = f["X"][:]
        import anndata
        return anndata.AnnData(X=X)


def eval_mode_b_perturbation_pearson(model, cfg, h5_path, args, device):
    """
    扰动级 Pearson r：
    1. 加载 HepG2 h5ad（带 obs[pert_col] 扰动标签）
    2. 识别对照组（ctrl_label）和各扰动组
    3. 对照组 CLS embedding → decode → 得到每个扰动的"对照预测"
    4. 真实扰动 log-FC = mean(pert) - mean(ctrl) in log1p space
    5. Pearson r(predicted_scores, actual_logFC)
    """
    from state.emb.data import H5adSentenceDataset, VCIDatasetSentenceCollator
    from state.emb.inference import Inference

    print(f"\n  [Mode B] 扰动级 Pearson r（pert_col='{args.pert_col}', ctrl='{args.ctrl_label}'）")

    adata = load_hepg2_adata(h5_path)

    if args.pert_col not in adata.obs.columns:
        print(f"  ❌ obs 列 '{args.pert_col}' 不存在。可用列: {list(adata.obs.columns)}")
        return None

    pert_labels = adata.obs[args.pert_col].values
    ctrl_mask = pert_labels == args.ctrl_label
    n_ctrl = ctrl_mask.sum()

    if n_ctrl == 0:
        print(f"  ❌ 未找到对照组（'{args.ctrl_label}'）")
        return None

    print(f"  对照细胞: {n_ctrl}")

    # log1p 标准化（若还没做）
    X_raw = adata.X.astype(np.float32)
    if X_raw.max() > 100:  # 可能是 raw counts
        X_log = np.log1p(X_raw)
    else:
        X_log = X_raw

    # 控制组平均表达
    ctrl_mean = X_log[ctrl_mask].mean(axis=0)  # (n_genes,)

    # 所有扰动
    pert_unique = [p for p in np.unique(pert_labels) if p != args.ctrl_label]
    print(f"  扰动数: {len(pert_unique)}")

    # 用 Inference 获取对照组 CLS embeddings
    # 用 H5adSentenceDataset 加载对照组细胞
    domain_name = Path(h5_path).stem
    n_cells_total, n_genes = read_h5_shape(h5_path)

    dataset_full = H5adSentenceDataset(
        cfg, test=True,
        datasets=[domain_name],
        shape_dict={domain_name: (n_cells_total, n_genes)},
    )
    dataset_full.dataset_path_map = {domain_name: h5_path}

    collator = VCIDatasetSentenceCollator(cfg, is_train=False)
    collator.cfg = cfg

    # 提取全部 HepG2 细胞的 CLS embedding
    all_z = []
    loader_full = DataLoader(
        dataset_full, batch_size=32, shuffle=False,
        collate_fn=collator, num_workers=args.num_workers,
    )
    with torch.no_grad():
        for batch in tqdm(loader_full, desc="  Encoding HepG2"):
            _, _, _, emb, _ = model._compute_embedding_for_batch(batch)
            all_z.append(emb.detach().cpu().float())
    all_z = torch.cat(all_z, dim=0)  # (n_cells, 512)

    # 对照组 CLS embeddings
    ctrl_z = all_z[ctrl_mask]  # (n_ctrl, 512)
    ctrl_z_mean = ctrl_z.mean(dim=0, keepdim=True)  # (1, 512)

    # 用 gene_embedding_layer 获取所有基因的 embedding
    # 基因名来自 adata.var
    gene_names = list(adata.var.index)
    gene_embs = model.get_gene_embedding(gene_names).detach()  # (n_genes, d_model)

    # 预测：ctrl CLS embedding → decode all genes
    # 使用 binary_decoder 的方式（与 shared_step 兼容）
    ctrl_z_device = ctrl_z_mean.to(device)  # (1, 512)
    gene_embs_device = gene_embs.to(device)  # (n_genes, d_model)

    pred_scores = predict_from_emb(model, ctrl_z_device, gene_embs_device, args.read_depth, device)
    # pred_scores: (n_genes,)

    # 扰动级 Pearson r
    pearson_list = []
    results_per_pert = []
    for pert in tqdm(pert_unique, desc="  Computing perturbation Pearson"):
        pert_mask = pert_labels == pert
        if pert_mask.sum() < 5:
            continue
        pert_mean = X_log[pert_mask].mean(axis=0)  # (n_genes,)
        log_fc = pert_mean - ctrl_mean  # (n_genes,)

        # 用 top 差异基因（基于 |log_fc| 大小）
        if args.top_genes > 0:
            top_idx = np.argsort(np.abs(log_fc))[-args.top_genes:]
        else:
            top_idx = np.arange(len(log_fc))

        pred_sub = pred_scores[top_idx]
        gt_sub = log_fc[top_idx]

        valid = ~(np.isnan(pred_sub) | np.isnan(gt_sub))
        if valid.sum() < 5:
            continue

        r, pval = pearsonr(pred_sub[valid], gt_sub[valid])
        if not np.isnan(r):
            pearson_list.append(float(r))
            results_per_pert.append({
                "perturbation": pert,
                "n_cells": int(pert_mask.sum()),
                "pearson_r": float(r),
                "pval": float(pval),
            })

    mean_r = float(np.mean(pearson_list)) if pearson_list else float("nan")
    return {
        "mode": "B_perturbation_level",
        "n_perturbations": len(pearson_list),
        "pearson_mean": mean_r,
        "pearson_median": float(np.median(pearson_list)) if pearson_list else float("nan"),
        "pearson_std": float(np.std(pearson_list)) if pearson_list else float("nan"),
        "per_perturbation": results_per_pert,
    }


def predict_from_emb(model, cell_emb, gene_embs, read_depth, device):
    """
    给定 cell CLS embedding 和所有基因 embedding，返回预测分数 (n_genes,)。
    分批计算以避免 OOM。
    """
    batch_size = 512  # 每次处理多少基因
    n_genes = gene_embs.size(0)
    cell_emb_512 = model.output_dim if hasattr(model, "output_dim") else cell_emb.size(1)

    all_scores = []
    for start in range(0, n_genes, batch_size):
        end = min(start + batch_size, n_genes)
        gene_batch = gene_embs[start:end]  # (batch, d_model)
        n_batch = gene_batch.size(0)

        # expand cell emb to (1, n_batch, 512)
        c = cell_emb.expand(1, -1)  # (1, 512)
        z = c.unsqueeze(1).repeat(1, n_batch, 1)   # (1, n_batch, 512)
        g = gene_batch.unsqueeze(0)                 # (1, n_batch, d_model)

        if model.z_dim_rd == 1:
            rd = torch.full((1, n_batch, 1), read_depth, device=device)
            combine = torch.cat([g, z, rd], dim=2)
        else:
            combine = torch.cat([g, z], dim=2)

        with torch.no_grad():
            scores = model.binary_decoder(combine).squeeze()  # (n_batch,)
            all_scores.append(scores.cpu().float().numpy())

    return np.concatenate(all_scores)  # (n_genes,)


# ============================================================================
# 绘图
# ============================================================================

def plot_pearson_distribution(pearson_values, output_path, title="Cell-level Pearson r"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(pearson_values, bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(pearson_values), color="#E74C3C", linestyle="--",
               linewidth=2, label=f"Mean={np.mean(pearson_values):.4f}")
    ax.axvline(np.median(pearson_values), color="#F39C12", linestyle="-.",
               linewidth=2, label=f"Median={np.median(pearson_values):.4f}")
    ax.set_xlabel("Pearson r", fontsize=12)
    ax.set_ylabel("# cells", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_model_comparison(results_dict, output_path):
    """results_dict: {model_name: {pearson_mean, pearson_std}}"""
    import matplotlib.pyplot as plt

    names = list(results_dict.keys())
    means = [results_dict[n]["pearson_mean"] for n in names]
    stds = [results_dict[n].get("pearson_std", 0) for n in names]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"][:len(names)]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(names, means, yerr=stds, color=colors[:len(names)],
                  capsize=6, alpha=0.85, edgecolor="white")
    ax.set_ylabel("Pearson r (mean ± std)", fontsize=12)
    ax.set_title("HepG2 Zero-Shot Prediction Pearson r", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{v:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# 主流程
# ============================================================================

def run_eval(checkpoint_path, cfg, args, label):
    model, device = load_state_model(checkpoint_path, cfg)
    results = {"label": label}

    # 模式 A：细胞级 Pearson
    print(f"\n  === 模式 A: 细胞级 Pearson r ===")
    r_a = eval_mode_a_cell_pearson(model, cfg, args.hepg2, args, device)
    results["mode_A"] = r_a
    print(f"  Mean Pearson r = {r_a['pearson_mean']:.4f} ± {r_a['pearson_std']:.4f}")
    print(f"  Median         = {r_a['pearson_median']:.4f}")
    print(f"  N cells        = {r_a['n_cells']}")

    # 模式 B：扰动级 Pearson（可选）
    if args.pert_col is not None:
        print(f"\n  === 模式 B: 扰动级 Pearson r ===")
        r_b = eval_mode_b_perturbation_pearson(model, cfg, args.hepg2, args, device)
        if r_b is not None:
            results["mode_B"] = r_b
            print(f"  Mean Pearson r = {r_b['pearson_mean']:.4f}")
            print(f"  N perturbations = {r_b['n_perturbations']}")

    return results


def main():
    args = parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.checkpoint).parent / "eval_pearson")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")
    print(f"HepG2 数据: {args.hepg2}")

    all_results = {}

    # ---- 主 checkpoint ----
    print(f"\n{'='*60}\n处理: STATE+MMD\n{'='*60}")
    res_mmd = run_eval(args.checkpoint, cfg, args, label="STATE+MMD")
    all_results["STATE+MMD"] = res_mmd

    # ---- 基线 checkpoint ----
    if args.baseline:
        print(f"\n{'='*60}\n处理: Baseline\n{'='*60}")
        res_base = run_eval(args.baseline, cfg, args, label="Baseline")
        all_results["Baseline"] = res_base

    # ---- 汇总打印 ----
    print("\n" + "=" * 70)
    print("📊 HepG2 零样本 Pearson r 汇总")
    print("=" * 70)
    for name, res in all_results.items():
        r = res["mode_A"]
        print(f"  {name:20s}  Mean={r['pearson_mean']:.4f} ± {r['pearson_std']:.4f}  "
              f"Median={r['pearson_median']:.4f}  N={r['n_cells']}")
    print("=" * 70)

    # ---- 绘图 ----
    if len(all_results) > 1:
        compare_data = {n: res["mode_A"] for n, res in all_results.items()}
        plot_model_comparison(compare_data, os.path.join(output_dir, "pearson_comparison.png"))

    # 分布直方图（仅主模型）
    # Note: 需要逐步重算才有 per-cell values; 简化：只报告汇总数
    # 若需要直方图可重新 run 并收集 pearson_per_cell

    # ---- 保存 JSON ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"pearson_results_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")
    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()

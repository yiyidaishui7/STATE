#!/usr/bin/env python3
"""
eval_state_domain_alignment.py — STATE+MMD 域对齐定量评估 + UMAP 可视化

功能：
  1. 加载 STATE+MMD checkpoint（支持对比两个 checkpoint）
  2. 提取 K562/RPE1/Jurkat/HepG2 的 CLS embeddings（512-dim）
  3. 计算域对齐指标：
     - Domain Classification Accuracy（越低越好，1/N_domains = 完美对齐）
     - Silhouette Score（越接近 0 越好）
     - Pairwise MMD（越低越好）
     - CORAL Distance（越低越好）
  4. 生成 UMAP 可视化（可对比有/无域对齐）

使用方法（在服务器 ~/state/src 目录下）：

    # 评估单个 checkpoint
    python ../scripts/eval_state_domain_alignment.py \\
        --checkpoint /path/to/mmd_aae_pretrain_final.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --output /path/to/eval_output

    # 对比有/无 MMD 的两个 checkpoint
    python ../scripts/eval_state_domain_alignment.py \\
        --checkpoint /path/to/mmd_ckpt.ckpt \\
        --baseline /path/to/baseline_ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --output /path/to/eval_output
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
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore")

# 确保能 import state 模块
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ============================================================================
# 配置
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"

DOMAIN_H5 = {
    "K562":   f"{BASE_DIR}/competition_support_set/k562.h5",
    "RPE1":   f"{BASE_DIR}/competition_support_set/rpe1.h5",
    "Jurkat": f"{BASE_DIR}/competition_support_set/jurkat.h5",
    "HepG2":  f"{BASE_DIR}/competition_support_set/hepg2.h5",
}

DOMAIN_COLORS = {
    "K562":   "#E74C3C",
    "RPE1":   "#3498DB",
    "Jurkat": "#2ECC71",
    "HepG2":  "#F39C12",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="STATE+MMD checkpoint 路径（.ckpt）")
    p.add_argument("--baseline", default=None, help="基线 checkpoint（无 MMD），用于对比")
    p.add_argument("--config", required=True, help="配置文件路径（mmd_aae_config.yaml）")
    p.add_argument("--output", default=None, help="输出目录（默认 checkpoint 同级 eval/）")
    p.add_argument("--max_samples", type=int, default=3000, help="每域最多取多少细胞")
    p.add_argument("--domains", nargs="+", default=["K562", "RPE1", "Jurkat"],
                   help="用于域对齐评估的域（HepG2 用于可视化但不参与对齐指标）")
    p.add_argument("--include_hepg2_umap", action="store_true",
                   help="是否把 HepG2 也加入 UMAP 可视化")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=2)
    return p.parse_args()


# ============================================================================
# 提取 CLS Embeddings
# ============================================================================

def read_h5_shape(h5_path):
    """从 h5/h5ad 文件中读取 (n_cells, n_genes)。"""
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
    """
    加载 STATE 模型。
    注意：绕过 Inference.encode()（当 dataset_correction=False 时 dataset_embedder 不存在）。
    直接使用 StateEmbeddingModel.load_from_checkpoint + pe_embedding 注入。
    """
    from state.emb.nn.model import StateEmbeddingModel
    from state.emb.train.trainer import get_embeddings

    print(f"  Loading checkpoint: {checkpoint_path}")
    model = StateEmbeddingModel.load_from_checkpoint(
        checkpoint_path,
        dropout=0.0,
        strict=False,
        cfg=cfg,
        map_location="cpu",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 注入 protein embeddings（先尝试从 checkpoint 读，再从 config 路径读）
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
    print(f"  Model loaded. device={device}, output_dim={cfg.model.output_dim}")
    return model, device


def extract_cls_embeddings(model, cfg, h5_path, domain_name, max_samples, batch_size, num_workers, device):
    """
    对单个域的 h5 文件提取 CLS embeddings。
    使用 H5adSentenceDataset + VCIDatasetSentenceCollator。
    """
    from state.emb.data import H5adSentenceDataset, VCIDatasetSentenceCollator

    n_cells, n_genes = read_h5_shape(h5_path)
    n_cells = min(n_cells, max_samples)
    print(f"    {domain_name}: ({n_cells}, {n_genes})")

    dataset = H5adSentenceDataset(
        cfg, test=True,
        datasets=[domain_name],
        shape_dict={domain_name: (n_cells, n_genes)},
    )
    dataset.dataset_path_map = {domain_name: h5_path}

    collator = VCIDatasetSentenceCollator(cfg, is_train=False)
    collator.cfg = cfg

    # Register unknown domains (e.g. "HepG2") by aliasing from the lowercase h5-stem
    # key (e.g. "hepg2") which exists in the competition ds_emb_mapping.  Also alias
    # valid_gene_mask so the correct per-domain gene filter is applied.
    if domain_name not in collator.dataset_to_protein_embeddings:
        lower = domain_name.lower()
        if lower in collator.dataset_to_protein_embeddings:
            collator.dataset_to_protein_embeddings[domain_name] = \
                collator.dataset_to_protein_embeddings[lower]
            collator.global_to_local[domain_name] = collator.global_to_local[lower]
            if (collator.valid_gene_mask is not None
                    and isinstance(collator.valid_gene_mask, dict)
                    and lower in collator.valid_gene_mask):
                collator.valid_gene_mask[domain_name] = collator.valid_gene_mask[lower]
        else:
            # hard fallback: borrow first registered domain's indices
            ref = next(iter(collator.dataset_to_protein_embeddings))
            collator.dataset_to_protein_embeddings[domain_name] = \
                collator.dataset_to_protein_embeddings[ref]
            collator.global_to_local[domain_name] = collator.global_to_local[ref]

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        persistent_workers=(num_workers > 0),
    )

    n_batches = (min(n_cells, max_samples) + batch_size - 1) // batch_size
    z_list = []
    total = 0
    with torch.no_grad():
        for batch in tqdm(loader, total=n_batches, desc=f"    {domain_name}", leave=False):
            _, _, _, emb, _ = model._compute_embedding_for_batch(batch)
            z_list.append(emb.detach().cpu().float().numpy())
            total += emb.size(0)
            if total >= max_samples:
                break

    z = np.concatenate(z_list)[:max_samples]
    return z


# ============================================================================
# 域对齐指标
# ============================================================================

def metric_domain_cls_acc(z_all, labels, n_domains):
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_all)
    clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300,
                        random_state=42, early_stopping=True)
    scores = cross_val_score(clf, z_scaled, labels, cv=5, scoring="accuracy")
    return {
        "domain_cls_acc": float(scores.mean()),
        "domain_cls_std": float(scores.std()),
        "random_chance": 1.0 / n_domains,
    }


def metric_silhouette(z_all, labels):
    from sklearn.metrics import silhouette_score
    n = len(z_all)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        z_sub, l_sub = z_all[idx], labels[idx]
    else:
        z_sub, l_sub = z_all, labels
    score = silhouette_score(z_sub, l_sub, metric="euclidean")
    return {"silhouette": float(score)}


def rbf_mmd_numpy(x, y, sigmas=(0.1, 1.0, 10.0)):
    xx = np.dot(x, x.T)
    yy = np.dot(y, y.T)
    xy = np.dot(x, y.T)
    rx = np.diag(xx)
    ry = np.diag(yy)
    dxx = np.maximum(rx[:, None] + rx[None, :] - 2 * xx, 0)
    dyy = np.maximum(ry[:, None] + ry[None, :] - 2 * yy, 0)
    dxy = np.maximum(rx[:, None] + ry[None, :] - 2 * xy, 0)
    mmd = 0.0
    for s in sigmas:
        g = 1.0 / (2 * s ** 2)
        mmd += np.exp(-g * dxx).mean() + np.exp(-g * dyy).mean() - 2 * np.exp(-g * dxy).mean()
    return mmd / len(sigmas)


def metric_mmd(z_all, labels, domain_names):
    results = {}
    vals = []
    unique = np.unique(labels)
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            zi = z_all[labels == unique[i]][:1000]
            zj = z_all[labels == unique[j]][:1000]
            if len(zi) < 2 or len(zj) < 2:
                continue
            v = rbf_mmd_numpy(zi, zj)
            name = f"{domain_names[unique[i]]}_vs_{domain_names[unique[j]]}"
            results[f"mmd_{name}"] = float(v)
            vals.append(v)
    results["mmd_mean"] = float(np.mean(vals)) if vals else 0.0
    return results


def metric_coral(z_all, labels, domain_names):
    results = {}
    vals = []
    unique = np.unique(labels)
    for i in range(len(unique)):
        for j in range(i + 1, len(unique)):
            zi = z_all[labels == unique[i]]
            zj = z_all[labels == unique[j]]
            if len(zi) < 2 or len(zj) < 2:
                continue
            ci = np.cov(zi, rowvar=False)
            cj = np.cov(zj, rowvar=False)
            diff = ci - cj
            d = float(np.sqrt((diff ** 2).sum()) / (4 * zi.shape[1] ** 2))
            name = f"{domain_names[unique[i]]}_vs_{domain_names[unique[j]]}"
            results[f"coral_{name}"] = d
            vals.append(d)
    results["coral_mean"] = float(np.mean(vals)) if vals else 0.0
    return results


def compute_all_metrics(z_all, labels, domain_names_list):
    """
    z_all: (N, D) numpy array
    labels: (N,) integer array (0, 1, 2, ...)
    domain_names_list: list of names indexed by label int
    """
    n_domains = len(np.unique(labels))
    domain_names_map = {i: n for i, n in enumerate(domain_names_list)}

    print("  [1/4] Domain Classification Accuracy...")
    r = metric_domain_cls_acc(z_all, labels, n_domains)
    print(f"        Acc = {r['domain_cls_acc']:.4f} ± {r['domain_cls_std']:.4f}  (random={r['random_chance']:.3f})")

    print("  [2/4] Silhouette Score...")
    r.update(metric_silhouette(z_all, labels))
    print(f"        Score = {r['silhouette']:.4f}  (0 = perfect mixing)")

    print("  [3/4] Pairwise MMD...")
    r.update(metric_mmd(z_all, labels, domain_names_map))
    print(f"        Mean MMD = {r['mmd_mean']:.6f}")

    print("  [4/4] CORAL Distance...")
    r.update(metric_coral(z_all, labels, domain_names_map))
    print(f"        Mean CORAL = {r['coral_mean']:.6f}")

    return r


# ============================================================================
# UMAP 可视化
# ============================================================================

def plot_umap(embeddings_dict, output_path, title="CLS Embedding UMAP"):
    """
    embeddings_dict: {"K562": array, "RPE1": array, ...}
    """
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed. Falling back to t-SNE...")
        return plot_tsne(embeddings_dict, output_path, title)

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_z = np.concatenate(list(embeddings_dict.values()), axis=0)
    all_names = []
    for name, z in embeddings_dict.items():
        all_names.extend([name] * len(z))
    all_names = np.array(all_names)

    print(f"  Running UMAP on {len(all_z)} cells...")
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    coords = reducer.fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in embeddings_dict:
        mask = all_names == name
        color = DOMAIN_COLORS.get(name, "#888888")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=name, s=8, alpha=0.6, linewidths=0)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=11)
    ax.set_ylabel("UMAP 2", fontsize=11)
    patches = [mpatches.Patch(color=DOMAIN_COLORS.get(n, "#888888"), label=n)
               for n in embeddings_dict]
    ax.legend(handles=patches, fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_tsne(embeddings_dict, output_path, title="CLS Embedding t-SNE"):
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_z = np.concatenate(list(embeddings_dict.values()), axis=0)
    all_names = []
    for name, z in embeddings_dict.items():
        all_names.extend([name] * len(z))
    all_names = np.array(all_names)

    print(f"  Running t-SNE on {len(all_z)} cells (perplexity=30)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    coords = tsne.fit_transform(all_z)

    fig, ax = plt.subplots(figsize=(9, 7))
    for name in embeddings_dict:
        mask = all_names == name
        color = DOMAIN_COLORS.get(name, "#888888")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, label=name, s=8, alpha=0.6, linewidths=0)

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    patches = [mpatches.Patch(color=DOMAIN_COLORS.get(n, "#888888"), label=n)
               for n in embeddings_dict]
    ax.legend(handles=patches, fontsize=10, markerscale=2)
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_comparison_umap(emb_before, emb_after, output_path):
    """Side-by-side comparison UMAP: baseline vs MMD-aligned."""
    try:
        import umap
    except ImportError:
        print("  umap-learn not installed, skipping comparison UMAP.")
        return

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    all_keys = list(emb_before.keys())
    all_z_b = np.concatenate(list(emb_before.values()), axis=0)
    all_z_a = np.concatenate(list(emb_after.values()), axis=0)
    all_names_b = np.array([n for n, z in emb_before.items() for _ in range(len(z))])
    all_names_a = np.array([n for n, z in emb_after.items() for _ in range(len(z))])

    # Fit UMAP on combined data for comparable coordinates
    combined = np.concatenate([all_z_b, all_z_a], axis=0)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.3)
    print("  Fitting UMAP on combined embeddings...")
    coords = reducer.fit_transform(combined)
    n_b = len(all_z_b)
    coords_b, coords_a = coords[:n_b], coords[n_b:]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    for ax, coords_set, names_set, title in [
        (ax1, coords_b, all_names_b, "Baseline (No Domain Alignment)"),
        (ax2, coords_a, all_names_a, "STATE + MMD Domain Alignment"),
    ]:
        for name in all_keys:
            mask = names_set == name
            color = DOMAIN_COLORS.get(name, "#888888")
            ax.scatter(coords_set[mask, 0], coords_set[mask, 1],
                       c=color, s=8, alpha=0.6, linewidths=0)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        patches = [mpatches.Patch(color=DOMAIN_COLORS.get(n, "#888888"), label=n)
                   for n in all_keys]
        ax.legend(handles=patches, fontsize=9, markerscale=2)
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


# ============================================================================
# 汇总表格打印
# ============================================================================

def print_comparison_table(results_dict):
    """results_dict: {'MMD': {...metrics}, 'Baseline': {...metrics}}"""
    headers = ["Model", "DomainClsAcc↓", "Random", "Silhouette↓", "MMD↓", "CORAL↓"]
    widths = [20, 15, 8, 14, 12, 12]

    print("\n" + "=" * 90)
    print("📊 域对齐指标对比")
    print("=" * 90)
    header_line = "".join(f"{h:<{w}}" for h, w in zip(headers, widths))
    print(header_line)
    print("-" * 90)

    for name, r in results_dict.items():
        row = [
            name,
            f"{r.get('domain_cls_acc', 0):.4f} ± {r.get('domain_cls_std', 0):.4f}",
            f"{r.get('random_chance', 0):.3f}",
            f"{r.get('silhouette', 0):.4f}",
            f"{r.get('mmd_mean', 0):.6f}",
            f"{r.get('coral_mean', 0):.6f}",
        ]
        print("".join(f"{v:<{w}}" for v, w in zip(row, widths)))

    print("-" * 90)
    print("\n解读: DomainClsAcc 越低 = 越好 (越难区分域); Silhouette/MMD/CORAL 越低 = 越混合")


# ============================================================================
# 主流程
# ============================================================================

def run_for_checkpoint(checkpoint_path, cfg, args, label):
    """对一个 checkpoint 提取 embedding + 计算指标。"""
    print(f"\n{'='*60}")
    print(f"处理: {label}")
    print(f"{'='*60}")

    model, device = load_state_model(checkpoint_path, cfg)

    # 提取 embedding
    emb_dict = {}
    domains_for_metrics = args.domains  # 默认 K562/RPE1/Jurkat

    all_domains = list(domains_for_metrics)
    if args.include_hepg2_umap and "HepG2" not in all_domains:
        all_domains = all_domains + ["HepG2"]

    for domain in all_domains:
        h5_path = DOMAIN_H5.get(domain)
        if h5_path is None or not os.path.exists(h5_path):
            print(f"  跳过 {domain}: 文件不存在 ({h5_path})")
            continue
        print(f"  编码 {domain}...")
        z = extract_cls_embeddings(
            model, cfg, h5_path, domain,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        emb_dict[domain] = z
        print(f"    → {len(z)} cells, shape={z.shape}")

    # 计算指标（仅用 alignment domains，不包含 HepG2）
    metric_domains = [d for d in domains_for_metrics if d in emb_dict]
    z_metric = np.concatenate([emb_dict[d] for d in metric_domains], axis=0)
    labels_metric = np.array([i for i, d in enumerate(metric_domains) for _ in range(len(emb_dict[d]))])

    print(f"\n  计算域对齐指标 ({'/'.join(metric_domains)}, N={len(z_metric)})...")
    metrics = compute_all_metrics(z_metric, labels_metric, metric_domains)
    metrics["label"] = label
    metrics["n_cells"] = len(z_metric)
    metrics["domains"] = metric_domains

    return emb_dict, metrics


def main():
    args = parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    # 输出目录
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(args.checkpoint).parent / "eval")
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n输出目录: {output_dir}")

    results_all = {}

    # ---- 主 checkpoint（STATE+MMD）----
    emb_mmd, metrics_mmd = run_for_checkpoint(args.checkpoint, cfg, args, label="STATE+MMD")
    results_all["STATE+MMD"] = metrics_mmd

    # ---- 基线 checkpoint（可选）----
    emb_baseline = None
    if args.baseline:
        emb_baseline, metrics_baseline = run_for_checkpoint(args.baseline, cfg, args, label="Baseline")
        results_all["Baseline"] = metrics_baseline

    # ---- 打印汇总表格 ----
    print_comparison_table(results_all)

    # ---- UMAP 可视化 ----
    print("\n生成 UMAP/t-SNE 可视化...")

    # 单模型 UMAP
    umap_path = os.path.join(output_dir, "umap_mmd_aligned.png")
    plot_umap(emb_mmd, umap_path, title="STATE+MMD CLS Embeddings")

    if emb_baseline is not None:
        # 基线 UMAP
        umap_base_path = os.path.join(output_dir, "umap_baseline.png")
        plot_umap(emb_baseline, umap_base_path, title="Baseline STATE CLS Embeddings")

        # 对比图
        compare_path = os.path.join(output_dir, "umap_comparison.png")
        plot_comparison_umap(emb_baseline, emb_mmd, compare_path)

    # ---- 保存结果 ----
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(output_dir, f"domain_alignment_metrics_{ts}.json")
    with open(json_path, "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\n结果已保存: {json_path}")

    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()

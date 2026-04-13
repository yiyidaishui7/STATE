#!/usr/bin/env python3
"""
compute_ctrl_stats.py — 离线预计算 ctrl 统计量（路线一：logFC MSE loss）

输出一个 .pt 文件，包含：
  ctrl_mean_by_pe_idx (global_size,) : 按 pe_embedding 全局索引存储的对照组均值
  ctrl_cls_ref        (output_dim,)  : 对照细胞 CLS embedding 的均值

运行方式（服务器 ~/state/src 目录）：
    python ../scripts/compute_ctrl_stats.py \\
        --checkpoint /path/to/ckpt.ckpt \\
        --config ../configs/mmd_aae_config.yaml \\
        --h5 /path/to/k562.h5 \\
        --output /path/to/ctrl_stats_k562.pt
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="预训练 checkpoint (.ckpt)")
    p.add_argument("--config",     required=True, help="config yaml 路径")
    p.add_argument("--h5",         required=True, help="训练数据 h5 文件（如 k562.h5）")
    p.add_argument("--pert_col",   default="gene",           help="obs 列名（扰动标签）")
    p.add_argument("--ctrl_label", default="non-targeting",  help="对照组标签")
    p.add_argument("--max_cells",  type=int, default=1000,   help="最多取多少个对照细胞")
    p.add_argument("--output",     default="ctrl_stats.pt",  help="输出文件路径")
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


# ============================================================================
# Step 1：从 h5 文件读取对照细胞的基因表达，计算真实均值
# ============================================================================

def compute_ctrl_mean_true(h5_path, pert_col, ctrl_label, max_cells):
    """
    返回：
        ctrl_mean_true  (n_local_genes,)  — 对照细胞的均值（log1p 空间，按 h5 列顺序）
        ctrl_indices    list[int]          — 用到的对照细胞行索引
        n_genes         int                — h5 文件的基因总数
    """
    import anndata as ad
    from scipy.sparse import issparse

    print(f"  读取数据: {h5_path}")
    try:
        adata = ad.read_h5ad(h5_path)
    except Exception:
        # fallback: 旧版 h5 格式
        import h5py
        from scipy.sparse import csr_matrix
        with h5py.File(h5_path, "r") as f:
            if "indices" in f["X"]:
                data    = f["X"]["data"][:]
                indices = f["X"]["indices"][:]
                indptr  = f["X"]["indptr"][:]
                attrs   = dict(f["X"].attrs)
                n_c = indptr.shape[0] - 1
                n_g = int(attrs["shape"][1])
                X = csr_matrix((data, indices, indptr), shape=(n_c, n_g)).toarray()
            else:
                X = f["X"][:]
        adata = type("A", (), {"X": X, "obs": {}})()  # 伪 AnnData，无 obs

    if issparse(adata.X):
        adata.X = adata.X.toarray()
    adata.X = adata.X.astype(np.float32)
    if adata.X.max() > 100:
        adata.X = np.log1p(adata.X)

    n_genes = adata.X.shape[1]

    # 找对照细胞
    if hasattr(adata, "obs") and isinstance(adata.obs, dict) is False and pert_col in adata.obs.columns:
        labels = adata.obs[pert_col].astype(str).values
        ctrl_idx = np.where(labels == ctrl_label)[0]
    else:
        print("  ⚠ 无 obs 元数据，取前 max_cells 个细胞作为近似对照")
        ctrl_idx = np.arange(min(max_cells, adata.X.shape[0]))

    if len(ctrl_idx) > max_cells:
        ctrl_idx = ctrl_idx[:max_cells]

    ctrl_mean_true = adata.X[ctrl_idx].mean(axis=0)  # (n_local_genes,)
    print(f"  对照细胞数: {len(ctrl_idx)}，基因数: {n_genes}")
    return torch.tensor(ctrl_mean_true, dtype=torch.float32), list(ctrl_idx), n_genes


# ============================================================================
# Step 2：构建 ctrl_mean_by_pe_idx（按 pe_embedding 全局索引）
# ============================================================================

def build_ctrl_mean_by_pe_idx(ctrl_mean_true, cfg):
    """
    把 ctrl_mean_true（按本地基因列索引）映射到 global pe_embedding 索引空间。

    批训练时 batch[1] 包含 global pe_embedding 索引，
    用这个映射可以直接做 ctrl_mean_by_pe_idx[batch[1]] 取值。
    """
    from state.emb import utils as eu

    global_size = eu.get_embedding_cfg(cfg).num  # 19790

    # 加载 ds_emb_mapping：dataset → 每个基因的 global pe_embedding 索引
    ds_emb_mapping_path = eu.get_embedding_cfg(cfg).ds_emb_mapping
    ds_emb_mapping = torch.load(ds_emb_mapping_path, map_location="cpu", weights_only=False)

    # 找 K562（大小写不敏感）
    key = None
    for k in ds_emb_mapping:
        if k.lower() == "k562":
            key = k
            break
    if key is None:
        # 找第一个非 default 的键
        key = next((k for k in ds_emb_mapping if k != "default"), "default")
        print(f"  ⚠ 未找到 'k562' key，使用 '{key}'")

    ds_emb_idxs = torch.tensor(ds_emb_mapping[key], dtype=torch.long)  # (n_genes_raw,)

    # 加载 valid_genes_mask（如果有）
    valid_masks_path = eu.get_embedding_cfg(cfg).valid_genes_masks
    if valid_masks_path is not None:
        valid_masks = torch.load(valid_masks_path, map_location="cpu", weights_only=False)
        mask_key = next((k for k in valid_masks if k.lower() == "k562"), None)
        if mask_key and ds_emb_idxs.shape[0] == valid_masks[mask_key].shape[0]:
            valid_mask = valid_masks[mask_key]
            ds_emb_idxs = ds_emb_idxs[valid_mask]
            # ctrl_mean_true 也只保留有效基因
            ctrl_mean_true = ctrl_mean_true[valid_mask]
            print(f"  应用 valid_gene_mask: {valid_mask.sum().item()} / {valid_mask.shape[0]} 基因保留")

    # 构建 global 空间的均值向量
    ctrl_mean_by_pe_idx = torch.zeros(global_size, dtype=torch.float32)
    for local_j, pe_idx in enumerate(ds_emb_idxs.tolist()):
        if 0 <= pe_idx < global_size and local_j < len(ctrl_mean_true):
            ctrl_mean_by_pe_idx[pe_idx] = ctrl_mean_true[local_j]

    n_filled = (ctrl_mean_by_pe_idx != 0).sum().item()
    print(f"  ctrl_mean_by_pe_idx: {global_size} 维，{n_filled} 个基因有值")
    return ctrl_mean_by_pe_idx


# ============================================================================
# Step 3：编码对照细胞 → 计算平均 CLS embedding
# ============================================================================

def compute_ctrl_cls_ref(checkpoint_path, cfg, h5_path, ctrl_indices, batch_size):
    """
    加载模型，对 ctrl_indices 指定的对照细胞做前向推理，
    返回平均 CLS embedding (output_dim,)。
    """
    from torch import nn
    from state.emb.nn.model import StateEmbeddingModel

    print(f"  加载 checkpoint: {checkpoint_path}")
    model = StateEmbeddingModel.load_from_checkpoint(
        checkpoint_path, dropout=0.0, strict=False, cfg=cfg, map_location="cpu"
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    # 挂载 protein embeddings
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "protein_embeds_dict" in ckpt:
        pe_dict = ckpt["protein_embeds_dict"]
        all_pe  = torch.vstack(list(pe_dict.values())).to(device)
    else:
        from state.emb.train.trainer import get_embeddings
        all_pe = get_embeddings(cfg).to(device)

    model.pe_embedding = nn.Embedding.from_pretrained(all_pe, freeze=True).to(device)

    # 构建 DataLoader
    import h5py
    from state.emb.data import H5adSentenceDataset, VCIDatasetSentenceCollator

    domain_name = Path(h5_path).stem
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f["X"].attrs)
        if "shape" in attrs:
            n_cells_total = int(attrs["shape"][0])
            n_genes       = int(attrs["shape"][1])
        elif hasattr(f["X"], "shape") and len(f["X"].shape) == 2:
            n_cells_total, n_genes = f["X"].shape
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

    subset   = Subset(dataset, ctrl_indices)
    collator = VCIDatasetSentenceCollator(cfg, is_train=False)
    collator.cfg = cfg

    loader = DataLoader(
        subset, batch_size=batch_size, shuffle=False,
        collate_fn=collator, num_workers=0,
    )

    all_cls = []
    print(f"  编码 {len(ctrl_indices)} 个对照细胞...")
    with torch.no_grad():
        for batch in loader:
            _, _, _, embs, _ = model._compute_embedding_for_batch(batch)
            all_cls.append(embs.detach().cpu())

    ctrl_cls_ref = torch.cat(all_cls, dim=0).mean(dim=0)  # (output_dim,)
    print(f"  ctrl_cls_ref shape: {ctrl_cls_ref.shape}")
    return ctrl_cls_ref


# ============================================================================
# 主流程
# ============================================================================

def main():
    args = parse_args()

    from omegaconf import OmegaConf
    cfg = OmegaConf.load(args.config)

    print("\n[1/3] 计算真实对照均值 ctrl_mean_true ...")
    ctrl_mean_true, ctrl_indices, n_genes = compute_ctrl_mean_true(
        args.h5, args.pert_col, args.ctrl_label, args.max_cells
    )

    print("\n[2/3] 构建 ctrl_mean_by_pe_idx ...")
    ctrl_mean_by_pe_idx = build_ctrl_mean_by_pe_idx(ctrl_mean_true, cfg)

    print("\n[3/3] 计算 ctrl_cls_ref ...")
    ctrl_cls_ref = compute_ctrl_cls_ref(
        args.checkpoint, cfg, args.h5, ctrl_indices, args.batch_size
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "ctrl_mean_by_pe_idx": ctrl_mean_by_pe_idx,  # (global_size,)
        "ctrl_cls_ref":        ctrl_cls_ref,          # (output_dim,)
        "meta": {
            "h5_path":     args.h5,
            "ctrl_label":  args.ctrl_label,
            "n_ctrl_cells": len(ctrl_indices),
            "n_local_genes": n_genes,
        }
    }, output_path)

    print(f"\n已保存: {output_path}")
    print("  ctrl_mean_by_pe_idx:", ctrl_mean_by_pe_idx.shape)
    print("  ctrl_cls_ref:       ", ctrl_cls_ref.shape)
    print("  对照细胞数:          ", len(ctrl_indices))


if __name__ == "__main__":
    main()

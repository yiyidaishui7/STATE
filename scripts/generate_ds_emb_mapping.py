#!/usr/bin/env python3
"""
Generate ds_emb_mapping and valid_genes_masks for competition data.

Reads gene names from each h5 file, matches against gene names in the
ESM2 embeddings dict, and produces mapping tensors compatible with
the VCIDatasetSentenceCollator.

Usage:
    python scripts/generate_ds_emb_mapping.py
"""

import os
import h5py
import torch

BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
EMB_FILE = f"{BASE_DIR}/competition_support_set/ESM2_pert_features.pt"
OUT_DIR = f"{BASE_DIR}/competition_support_set"

H5_FILES = {
    "K562":   f"{BASE_DIR}/competition_support_set/k562.h5",
    "RPE1":   f"{BASE_DIR}/competition_support_set/rpe1.h5",
    "Jurkat": f"{BASE_DIR}/competition_support_set/jurkat.h5",
}


def extract_gene_names(h5_path):
    """Extract gene names from h5 file, trying multiple var/ fields."""
    with h5py.File(h5_path, "r") as f:
        if "var" not in f:
            raise ValueError(f"No var/ group in {h5_path}")
        
        var = f["var"]
        # Try common gene name fields
        for field in ("_index", "gene_name", "gene_symbols", "feature_name", "gene_id", "symbol"):
            if field not in var:
                continue
            grp = var[field]
            # Handle categorical encoding
            if "categories" in grp and "codes" in grp:
                raw_cats = grp["categories"][:]
                codes = grp["codes"][:]
                cats = [c.decode("utf-8").upper() if isinstance(c, (bytes, bytearray)) else str(c).upper()
                        for c in raw_cats]
                genes = [cats[int(code)] for code in codes]
            elif "categories" in grp:
                raw = grp["categories"][:]
                genes = [item.decode("utf-8").upper() if isinstance(item, (bytes, bytearray)) else str(item).upper()
                         for item in raw]
            else:
                raw = grp[:]
                genes = [item.decode("utf-8").upper() if isinstance(item, (bytes, bytearray)) else str(item).upper()
                         for item in raw]
            return genes, field
    
    raise ValueError(f"Could not find gene names in {h5_path}")


def main():
    print("=" * 60)
    print("Generating ds_emb_mapping for competition data")
    print("=" * 60)
    
    # Load ESM2 embeddings to get gene name → index mapping
    print(f"\nLoading ESM2 embeddings from: {EMB_FILE}")
    all_embs = torch.load(EMB_FILE, weights_only=False)
    
    if isinstance(all_embs, dict):
        # Dict mapping gene_name -> embedding tensor
        gene_to_idx = {str(k).upper(): i for i, k in enumerate(all_embs.keys())}
        num_embeddings = len(gene_to_idx)
        emb_size = next(iter(all_embs.values())).shape[0]
        print(f"  Embeddings: {num_embeddings} genes, dim={emb_size}")
    else:
        # If it's a stacked tensor, we can't do name matching
        print(f"  WARNING: Embeddings is a tensor of shape {all_embs.shape}, not a dict!")
        print("  Cannot match gene names. Using index-based mapping.")
        num_embeddings = all_embs.shape[0]
        gene_to_idx = None
    
    ds_map = {}
    masks = {}
    
    for name, path in H5_FILES.items():
        print(f"\n--- {name} ---")
        print(f"  Path: {path}")
        
        if not os.path.exists(path):
            print(f"  SKIP: file not found")
            continue
        
        if gene_to_idx is not None:
            genes, field = extract_gene_names(path)
            print(f"  Gene field: {field}")
            print(f"  Genes in data: {len(genes)}")
            
            # Build mapping: for each gene in h5, find its index in embeddings
            mapping = []
            mask = []
            for g in genes:
                idx = gene_to_idx.get(g, -1)
                mapping.append(idx)
                mask.append(idx != -1)
            
            mapping_tensor = torch.tensor(mapping, dtype=torch.long)
            mask_tensor = torch.tensor(mask, dtype=torch.bool)
            
            matched = mask_tensor.sum().item()
            print(f"  Matched: {matched}/{len(genes)} ({100*matched/len(genes):.1f}%)")
        else:
            # Fallback: index-based identity mapping
            with h5py.File(path, "r") as f:
                attrs = dict(f["X"].attrs)
                if "shape" in attrs:
                    n_genes = int(attrs["shape"][1])
                else:
                    n_genes = f["X"].shape[1]
            
            mapping_tensor = torch.arange(n_genes, dtype=torch.long)
            mask_tensor = torch.ones(n_genes, dtype=torch.bool)
            print(f"  Using identity mapping for {n_genes} genes")
        
        ds_map[name] = mapping_tensor
        masks[name] = mask_tensor
    
    # Save
    mapping_file = os.path.join(OUT_DIR, "ds_emb_mapping_competition.torch")
    masks_file = os.path.join(OUT_DIR, "valid_genes_masks_competition.torch")
    
    torch.save(ds_map, mapping_file)
    torch.save(masks, masks_file)
    
    print(f"\n{'='*60}")
    print(f"Saved mapping: {mapping_file}")
    print(f"Saved masks:   {masks_file}")
    print(f"\nUpdate config with:")
    print(f"  ds_emb_mapping: {mapping_file}")
    print(f"  valid_genes_masks: {masks_file}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

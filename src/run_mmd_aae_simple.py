#!/usr/bin/env python
"""
run_mmd_aae_simple.py - MMD-AAE 简化测试脚本
直接使用 h5py 加载数据，跳过复杂的 Collator 依赖

使用方法:
    cd ~/state/src
    python run_mmd_aae_simple.py
"""
import os
import sys
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

print("=" * 60)
print("MMD-AAE 简化测试脚本")
print("=" * 60)


# ============================================================================
# 简化版 Dataset - 直接从 h5 文件读取
# ============================================================================

class SimpleH5Dataset(Dataset):
    """简化版 Dataset，直接从 h5 文件读取密集矩阵"""
    
    def __init__(self, h5_path, domain_name="unknown"):
        self.h5_path = h5_path
        self.domain_name = domain_name
        
        # 打开文件获取形状信息
        with h5py.File(h5_path, 'r') as f:
            self.shape = f['X'].shape
            self.num_cells = self.shape[0]
            self.num_genes = self.shape[1]
        
        print(f"  [{domain_name}] 加载: {self.num_cells} cells × {self.num_genes} genes")
        
        # 保持文件句柄打开以加速读取
        self._file = None
    
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file
    
    def __len__(self):
        return self.num_cells
    
    def __getitem__(self, idx):
        f = self._get_file()
        # 读取单个细胞的表达数据
        counts = torch.tensor(f['X'][idx], dtype=torch.float32)
        return {
            'counts': counts,
            'idx': idx,
            'domain': self.domain_name
        }
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def simple_collate_fn(batch):
    """简单的 collate 函数"""
    counts = torch.stack([item['counts'] for item in batch])
    idxs = torch.tensor([item['idx'] for item in batch])
    domains = [item['domain'] for item in batch]
    return {
        'counts': counts,
        'idxs': idxs,
        'domains': domains
    }


class ParallelZipLoader:
    """并行多域 DataLoader"""
    
    def __init__(self, loaders, domain_names):
        self.loaders = loaders
        self.domain_names = domain_names
    
    def __iter__(self):
        return zip(*self.loaders)
    
    def __len__(self):
        return min(len(l) for l in self.loaders)
    
    @property
    def batch_size(self):
        return sum(l.batch_size for l in self.loaders)


def main():
    # 数据路径配置
    BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
    DOMAIN_CONFIGS = [
        {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
        {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
        {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
    ]
    
    batch_size = 32  # 每个域的 batch size
    
    # ========================================
    # Step 1: 创建三路 DataLoader
    # ========================================
    print("\n[Step 1] 创建三路并行 DataLoader...")
    
    loaders = []
    datasets = []
    domain_names = []
    
    for domain in DOMAIN_CONFIGS:
        name = domain["name"]
        path = domain["path"]
        
        if not os.path.exists(path):
            print(f"  ✗ 文件不存在: {path}")
            continue
        
        dataset = SimpleH5Dataset(path, name)
        datasets.append(dataset)
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=simple_collate_fn,
            num_workers=0,  # 简化起见，单进程
            drop_last=True
        )
        loaders.append(loader)
        domain_names.append(name)
    
    if len(loaders) < 3:
        print(f"\n⚠️ 只成功创建了 {len(loaders)} 个 DataLoader")
    
    parallel_loader = ParallelZipLoader(loaders, domain_names)
    
    print(f"\n✓ 并行加载器创建完成")
    print(f"  - 域: {domain_names}")
    print(f"  - 每域 batch size: {batch_size}")
    print(f"  - 总 batch size: {parallel_loader.batch_size}")
    print(f"  - 每轮迭代数: {len(parallel_loader)}")
    
    # ========================================
    # Step 2: 验证数据加载
    # ========================================
    print("\n[Step 2] 验证数据加载...")
    
    batch = next(iter(parallel_loader))
    
    print(f"\nBatch 信息:")
    print(f"  - 类型: {type(batch)}")
    print(f"  - 长度: {len(batch)} (应该等于域数量)")
    
    for i, (domain_batch, domain_name) in enumerate(zip(batch, domain_names)):
        counts = domain_batch['counts']
        print(f"\n  [{domain_name}]:")
        print(f"    counts shape: {counts.shape}")
        print(f"    非零比例: {(counts > 0).float().mean():.4f}")
        print(f"    平均值: {counts.mean():.4f}")
        print(f"    最大值: {counts.max():.4f}")
    
    # ========================================
    # Step 3: 模拟 MMD-AAE 前向传播
    # ========================================
    print("\n[Step 3] 模拟 MMD-AAE 架构...")
    
    # 合并三个域的 batch
    all_counts = torch.cat([b['counts'] for b in batch], dim=0)
    domain_labels = []
    for i, b in enumerate(batch):
        domain_labels.extend([i] * len(b['counts']))
    domain_labels = torch.tensor(domain_labels)
    
    print(f"\n合并后的数据:")
    print(f"  - 总 batch shape: {all_counts.shape}")
    print(f"  - 域标签分布: {torch.bincount(domain_labels).tolist()}")
    
    # 简单的 Encoder 测试
    print("\n模拟 Encoder 输出...")
    hidden_dim = 512
    encoder = torch.nn.Sequential(
        torch.nn.Linear(all_counts.shape[1], hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, hidden_dim)
    )
    
    with torch.no_grad():
        z = encoder(all_counts)
        print(f"  - Latent space shape: {z.shape}")
        print(f"  - Latent mean: {z.mean():.4f}")
        print(f"  - Latent std: {z.std():.4f}")
    
    # ========================================
    # 完成
    # ========================================
    print("\n" + "=" * 60)
    print("✅ 测试成功！三路并行数据加载工作正常。")
    print("=" * 60)
    
    print("\n下一步:")
    print("  1. 生成 gene_embidx_mapping.torch 文件")
    print("  2. 或修改 loader.py 来动态创建映射")
    print("  3. 然后运行完整的 MMD-AAE 训练")
    
    # 清理
    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()

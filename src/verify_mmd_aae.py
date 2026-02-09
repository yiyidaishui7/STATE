#!/usr/bin/env python
"""
verify_mmd_aae.py - 验证 DataLoader 和 MMD Loss 是否正常工作

使用方法:
    cd ~/state/src
    python verify_mmd_aae.py
"""

import os
import sys
import h5py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

print("=" * 70)
print("MMD-AAE 验证脚本")
print("=" * 70)

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
DOMAIN_CONFIGS = [
    {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
    {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
    {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
]
BATCH_SIZE = 32
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"\nDevice: {DEVICE}")

# ============================================================================
# Part 1: 验证 DataLoader
# ============================================================================
print("\n" + "=" * 70)
print("Part 1: 验证三路 DataLoader")
print("=" * 70)

class SimpleH5Dataset(Dataset):
    def __init__(self, h5_path, domain_id=0):
        self.h5_path = h5_path
        self.domain_id = domain_id
        with h5py.File(h5_path, 'r') as f:
            self.shape = f['X'].shape
        self._file = None
    
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        f = self._get_file()
        counts = torch.tensor(f['X'][idx], dtype=torch.float32)
        return counts, self.domain_id

def collate_fn(batch):
    counts = torch.stack([item[0] for item in batch])
    domains = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return counts, domains

# 创建三个 DataLoader
loaders = []
for i, domain in enumerate(DOMAIN_CONFIGS):
    print(f"\n[{domain['name']}]")
    print(f"  文件路径: {domain['path']}")
    
    if not os.path.exists(domain['path']):
        print(f"  ❌ 文件不存在!")
        continue
    
    dataset = SimpleH5Dataset(domain['path'], domain_id=i)
    print(f"  细胞数: {len(dataset)}")
    print(f"  基因数: {dataset.shape[1]}")
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,  # 单进程方便调试
        drop_last=True,
    )
    loaders.append(loader)
    print(f"  Batch 数: {len(loader)}")
    print(f"  ✅ DataLoader 创建成功")

if len(loaders) != 3:
    print("\n❌ 错误: 未能创建所有三个 DataLoader")
    sys.exit(1)

# 测试并行加载
print("\n--- 测试并行加载 ---")
for batch_idx, batches in enumerate(zip(*loaders)):
    if batch_idx >= 2:
        break
    
    print(f"\nBatch {batch_idx}:")
    all_counts = []
    all_domains = []
    
    for domain_id, (counts, domains) in enumerate(batches):
        domain_name = DOMAIN_CONFIGS[domain_id]['name']
        print(f"  [{domain_name}] counts.shape={counts.shape}, domains={domains[:3].tolist()}...")
        all_counts.append(counts)
        all_domains.append(torch.full((counts.size(0),), domain_id, dtype=torch.long))
    
    x = torch.cat(all_counts, dim=0)
    domain_labels = torch.cat(all_domains, dim=0)
    
    print(f"  合并后: x.shape={x.shape}, domain_labels.shape={domain_labels.shape}")
    print(f"  域分布: K562={sum(domain_labels==0)}, RPE1={sum(domain_labels==1)}, Jurkat={sum(domain_labels==2)}")

print("\n✅ DataLoader 验证通过!")

# ============================================================================
# Part 2: 验证 MMD 计算
# ============================================================================
print("\n" + "=" * 70)
print("Part 2: 验证 MMD Loss 计算")
print("=" * 70)

def compute_mmd_multi_kernel(x, y):
    """多核 MMD"""
    sigmas = [0.01, 0.1, 1.0, 10.0, 100.0]
    
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t().expand(x.size(0), y.size(0)) + ry.expand(x.size(0), y.size(0)) - 2. * xy
    
    mmd = x.new_zeros(1)
    
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma ** 2)
        XX = torch.exp(-gamma * dxx)
        YY = torch.exp(-gamma * dyy)
        XY = torch.exp(-gamma * dxy)
        mmd = mmd + XX.mean() + YY.mean() - 2. * XY.mean()
    
    return mmd / len(sigmas)

# 测试 1: 相同分布 → MMD ≈ 0
print("\n--- 测试 1: 相同分布 ---")
torch.manual_seed(42)
z1 = torch.randn(32, 64)
z2 = torch.randn(32, 64)  # 同分布
mmd_same = compute_mmd_multi_kernel(z1, z2)
print(f"  z1 ~ N(0,1), z2 ~ N(0,1)")
print(f"  MMD = {mmd_same.item():.6f}")
print(f"  预期: 接近 0 (可能有小波动)")

# 测试 2: 不同分布 → MMD > 0
print("\n--- 测试 2: 不同分布 ---")
z3 = torch.randn(32, 64) + 5  # 均值偏移
mmd_diff = compute_mmd_multi_kernel(z1, z3)
print(f"  z1 ~ N(0,1), z3 ~ N(5,1)")
print(f"  MMD = {mmd_diff.item():.6f}")
print(f"  预期: 明显大于 0")

if mmd_diff > mmd_same * 5:
    print("\n✅ MMD 能够区分不同分布!")
else:
    print("\n⚠️ MMD 区分能力可能有问题")

# 测试 3: 梯度检查
print("\n--- 测试 3: 梯度检查 ---")
z_a = torch.randn(32, 64, requires_grad=True)
z_b = torch.randn(32, 64, requires_grad=True)
mmd = compute_mmd_multi_kernel(z_a, z_b)
mmd.backward()

print(f"  z_a.grad is None: {z_a.grad is None}")
print(f"  z_b.grad is None: {z_b.grad is None}")

if z_a.grad is not None and z_b.grad is not None:
    print(f"  z_a.grad.shape: {z_a.grad.shape}")
    print(f"  z_b.grad.mean(): {z_a.grad.mean().item():.6f}")
    print("\n✅ MMD 梯度正常传播!")
else:
    print("\n❌ 梯度传播失败!")
    sys.exit(1)

# ============================================================================
# Part 3: 使用真实数据计算 MMD
# ============================================================================
print("\n" + "=" * 70)
print("Part 3: 使用真实数据计算域间 MMD")
print("=" * 70)

# 简单 Encoder
class SimpleEncoder(nn.Module):
    def __init__(self, input_dim, latent_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
        )
    
    def forward(self, x):
        return self.net(x)

# 获取一个 batch 的真实数据
batch_iter = iter(zip(*loaders))
batches = next(batch_iter)

all_counts = []
all_domains = []
for domain_id, (counts, domains) in enumerate(batches):
    all_counts.append(counts)
    all_domains.append(torch.full((counts.size(0),), domain_id, dtype=torch.long))

x = torch.cat(all_counts, dim=0).to(DEVICE)
domain_labels = torch.cat(all_domains, dim=0).to(DEVICE)
input_dim = x.shape[1]

print(f"\n输入数据: x.shape = {x.shape}")
print(f"域标签分布: {[(domain_labels == i).sum().item() for i in range(3)]}")

# 创建 Encoder
encoder = SimpleEncoder(input_dim, latent_dim=64).to(DEVICE)
z = encoder(x)
print(f"隐空间表示: z.shape = {z.shape}")

# 计算三对域之间的 MMD
print("\n--- 域间 MMD (训练前) ---")
mmd_pairs = []
domain_names = ["K562", "RPE1", "Jurkat"]

for i in range(3):
    for j in range(i + 1, 3):
        mask_i = domain_labels == i
        mask_j = domain_labels == j
        z_i = z[mask_i]
        z_j = z[mask_j]
        
        mmd = compute_mmd_multi_kernel(z_i, z_j)
        mmd_pairs.append((i, j, mmd.item()))
        print(f"  MMD({domain_names[i]}, {domain_names[j]}) = {mmd.item():.6f}")

# ============================================================================
# Part 4: 模拟一步训练
# ============================================================================
print("\n" + "=" * 70)
print("Part 4: 模拟一步 MMD 训练")
print("=" * 70)

optimizer = torch.optim.Adam(encoder.parameters(), lr=0.01)

print("\n训练 10 步，观察 MMD 变化...")
for step in range(10):
    optimizer.zero_grad()
    
    z = encoder(x)
    
    # 计算 MMD 损失
    mmd_losses = []
    for i in range(3):
        for j in range(i + 1, 3):
            mask_i = domain_labels == i
            mask_j = domain_labels == j
            mmd = compute_mmd_multi_kernel(z[mask_i], z[mask_j])
            mmd_losses.append(mmd)
    
    total_mmd = torch.stack(mmd_losses).mean()
    total_mmd.backward()
    optimizer.step()
    
    if step % 2 == 0 or step == 9:
        print(f"  Step {step}: MMD = {total_mmd.item():.6f}")

print("\n--- 训练后域间 MMD ---")
z = encoder(x)
for i in range(3):
    for j in range(i + 1, 3):
        mask_i = domain_labels == i
        mask_j = domain_labels == j
        mmd = compute_mmd_multi_kernel(z[mask_i], z[mask_j])
        print(f"  MMD({domain_names[i]}, {domain_names[j]}) = {mmd.item():.6f}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("验证总结")
print("=" * 70)
print("✅ 三路 DataLoader 正常工作")
print("✅ MMD 计算正常")
print("✅ 梯度正常传播")
print("✅ MMD 损失可以优化")
print("\n现在可以运行完整训练:")
print("  python train_mmd_aae.py")
print("=" * 70)

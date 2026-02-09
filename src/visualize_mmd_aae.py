#!/usr/bin/env python
"""
visualize_mmd_aae.py - 可视化 MMD-AAE 训练结果

使用方法:
    cd ~/state/src
    python visualize_mmd_aae.py

输出:
    - 隐空间 t-SNE 可视化
    - 训练损失曲线
"""

import os
import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
CHECKPOINT_PATH = f"{BASE_DIR}/checkpoints/mmd_aae/best_model.pt"
OUTPUT_DIR = f"{BASE_DIR}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_CONFIGS = [
    {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5", "color": "#E74C3C"},
    {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5", "color": "#3498DB"},
    {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5", "color": "#2ECC71"},
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SAMPLES_PER_DOMAIN = 500  # 每个域采样多少个细胞用于可视化

print("=" * 60)
print("MMD-AAE 可视化")
print("=" * 60)

# ============================================================================
# 模型定义 (与训练脚本一致)
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=512, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
    
    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=1024, output_dim=18080, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, z):
        return self.net(z)

class DomainDiscriminator(nn.Module):
    def __init__(self, latent_dim, hidden_dim=256, num_domains=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_domains),
        )
    
    def forward(self, z):
        return self.net(z)

class MMD_AAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=512, num_domains=3, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout)
        self.discriminator = DomainDiscriminator(latent_dim, hidden_dim // 4, num_domains)
    
    def forward(self, x):
        z = self.encoder(x)
        return z

# ============================================================================
# 数据加载
# ============================================================================
class SimpleH5Dataset(Dataset):
    def __init__(self, h5_path, max_samples=None):
        self.h5_path = h5_path
        with h5py.File(h5_path, 'r') as f:
            self.shape = f['X'].shape
            self.num_samples = min(self.shape[0], max_samples) if max_samples else self.shape[0]
        self._file = None
    
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        f = self._get_file()
        counts = torch.tensor(f['X'][idx], dtype=torch.float32)
        return counts

print("\n加载数据...")

# ============================================================================
# 加载模型
# ============================================================================
print(f"\n加载模型: {CHECKPOINT_PATH}")

if not os.path.exists(CHECKPOINT_PATH):
    print(f"❌ 检查点不存在: {CHECKPOINT_PATH}")
    print("请先运行 python train_mmd_aae.py")
    exit(1)

checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
input_dim = 18080

model = MMD_AAE(input_dim=input_dim).to(DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ 模型加载成功 (Epoch {checkpoint['epoch']})")

# ============================================================================
# 提取隐空间表示
# ============================================================================
print("\n提取隐空间表示...")

all_z = []
all_domains = []
domain_names = []

for domain_id, domain in enumerate(DOMAIN_CONFIGS):
    print(f"  处理 {domain['name']}...")
    
    dataset = SimpleH5Dataset(domain['path'], max_samples=SAMPLES_PER_DOMAIN)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    z_list = []
    with torch.no_grad():
        for batch in loader:
            x = batch.to(DEVICE)
            z = model(x)
            z_list.append(z.cpu())
            
            if len(torch.cat(z_list)) >= SAMPLES_PER_DOMAIN:
                break
    
    z_domain = torch.cat(z_list)[:SAMPLES_PER_DOMAIN]
    all_z.append(z_domain)
    all_domains.extend([domain_id] * len(z_domain))
    domain_names.append(domain['name'])
    
    print(f"    采样 {len(z_domain)} 个细胞")

all_z = torch.cat(all_z).numpy()
all_domains = np.array(all_domains)

print(f"\n总样本: {len(all_z)}, 隐空间维度: {all_z.shape[1]}")

# ============================================================================
# t-SNE 可视化
# ============================================================================
print("\n运行 t-SNE (可能需要 1-2 分钟)...")

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
z_2d = tsne.fit_transform(all_z)

print("✅ t-SNE 完成")

# ============================================================================
# 绘图
# ============================================================================
print("\n生成可视化图...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 图 1: t-SNE 按域着色
ax1 = axes[0]
for domain_id, domain in enumerate(DOMAIN_CONFIGS):
    mask = all_domains == domain_id
    ax1.scatter(
        z_2d[mask, 0], z_2d[mask, 1],
        c=domain['color'],
        label=domain['name'],
        alpha=0.6,
        s=10
    )

ax1.set_title('t-SNE of Latent Space (by Domain)', fontsize=14)
ax1.set_xlabel('t-SNE 1')
ax1.set_ylabel('t-SNE 2')
ax1.legend(title='Domain')
ax1.grid(True, alpha=0.3)

# 图 2: 域重叠可视化 (Kernel Density)
ax2 = axes[1]
from scipy.stats import gaussian_kde

for domain_id, domain in enumerate(DOMAIN_CONFIGS):
    mask = all_domains == domain_id
    x = z_2d[mask, 0]
    y = z_2d[mask, 1]
    
    # 计算密度
    try:
        xy = np.vstack([x, y])
        kde = gaussian_kde(xy)
        
        # 绘制边界
        xmin, xmax = z_2d[:, 0].min(), z_2d[:, 0].max()
        ymin, ymax = z_2d[:, 1].min(), z_2d[:, 1].max()
        xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(kde(positions).T, xx.shape)
        
        ax2.contour(xx, yy, f, levels=3, colors=domain['color'], alpha=0.8)
    except:
        pass

for domain_id, domain in enumerate(DOMAIN_CONFIGS):
    mask = all_domains == domain_id
    ax2.scatter(
        z_2d[mask, 0], z_2d[mask, 1],
        c=domain['color'],
        label=domain['name'],
        alpha=0.3,
        s=5
    )

ax2.set_title('Domain Overlap Visualization', fontsize=14)
ax2.set_xlabel('t-SNE 1')
ax2.set_ylabel('t-SNE 2')
ax2.legend(title='Domain')
ax2.grid(True, alpha=0.3)

plt.tight_layout()

# 保存图片
output_path = os.path.join(OUTPUT_DIR, 'tsne_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"\n✅ 图片已保存: {output_path}")

# 显示总结
print("\n" + "=" * 60)
print("可视化完成!")
print("=" * 60)
print(f"\n查看结果: {output_path}")
print("\n解读指南:")
print("  - 如果三个域完全分离: 域对齐未生效")
print("  - 如果三个域明显重叠: 域对齐成功! ✅")
print("  - 如果三个域部分重叠: 需要更多训练或调整权重")
print("=" * 60)

plt.close()

#!/usr/bin/env python
"""
visualize_mmd_aae.py - 可视化 MMD-AAE 训练结果 (支持实验名称)

使用方法:
    cd ~/state/src
    
    # 指定实验名
    python visualize_mmd_aae.py --exp_name r1.0_m10.0_a0.5
    
    # 或使用默认路径
    python visualize_mmd_aae.py --checkpoint /path/to/best_model.pt
"""

import os
import argparse
import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader

# ============================================================================
# 参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='MMD-AAE Visualization')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='实验名称 (如 r1.0_m10.0_a0.5)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='检查点路径 (直接指定)')
    parser.add_argument('--samples', type=int, default=500,
                        help='每个域采样数量 (默认: 500)')
    return parser.parse_args()


# ============================================================================
# 配置
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
OUTPUT_DIR = f"{BASE_DIR}/visualizations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_CONFIGS = [
    {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5", "color": "#E74C3C"},
    {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5", "color": "#3498DB"},
    {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5", "color": "#2ECC71"},
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ============================================================================
# 模型定义
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


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MMD-AAE 可视化")
    print("=" * 60)
    
    # 确定检查点路径
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        exp_name = os.path.basename(os.path.dirname(checkpoint_path))
    elif args.exp_name:
        checkpoint_path = f"{BASE_DIR}/checkpoints/mmd_aae/{args.exp_name}/best_model.pt"
        exp_name = args.exp_name
    else:
        # 查找最新的实验
        exp_dir = f"{BASE_DIR}/checkpoints/mmd_aae"
        if os.path.exists(exp_dir):
            exps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
            if exps:
                # 按修改时间排序
                exps.sort(key=lambda x: os.path.getmtime(os.path.join(exp_dir, x)), reverse=True)
                exp_name = exps[0]
                checkpoint_path = f"{exp_dir}/{exp_name}/best_model.pt"
            else:
                checkpoint_path = f"{exp_dir}/best_model.pt"
                exp_name = "default"
        else:
            print("❌ 找不到检查点目录")
            return
    
    print(f"\n实验名称: {exp_name}")
    print(f"检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 检查点不存在: {checkpoint_path}")
        return
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
    
    # 显示保存的 lambda 信息
    if 'lambdas' in checkpoint:
        lambdas = checkpoint['lambdas']
        print(f"\nLambda 配置:")
        print(f"  λ_recon = {lambdas.get('recon', 'N/A')}")
        print(f"  λ_mmd   = {lambdas.get('mmd', 'N/A')}")
        print(f"  λ_adv   = {lambdas.get('adv', 'N/A')}")
    
    print(f"训练 Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    input_dim = 18080
    model = MMD_AAE(input_dim=input_dim).to(DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✅ 模型加载成功")
    
    # 提取隐空间表示
    print("\n提取隐空间表示...")
    
    all_z = []
    all_domains = []
    
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        print(f"  处理 {domain['name']}...")
        
        dataset = SimpleH5Dataset(domain['path'], max_samples=args.samples)
        loader = DataLoader(dataset, batch_size=64, shuffle=True)
        
        z_list = []
        with torch.no_grad():
            for batch in loader:
                x = batch.to(DEVICE)
                z = model(x)
                z_list.append(z.cpu())
                
                if len(torch.cat(z_list)) >= args.samples:
                    break
        
        z_domain = torch.cat(z_list)[:args.samples]
        all_z.append(z_domain)
        all_domains.extend([domain_id] * len(z_domain))
        
        print(f"    采样 {len(z_domain)} 个细胞")
    
    all_z = torch.cat(all_z).numpy()
    all_domains = np.array(all_domains)
    
    print(f"\n总样本: {len(all_z)}, 隐空间维度: {all_z.shape[1]}")
    
    # t-SNE
    print("\n运行 t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(all_z)
    print("✅ t-SNE 完成")
    
    # 绘图
    print("\n生成可视化图...")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 添加标题显示 lambda 信息
    if 'lambdas' in checkpoint:
        lambdas = checkpoint['lambdas']
        title_str = f"Exp: {exp_name} | λ_r={lambdas.get('recon', '?')}, λ_m={lambdas.get('mmd', '?')}, λ_a={lambdas.get('adv', '?')}"
    else:
        title_str = f"Exp: {exp_name}"
    
    fig.suptitle(title_str, fontsize=14, fontweight='bold')
    
    # 图 1: t-SNE
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
    
    ax1.set_title('t-SNE of Latent Space (by Domain)', fontsize=12)
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Domain')
    ax1.grid(True, alpha=0.3)
    
    # 图 2: 密度
    ax2 = axes[1]
    try:
        from scipy.stats import gaussian_kde
        
        for domain_id, domain in enumerate(DOMAIN_CONFIGS):
            mask = all_domains == domain_id
            x = z_2d[mask, 0]
            y = z_2d[mask, 1]
            
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy)
            
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
    
    ax2.set_title('Domain Overlap Visualization', fontsize=12)
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(title='Domain')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存 (使用实验名称命名)
    output_path = os.path.join(OUTPUT_DIR, f'tsne_{exp_name}.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {output_path}")
    
    print("\n" + "=" * 60)
    print("可视化完成!")
    print(f"实验: {exp_name}")
    print(f"图片: {output_path}")
    print("=" * 60)
    
    plt.close()


if __name__ == "__main__":
    main()

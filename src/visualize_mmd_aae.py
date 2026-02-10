#!/usr/bin/env python
"""
visualize_mmd_aae.py - 可视化 MMD-AAE 训练结果 (自动适配 v1/v2)

自动检测 checkpoint 中的模型架构 (BatchNorm/LayerNorm, latent_dim 等)

使用方法:
    cd ~/state/src
    python visualize_mmd_aae.py --exp_name v2_m10_ep50
    python visualize_mmd_aae.py --exp_name v2_m10_ep50 --samples 3000
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 模式
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.utils.data import Dataset, DataLoader


def parse_args():
    parser = argparse.ArgumentParser(description='MMD-AAE Visualization')
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--samples', type=int, default=2000,
                        help='每个域的采样数量 (默认: 2000)')
    return parser.parse_args()


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
# 从 checkpoint keys 自动检测架构
# ============================================================================
def detect_architecture(state_dict):
    """从 checkpoint 中自动推断模型架构"""
    info = {
        'use_layernorm': False,
        'input_dim': None,
        'latent_dim': None,
        'hidden_dim': None,
        'mid_dim': None,
    }
    
    for key in state_dict.keys():
        # 检测 LayerNorm vs BatchNorm
        if 'LayerNorm' in key or 'layer_norm' in key:
            info['use_layernorm'] = True
        
        # 从 encoder 第一层推断 input_dim 和 hidden_dim
        if key == 'encoder.net.0.weight':
            info['hidden_dim'] = state_dict[key].shape[0]
            info['input_dim'] = state_dict[key].shape[1]
        
        # 检测中间层维度 (encoder 第二个 Linear 层)
        if key == 'encoder.net.4.weight':
            info['mid_dim'] = state_dict[key].shape[0]
        
        # 检测 latent_dim (encoder 最后一层)
        if key == 'encoder.net.8.weight':
            info['latent_dim'] = state_dict[key].shape[0]
    
    # 如果没找到 8.weight, 试 6.weight (可能只有2层)
    if info['latent_dim'] is None:
        for key in state_dict.keys():
            if key == 'encoder.net.6.weight':
                info['latent_dim'] = state_dict[key].shape[0]
            elif key == 'encoder.net.4.weight' and info['latent_dim'] is None:
                info['latent_dim'] = state_dict[key].shape[0]
    
    # 检测 BatchNorm 的存在
    for key in state_dict.keys():
        if 'weight' in key and 'encoder.net.1.' in key:
            # 如果第 1 层有 weight 参数，检查是否有 running_mean (BatchNorm 标志)
            bn_key = key.replace('weight', 'running_mean')
            if bn_key in state_dict:
                info['use_layernorm'] = False
            else:
                info['use_layernorm'] = True
            break
    
    return info


def build_encoder(info):
    """根据检测到的架构信息构建 Encoder"""
    input_dim = info['input_dim']
    hidden_dim = info['hidden_dim']
    mid_dim = info.get('mid_dim', hidden_dim // 2)
    latent_dim = info['latent_dim']
    NormLayer = nn.LayerNorm if info['use_layernorm'] else nn.BatchNorm1d
    
    encoder = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        NormLayer(hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_dim, mid_dim),
        NormLayer(mid_dim),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(mid_dim, latent_dim),
    )
    return encoder


class FlexibleModel(nn.Module):
    """灵活模型，自动适配 checkpoint"""
    def __init__(self, state_dict):
        super().__init__()
        info = detect_architecture(state_dict)
        self.info = info
        self.encoder = build_encoder(info)
        
        # 只需要 encoder，不需要加载 decoder/discriminator
        encoder_state = {k.replace('encoder.', ''): v 
                        for k, v in state_dict.items() 
                        if k.startswith('encoder.')}
        self.encoder.load_state_dict(encoder_state)
    
    def forward(self, x):
        return self.encoder(x)


# ============================================================================
# Dataset
# ============================================================================
class SimpleH5Dataset(Dataset):
    def __init__(self, h5_path, max_samples=None, use_normalize=True):
        self.h5_path = h5_path
        self.use_normalize = use_normalize
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
        if self.use_normalize:
            counts = torch.log1p(counts)
            norm = counts.norm(p=2)
            if norm > 0:
                counts = counts / norm
        return counts


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MMD-AAE 可视化 (自动架构检测)")
    print("=" * 60)
    
    # ===== 1. 找到 checkpoint =====
    if args.checkpoint:
        checkpoint_path = args.checkpoint
        exp_name = os.path.basename(os.path.dirname(checkpoint_path))
    elif args.exp_name:
        checkpoint_path = f"{BASE_DIR}/checkpoints/mmd_aae/{args.exp_name}/best_model.pt"
        exp_name = args.exp_name
    else:
        exp_dir = f"{BASE_DIR}/checkpoints/mmd_aae"
        if os.path.exists(exp_dir):
            exps = [d for d in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, d))]
            if exps:
                exps.sort(key=lambda x: os.path.getmtime(os.path.join(exp_dir, x)), reverse=True)
                exp_name = exps[0]
                checkpoint_path = f"{exp_dir}/{exp_name}/best_model.pt"
                print(f"自动选择最新实验: {exp_name}")
            else:
                print("❌ 没有找到实验目录"); return
        else:
            print("❌ 找不到检查点目录"); return
    
    print(f"\n实验: {exp_name}")
    print(f"检查点: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}"); return
    
    # ===== 2. 加载 checkpoint =====
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    
    print(f"\n--- Checkpoint 信息 ---")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Loss: {checkpoint.get('loss', '?')}")
    
    version = checkpoint.get('version', 'v1')
    print(f"  Version: {version}")
    
    if 'lambdas' in checkpoint:
        l = checkpoint['lambdas']
        print(f"  Lambda: r={l.get('recon')}, m={l.get('mmd')}, a={l.get('adv')}")
    
    if 'latent_dim' in checkpoint:
        print(f"  Saved latent_dim: {checkpoint['latent_dim']}")
    
    # ===== 3. 自动检测架构 =====
    state_dict = checkpoint['model_state_dict']
    
    print(f"\n--- 自动检测架构 ---")
    arch_info = detect_architecture(state_dict)
    print(f"  input_dim: {arch_info['input_dim']}")
    print(f"  hidden_dim: {arch_info['hidden_dim']}")
    print(f"  mid_dim: {arch_info['mid_dim']}")
    print(f"  latent_dim: {arch_info['latent_dim']}")
    print(f"  use_layernorm: {arch_info['use_layernorm']}")
    
    # 检测是否使用了输入归一化 (v2 使用，v1 不使用)
    use_normalize = arch_info['use_layernorm']  # LayerNorm 版本使用了输入归一化
    if version == 'v2':
        use_normalize = True
    print(f"  use_normalize: {use_normalize}")
    
    # ===== 4. 构建模型 (只需 Encoder) =====
    try:
        model = FlexibleModel(state_dict).to(DEVICE)
        model.eval()
        print("✅ 模型加载成功 (自动适配)")
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("\n[DEBUG] checkpoint keys:")
        for k, v in state_dict.items():
            if 'encoder' in k:
                print(f"  {k}: {v.shape}")
        return
    
    # ===== 5. 提取隐空间 =====
    print(f"\n--- 提取隐空间 (每域 {args.samples} 个样本) ---")
    
    all_z = []
    all_domains = []
    total_points = 0
    
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        if not os.path.exists(domain['path']):
            print(f"  ⚠️ 跳过 {domain['name']}: 文件不存在 {domain['path']}")
            continue
        
        dataset = SimpleH5Dataset(domain['path'], max_samples=args.samples, 
                                  use_normalize=use_normalize)
        loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
        
        z_list = []
        sample_count = 0
        
        with torch.no_grad():
            for batch in loader:
                x = batch.to(DEVICE)
                z = model(x)
                z_list.append(z.cpu())
                sample_count += x.size(0)
                if sample_count >= args.samples:
                    break
        
        if len(z_list) == 0:
            print(f"  ⚠️ {domain['name']}: 0 个样本!")
            continue
        
        z_domain = torch.cat(z_list)[:args.samples]
        all_z.append(z_domain)
        all_domains.extend([domain_id] * len(z_domain))
        total_points += len(z_domain)
        print(f"  ✅ {domain['name']}: {len(z_domain)} 个样本, z.shape={z_domain.shape}")
    
    if total_points == 0:
        print("❌ 没有采集到任何样本!"); return
    
    all_z = torch.cat(all_z).numpy()
    all_domains = np.array(all_domains)
    
    print(f"\n总计: {total_points} 个点, latent_dim={all_z.shape[1]}")
    
    # ===== 6. t-SNE =====
    print(f"\n运行 t-SNE...")
    perplexity = min(30, total_points // 4)  # 自适应 perplexity
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(all_z)
    print("✅ t-SNE 完成")
    
    # ===== 7. 绘图 =====
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    if 'lambdas' in checkpoint:
        l = checkpoint['lambdas']
        title = (f"Exp: {exp_name} | λ_r={l.get('recon')}, λ_m={l.get('mmd')}, "
                 f"λ_a={l.get('adv')} | latent={arch_info['latent_dim']} | "
                 f"{'LayerNorm' if arch_info['use_layernorm'] else 'BatchNorm'} | "
                 f"N={total_points}")
    else:
        title = f"Exp: {exp_name} | N={total_points}"
    fig.suptitle(title, fontsize=12, fontweight='bold')
    
    # 散点图
    ax1 = axes[0]
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        mask = all_domains == domain_id
        n_points = mask.sum()
        if n_points > 0:
            ax1.scatter(z_2d[mask, 0], z_2d[mask, 1], c=domain['color'],
                        label=f"{domain['name']} (n={n_points})",
                        alpha=0.5, s=20, edgecolors='none')
    ax1.set_title('t-SNE of Latent Space (by Domain)')
    ax1.set_xlabel('t-SNE 1')
    ax1.set_ylabel('t-SNE 2')
    ax1.legend(title='Domain', fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 密度图
    ax2 = axes[1]
    try:
        from scipy.stats import gaussian_kde
        for domain_id, domain in enumerate(DOMAIN_CONFIGS):
            mask = all_domains == domain_id
            if mask.sum() < 10:
                continue
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
    except Exception as e:
        print(f"  KDE 绘制跳过: {e}")
    
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        mask = all_domains == domain_id
        if mask.sum() > 0:
            ax2.scatter(z_2d[mask, 0], z_2d[mask, 1], c=domain['color'],
                        label=f"{domain['name']} (n={mask.sum()})",
                        alpha=0.4, s=12, edgecolors='none')
    ax2.set_title('Domain Overlap + Density')
    ax2.set_xlabel('t-SNE 1')
    ax2.set_ylabel('t-SNE 2')
    ax2.legend(title='Domain', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(OUTPUT_DIR, f'tsne_{exp_name}.png')
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    print(f"\n✅ 图片已保存: {output_path}")
    print("=" * 60)
    plt.close()


if __name__ == "__main__":
    main()

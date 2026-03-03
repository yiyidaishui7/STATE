#!/usr/bin/env python
"""
visualize_results.py - MMD-AAE 实验结果可视化

从 evaluations/ 下的 JSON 文件读取指标，生成:
  1. 多指标柱状图对比 (Domain Cls Acc, Silhouette, MMD, CORAL, ReconMSE)
  2. 雷达图 (综合对比各实验)
  3. t-SNE 隐空间可视化 (需要 checkpoint)

使用方法:
    cd ~/state/src
    python visualize_results.py                       # 用最新 JSON
    python visualize_results.py --json PATH           # 指定 JSON
    python visualize_results.py --tsne EXP_NAME       # 生成 t-SNE 图
"""

import os
import sys
import json
import glob
import argparse
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import matplotlib.patheffects as pe

# ============================================================================
# 配置
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
EVAL_DIR = f"{BASE_DIR}/evaluations"
VIS_DIR = f"{BASE_DIR}/visualizations"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/mmd_aae"

os.makedirs(VIS_DIR, exist_ok=True)

# 配色方案
COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

DOMAIN_COLORS = {'K562': '#E74C3C', 'RPE1': '#3498DB', 'Jurkat': '#2ECC71'}


def parse_args():
    parser = argparse.ArgumentParser(description='MMD-AAE 结果可视化')
    parser.add_argument('--json', type=str, default=None,
                        help='指定 JSON 文件路径 (默认用最新的)')
    parser.add_argument('--tsne', type=str, default=None,
                        help='对指定实验生成 t-SNE (需要 checkpoint)')
    parser.add_argument('--all_tsne', action='store_true',
                        help='对所有实验生成 t-SNE')
    return parser.parse_args()


def load_latest_json(json_path=None):
    """加载最新的评估 JSON"""
    if json_path:
        with open(json_path) as f:
            return json.load(f)
    
    json_files = sorted(glob.glob(os.path.join(EVAL_DIR, 'eval_*.json')))
    if not json_files:
        print("❌ 未找到评估 JSON 文件")
        sys.exit(1)
    
    latest = json_files[-1]
    print(f"📂 加载: {latest}")
    with open(latest) as f:
        return json.load(f)


def safe_val(result, key, default=0.0):
    """安全取值，处理 -inf/nan"""
    v = result.get(key, default)
    if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
        return default
    return v


# ============================================================================
# 图 1: 多指标柱状图
# ============================================================================
def plot_metric_bars(results, output_path):
    """5 个指标的柱状图对比"""
    
    exp_names = [r['exp_name'] for r in results]
    short_names = [n.replace('v3_auto_', '') for n in exp_names]
    n = len(results)
    
    metrics = [
        ('domain_cls_acc', 'Domain Cls Acc ↓', '越低越好 (33.3%=完美)', True),
        ('silhouette_score', 'Silhouette Score ↓', '越接近0越好', True),
        ('mmd_mean', 'MMD ↓', '越低越好', True),
        ('coral_mean', 'CORAL ↓', '越低越好', True),
        ('recon_mse', 'Recon MSE ↓', '越低越好', True),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    fig.suptitle('MMD-AAE 实验对比 — 各指标柱状图', fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (key, title, subtitle, lower_better) in enumerate(metrics):
        ax = axes[idx]
        vals = [safe_val(r, key) for r in results]
        
        # 找最佳
        if lower_better:
            best_idx = np.argmin(vals) if vals else -1
        else:
            best_idx = np.argmax(vals) if vals else -1
        
        bar_colors = [COLORS[i % len(COLORS)] for i in range(n)]
        if best_idx >= 0:
            bar_colors[best_idx] = '#27ae60'  # 最佳标绿
        
        bars = ax.barh(range(n), vals, color=bar_colors, edgecolor='white', height=0.6)
        
        ax.set_yticks(range(n))
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        
        # 数值标注
        for i, (bar, val) in enumerate(zip(bars, vals)):
            fmt = f'{val:.4f}' if abs(val) < 10 else f'{val:.2f}'
            ax.text(bar.get_width() + ax.get_xlim()[1] * 0.02, i, fmt,
                    va='center', fontsize=9, fontweight='bold' if i == best_idx else 'normal')
        
        # 理想线
        if key == 'domain_cls_acc':
            ax.axvline(x=1/3, color='green', linestyle='--', alpha=0.5, label='理想=33.3%')
            ax.legend(fontsize=8)
        elif key == 'silhouette_score':
            ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='理想=0')
            ax.legend(fontsize=8)
        
        ax.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 柱状图: {output_path}")


# ============================================================================
# 图 2: 雷达图
# ============================================================================
def plot_radar(results, output_path):
    """雷达图综合对比"""
    
    exp_names = [r['exp_name'] for r in results]
    short_names = [n.replace('v3_auto_', '') for n in exp_names]
    
    # 归一化指标到 0-1 区间 (越低越好 → 翻转)
    metrics_keys = ['domain_cls_acc', 'silhouette_score', 'mmd_mean', 'coral_mean', 'recon_mse']
    labels = ['Domain\nCls Acc', 'Silhouette', 'MMD', 'CORAL', 'Recon\nMSE']
    
    raw = np.array([[safe_val(r, k) for k in metrics_keys] for r in results])
    
    # 归一化: 1 = 最好, 0 = 最差
    normed = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-10:
            normed[:, j] = 0.5
        else:
            normed[:, j] = 1.0 - (col - lo) / (hi - lo)  # 翻转: 低值=好=1
    
    n_metrics = len(labels)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'projection': 'polar'})
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.1)
    
    for i, (name, vals) in enumerate(zip(short_names, normed)):
        values = vals.tolist() + vals[:1].tolist()
        color = COLORS[i % len(COLORS)]
        ax.plot(angles, values, 'o-', color=color, linewidth=2, label=name, markersize=5)
        ax.fill(angles, values, alpha=0.08, color=color)
    
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=9)
    ax.set_title('实验综合对比 (外圈=更好)', fontsize=14, fontweight='bold', pad=20)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 雷达图: {output_path}")


# ============================================================================
# 图 3: Lambda vs 指标散点图
# ============================================================================
def plot_lambda_vs_metrics(results, output_path):
    """Lambda 值与各指标的关系"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('λ_mmd 对域对齐指标的影响', fontsize=14, fontweight='bold')
    
    lambdas = []
    for r in results:
        lm = r.get('lambdas', {})
        if isinstance(lm, dict):
            lambdas.append(lm.get('mmd', 0))
        else:
            lambdas.append(0)
    
    metrics = [
        ('domain_cls_acc', 'Domain Cls Acc', '越低越好'),
        ('silhouette_score', 'Silhouette Score', '越接近0越好'),
        ('recon_mse', 'Recon MSE', '不应太高'),
    ]
    
    for idx, (key, title, note) in enumerate(metrics):
        ax = axes[idx]
        vals = [safe_val(r, key) for r in results]
        
        for i, (lm, v) in enumerate(zip(lambdas, vals)):
            name = results[i]['exp_name'].replace('v3_auto_', '')
            ax.scatter(lm, v, c=COLORS[i % len(COLORS)], s=120, zorder=5, edgecolors='white')
            ax.annotate(name, (lm, v), fontsize=8, ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('λ_mmd', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title}\n({note})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if key == 'domain_cls_acc':
            ax.axhline(y=1/3, color='green', linestyle='--', alpha=0.5, label='理想=33.3%')
            ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ Lambda 散点图: {output_path}")


# ============================================================================
# 图 4: 排行总结卡片
# ============================================================================
def plot_summary_card(results, output_path):
    """生成结果总结卡片"""
    
    # 排序
    sorted_results = sorted(results, key=lambda x: safe_val(x, 'domain_cls_acc', 1.0))
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(results) * 0.8 + 2)))
    ax.axis('off')
    
    # 标题
    ax.text(0.5, 0.98, 'MMD-AAE 实验排行榜', fontsize=18, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.93, '按 Domain Classification Accuracy 排序 (越低=域对齐越好)',
            fontsize=11, ha='center', va='top', transform=ax.transAxes, color='gray')
    
    # 表格
    cols = ['排名', '实验', 'λ_mmd', 'DomClsAcc', 'Silhouette', 'MMD', 'ReconMSE']
    col_x = [0.02, 0.08, 0.25, 0.38, 0.53, 0.68, 0.83]
    
    y = 0.85
    # 表头
    for c, x in zip(cols, col_x):
        ax.text(x, y, c, fontsize=10, fontweight='bold', va='center', transform=ax.transAxes)
    y -= 0.04
    ax.axhline(y=y, xmin=0.02, xmax=0.95, color='gray', linewidth=0.5, transform=ax.transAxes)
    y -= 0.03
    
    medals = ['🥇', '🥈', '🥉']
    for i, r in enumerate(sorted_results):
        rank = medals[i] if i < 3 else f' {i+1}'
        name = r['exp_name'].replace('v3_auto_', '')
        lm = r.get('lambdas', {})
        lambda_mmd = lm.get('mmd', 0) if isinstance(lm, dict) else 0
        
        row = [
            rank,
            name,
            f'{lambda_mmd:.2f}',
            f'{safe_val(r, "domain_cls_acc"):.4f}',
            f'{safe_val(r, "silhouette_score"):.4f}',
            f'{safe_val(r, "mmd_mean"):.6f}',
            f'{safe_val(r, "recon_mse"):.4f}',
        ]
        
        bg_color = '#e8f8e8' if i == 0 else ('white' if i % 2 == 0 else '#f8f8f8')
        ax.axhspan(y - 0.01, y + 0.03, xmin=0.01, xmax=0.96, color=bg_color, 
                   transform=ax.transAxes, zorder=0)
        
        for val, x in zip(row, col_x):
            fw = 'bold' if i == 0 else 'normal'
            ax.text(x, y, val, fontsize=10, fontweight=fw, va='center', 
                    transform=ax.transAxes, family='monospace')
        y -= 0.06
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ 排行卡片: {output_path}")


# ============================================================================
# 图 5: t-SNE 隐空间可视化
# ============================================================================
def plot_tsne(exp_name, output_path):
    """对指定实验生成 t-SNE 可视化"""
    import torch
    import h5py
    from sklearn.manifold import TSNE
    from torch.utils.data import Dataset, DataLoader
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DOMAIN_CONFIGS = [
        {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
        {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
        {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
    ]
    
    # 加载 checkpoint
    for ckpt_name in ['final_model.pt', 'best_model.pt']:
        ckpt_path = os.path.join(CHECKPOINT_DIR, exp_name, ckpt_name)
        if os.path.exists(ckpt_path):
            break
    else:
        print(f"  ❌ 未找到 checkpoint: {exp_name}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    global_mean = checkpoint.get('global_mean', None)
    global_std = checkpoint.get('global_std', None)
    if global_mean is not None:
        global_mean = global_mean.cpu()
        global_std = global_std.cpu()
    
    # 构建 encoder
    import torch.nn as nn
    
    encoder_keys = [k for k in state_dict if k.startswith('encoder.')]
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    
    # 推断架构
    first_weight = state_dict.get('encoder.net.0.weight', state_dict.get('encoder.0.weight', None))
    if first_weight is None:
        for k, v in state_dict.items():
            if 'encoder' in k and 'weight' in k and v.dim() == 2:
                first_weight = v
                break
    
    input_dim = first_weight.shape[1] if first_weight is not None else 18080
    
    # 检测 latent_dim
    latent_dim = checkpoint.get('latent_dim', 64)
    
    # 检测是否使用 net. prefix
    use_net = any('net.' in k for k in encoder_keys)
    
    # 构建 encoder
    layers = []
    if use_net:
        # 便利 encoder.net.{i}.weight 找所有线性层
        weight_keys = sorted([k for k in encoder_state if k.startswith('net.') and 'weight' in k and '.weight' == k[-7:]])
        dims = []
        for k in weight_keys:
            w = encoder_state[k]
            if w.dim() == 2:
                dims.append((w.shape[1], w.shape[0]))
        
        for i, (in_d, out_d) in enumerate(dims):
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 1:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        encoder = nn.Sequential(*layers)
        # 重映射 state dict keys
        new_state = {}
        for k, v in encoder_state.items():
            new_key = k.replace('net.', '')
            new_state[new_key] = v
        encoder.load_state_dict(new_state, strict=False)
    else:
        # 直接 encoder.{i}
        weight_keys = sorted([k for k in encoder_state if 'weight' in k and k.replace('.weight', '').isdigit()],
                              key=lambda x: int(x.split('.')[0]))
        dims = []
        for k in weight_keys:
            w = encoder_state[k]
            if w.dim() == 2:
                dims.append((w.shape[1], w.shape[0]))
        
        for i, (in_d, out_d) in enumerate(dims):
            layers.append(nn.Linear(in_d, out_d))
            if i < len(dims) - 1:
                layers.append(nn.LayerNorm(out_d))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.1))
        
        encoder = nn.Sequential(*layers)
        encoder.load_state_dict(encoder_state, strict=False)
    
    encoder = encoder.to(DEVICE)
    encoder.eval()
    
    # 提取 latent z
    max_per_domain = 2000
    all_z, all_labels, all_names = [], [], []
    
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        with h5py.File(domain['path'], 'r') as f:
            X = f['X']
            n = min(X.shape[0], max_per_domain)
            data = torch.tensor(X[:n], dtype=torch.float32)
        
        # 归一化
        data = torch.log1p(data)
        if global_mean is not None:
            data = (data - global_mean) / global_std
        
        with torch.no_grad():
            z = encoder(data.to(DEVICE)).cpu().numpy()
        
        all_z.append(z)
        all_labels.extend([domain_id] * n)
        all_names.extend([domain['name']] * n)
    
    z_all = np.concatenate(all_z)
    labels = np.array(all_labels)
    
    print(f"  t-SNE on {len(z_all)} samples, dim={z_all.shape[1]}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    z_2d = tsne.fit_transform(z_all)
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    domain_names = ['K562', 'RPE1', 'Jurkat']
    for d_id, d_name in enumerate(domain_names):
        mask = labels == d_id
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=DOMAIN_COLORS[d_name], label=d_name,
                   s=8, alpha=0.5, edgecolors='none')
    
    ax.legend(fontsize=12, markerscale=3)
    ax.set_title(f't-SNE: {exp_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✅ t-SNE: {output_path}")


# ============================================================================
# 主函数
# ============================================================================
def main():
    args = parse_args()
    
    print("=" * 60)
    print("MMD-AAE 结果可视化")
    print("=" * 60)
    
    # 只做 t-SNE
    if args.tsne:
        tsne_path = os.path.join(VIS_DIR, f'tsne_{args.tsne}.png')
        plot_tsne(args.tsne, tsne_path)
        return
    
    # 加载 JSON
    results = load_latest_json(args.json)
    
    if not results:
        print("❌ JSON 为空或无结果")
        return
    
    print(f"  共 {len(results)} 个实验结果\n")
    
    # 生成所有图表
    print("📊 生成图表...")
    
    plot_metric_bars(results, os.path.join(VIS_DIR, 'metric_bars.png'))
    plot_radar(results, os.path.join(VIS_DIR, 'radar.png'))
    plot_lambda_vs_metrics(results, os.path.join(VIS_DIR, 'lambda_scatter.png'))
    plot_summary_card(results, os.path.join(VIS_DIR, 'summary_card.png'))
    
    # t-SNE for all
    if args.all_tsne:
        print("\n🔬 生成 t-SNE...")
        for r in results:
            name = r['exp_name']
            tsne_path = os.path.join(VIS_DIR, f'tsne_{name}.png')
            try:
                plot_tsne(name, tsne_path)
            except Exception as e:
                print(f"  ⚠️ {name}: {e}")
    
    print(f"\n✅ 所有图表已保存到: {VIS_DIR}")
    print("  - metric_bars.png   : 指标柱状图对比")
    print("  - radar.png         : 雷达图综合对比")
    print("  - lambda_scatter.png: λ_mmd 影响散点图")
    print("  - summary_card.png  : 排行总结卡片")


if __name__ == "__main__":
    main()

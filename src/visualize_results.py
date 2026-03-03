#!/usr/bin/env python
"""
visualize_results.py - MMD-AAE experiment result visualization

Reads evaluation JSON files and generates:
  1. Metric bar chart comparison
  2. Radar plot (multi-metric overview)
  3. Lambda scatter plot
  4. Ranking summary card
  5. t-SNE latent space visualization

Usage:
    cd ~/state/src
    python visualize_results.py                       # use latest JSON
    python visualize_results.py --json PATH           # specify JSON
    python visualize_results.py --tsne EXP_NAME       # t-SNE for one exp
    python visualize_results.py --all_tsne            # t-SNE for all
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

# Force DejaVu Sans font (available on this server)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================================
# Config
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
EVAL_DIR = f"{BASE_DIR}/evaluations"
VIS_DIR = f"{BASE_DIR}/visualizations"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/mmd_aae"

os.makedirs(VIS_DIR, exist_ok=True)

COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6',
          '#1abc9c', '#e67e22', '#34495e', '#16a085', '#c0392b']

DOMAIN_COLORS = {'K562': '#E74C3C', 'RPE1': '#3498DB', 'Jurkat': '#2ECC71'}


def parse_args():
    parser = argparse.ArgumentParser(description='MMD-AAE Result Visualization')
    parser.add_argument('--json', type=str, default=None, help='JSON file path (default: latest)')
    parser.add_argument('--tsne', type=str, default=None, help='Run t-SNE for specified experiment')
    parser.add_argument('--all_tsne', action='store_true', help='Run t-SNE for all experiments')
    return parser.parse_args()


def load_latest_json(json_path=None):
    """Load the latest evaluation JSON"""
    if json_path:
        with open(json_path) as f:
            return json.load(f)
    
    json_files = sorted(glob.glob(os.path.join(EVAL_DIR, 'eval_*.json')))
    if not json_files:
        print("ERROR: No eval JSON files found")
        sys.exit(1)
    
    latest = json_files[-1]
    print(f"Loading: {latest}")
    with open(latest) as f:
        return json.load(f)


def safe_val(result, key, default=0.0):
    """Safe value extraction, handles -inf/nan"""
    v = result.get(key, default)
    if v is None or (isinstance(v, float) and (np.isinf(v) or np.isnan(v))):
        return default
    return v


# ============================================================================
# Chart 1: Metric Bar Chart
# ============================================================================
def plot_metric_bars(results, output_path):
    """Bar chart comparison of 5 metrics"""
    
    exp_names = [r['exp_name'] for r in results]
    short_names = [n.replace('v3_auto_', '') for n in exp_names]
    n = len(results)
    
    metrics = [
        ('domain_cls_acc', 'Domain Cls Acc', 'lower=better (33.3%=perfect)', True),
        ('silhouette_score', 'Silhouette Score', 'closer to 0=better', True),
        ('mmd_mean', 'MMD', 'lower=better', True),
        ('coral_mean', 'CORAL', 'lower=better', True),
        ('recon_mse', 'Recon MSE', 'lower=better', True),
    ]
    
    fig, axes = plt.subplots(1, 5, figsize=(24, 6))
    fig.suptitle('MMD-AAE Experiment Comparison', fontsize=16, fontweight='bold', y=1.02)
    
    for idx, (key, title, subtitle, lower_better) in enumerate(metrics):
        ax = axes[idx]
        vals = [safe_val(r, key) for r in results]
        
        if lower_better:
            best_idx = np.argmin(vals) if vals else -1
        else:
            best_idx = np.argmax(vals) if vals else -1
        
        bar_colors = [COLORS[i % len(COLORS)] for i in range(n)]
        if best_idx >= 0:
            bar_colors[best_idx] = '#27ae60'
        
        bars = ax.barh(range(n), vals, color=bar_colors, edgecolor='white', height=0.6)
        
        ax.set_yticks(range(n))
        ax.set_yticklabels(short_names, fontsize=10)
        ax.set_title(f'{title}\n({subtitle})', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        
        for i, (bar, val) in enumerate(zip(bars, vals)):
            fmt = f'{val:.4f}' if abs(val) < 10 else f'{val:.2f}'
            ax.text(bar.get_width() + ax.get_xlim()[1] * 0.02, i, fmt,
                    va='center', fontsize=9, fontweight='bold' if i == best_idx else 'normal')
        
        if key == 'domain_cls_acc':
            ax.axvline(x=1/3, color='green', linestyle='--', alpha=0.5, label='ideal=33.3%')
            ax.legend(fontsize=8)
        elif key == 'silhouette_score':
            ax.axvline(x=0, color='green', linestyle='--', alpha=0.5, label='ideal=0')
            ax.legend(fontsize=8)
        
        ax.grid(axis='x', alpha=0.2)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Done: bar chart -> {output_path}")


# ============================================================================
# Chart 2: Radar Plot
# ============================================================================
def plot_radar(results, output_path):
    """Radar plot for multi-metric comparison"""
    
    exp_names = [r['exp_name'] for r in results]
    short_names = [n.replace('v3_auto_', '') for n in exp_names]
    
    metrics_keys = ['domain_cls_acc', 'silhouette_score', 'mmd_mean', 'coral_mean', 'recon_mse']
    labels = ['Domain\nCls Acc', 'Silhouette', 'MMD', 'CORAL', 'Recon\nMSE']
    
    raw = np.array([[safe_val(r, k) for k in metrics_keys] for r in results])
    
    # Normalize: 1 = best, 0 = worst (inverted since lower is better)
    normed = np.zeros_like(raw)
    for j in range(raw.shape[1]):
        col = raw[:, j]
        lo, hi = col.min(), col.max()
        if hi - lo < 1e-10:
            normed[:, j] = 0.5
        else:
            normed[:, j] = 1.0 - (col - lo) / (hi - lo)
    
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
    ax.set_title('Experiment Comparison (outer = better)', fontsize=14, fontweight='bold', pad=20)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Done: radar plot -> {output_path}")


# ============================================================================
# Chart 3: Lambda vs Metrics Scatter
# ============================================================================
def plot_lambda_vs_metrics(results, output_path):
    """Scatter plot: lambda_mmd vs alignment metrics"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Effect of lambda_mmd on Domain Alignment', fontsize=14, fontweight='bold')
    
    lambdas = []
    for r in results:
        lm = r.get('lambdas', {})
        if isinstance(lm, dict):
            lambdas.append(lm.get('mmd', 0))
        else:
            lambdas.append(0)
    
    metrics = [
        ('domain_cls_acc', 'Domain Cls Acc', 'lower=better'),
        ('silhouette_score', 'Silhouette Score', 'closer to 0=better'),
        ('recon_mse', 'Recon MSE', 'lower=better'),
    ]
    
    for idx, (key, title, note) in enumerate(metrics):
        ax = axes[idx]
        vals = [safe_val(r, key) for r in results]
        
        for i, (lm, v) in enumerate(zip(lambdas, vals)):
            name = results[i]['exp_name'].replace('v3_auto_', '')
            ax.scatter(lm, v, c=COLORS[i % len(COLORS)], s=120, zorder=5, edgecolors='white')
            ax.annotate(name, (lm, v), fontsize=8, ha='left', va='bottom',
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('lambda_mmd', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f'{title}\n({note})', fontsize=11, fontweight='bold')
        ax.grid(alpha=0.3)
        
        if key == 'domain_cls_acc':
            ax.axhline(y=1/3, color='green', linestyle='--', alpha=0.5, label='ideal=33.3%')
            ax.legend()
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Done: lambda scatter -> {output_path}")


# ============================================================================
# Chart 4: Summary Ranking Card
# ============================================================================
def plot_summary_card(results, output_path):
    """Generate summary ranking table as image"""
    
    sorted_results = sorted(results, key=lambda x: safe_val(x, 'domain_cls_acc', 1.0))
    
    fig, ax = plt.subplots(figsize=(14, max(3, len(results) * 0.8 + 2)))
    ax.axis('off')
    
    ax.text(0.5, 0.98, 'MMD-AAE Experiment Ranking', fontsize=18, fontweight='bold',
            ha='center', va='top', transform=ax.transAxes)
    ax.text(0.5, 0.93, 'Sorted by Domain Classification Accuracy (lower = better alignment)',
            fontsize=11, ha='center', va='top', transform=ax.transAxes, color='gray')
    
    cols = ['Rank', 'Experiment', 'lam_mmd', 'DomClsAcc', 'Silhouette', 'MMD', 'ReconMSE']
    col_x = [0.02, 0.08, 0.25, 0.38, 0.53, 0.68, 0.83]
    
    y = 0.85
    for c, x in zip(cols, col_x):
        ax.text(x, y, c, fontsize=10, fontweight='bold', va='center', transform=ax.transAxes)
    y -= 0.04
    ax.plot([0.02, 0.95], [y, y], color='gray', linewidth=0.5, transform=ax.transAxes, clip_on=False)
    y -= 0.03
    
    medals = ['#1', '#2', '#3']
    for i, r in enumerate(sorted_results):
        rank = medals[i] if i < 3 else f'#{i+1}'
        name = r['exp_name'].replace('v3_auto_', '')
        lm = r.get('lambdas', {})
        lambda_mmd = lm.get('mmd', 0) if isinstance(lm, dict) else 0
        
        row = [
            rank, name, f'{lambda_mmd:.2f}',
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
    print(f"  Done: summary card -> {output_path}")


# ============================================================================
# Chart 5: t-SNE Latent Space
# ============================================================================
def plot_tsne(exp_name, output_path):
    """Generate t-SNE visualization for specified experiment"""
    import torch
    import torch.nn as nn
    import h5py
    from sklearn.manifold import TSNE
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    DOMAIN_CONFIGS = [
        {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
        {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
        {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
    ]
    
    # Load checkpoint
    for ckpt_name in ['final_model.pt', 'best_model.pt']:
        ckpt_path = os.path.join(CHECKPOINT_DIR, exp_name, ckpt_name)
        if os.path.exists(ckpt_path):
            break
    else:
        print(f"  ERROR: No checkpoint found for {exp_name}")
        return
    
    checkpoint = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    global_mean = checkpoint.get('global_mean', None)
    global_std = checkpoint.get('global_std', None)
    if global_mean is not None:
        global_mean = global_mean.cpu()
        global_std = global_std.cpu()
    
    # Build encoder from state dict
    encoder_keys = [k for k in state_dict if k.startswith('encoder.')]
    encoder_state = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}
    
    # Detect architecture
    first_weight = state_dict.get('encoder.net.0.weight', state_dict.get('encoder.0.weight', None))
    if first_weight is None:
        for k, v in state_dict.items():
            if 'encoder' in k and 'weight' in k and v.dim() == 2:
                first_weight = v
                break
    
    use_net = any('net.' in k for k in encoder_keys)
    
    # Find all linear layer dimensions
    prefix = 'net.' if use_net else ''
    weight_keys = sorted(
        [k for k in encoder_state if k.startswith(prefix) and k.endswith('.weight')],
        key=lambda x: int(x.replace(prefix, '').split('.')[0])
    )
    
    dims = []
    for k in weight_keys:
        w = encoder_state[k]
        if w.dim() == 2:
            dims.append((w.shape[1], w.shape[0]))
    
    layers = []
    for i, (in_d, out_d) in enumerate(dims):
        layers.append(nn.Linear(in_d, out_d))
        if i < len(dims) - 1:
            layers.append(nn.LayerNorm(out_d))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
    
    encoder = nn.Sequential(*layers)
    
    # Remap keys
    new_state = {}
    for k, v in encoder_state.items():
        new_key = k.replace('net.', '') if use_net else k
        new_state[new_key] = v
    encoder.load_state_dict(new_state, strict=False)
    
    encoder = encoder.to(DEVICE)
    encoder.eval()
    
    # Extract latent vectors
    max_per_domain = 2000
    all_z, all_labels = [], []
    
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        with h5py.File(domain['path'], 'r') as f:
            X = f['X']
            n = min(X.shape[0], max_per_domain)
            data = torch.tensor(X[:n], dtype=torch.float32)
        
        data = torch.log1p(data)
        if global_mean is not None:
            data = (data - global_mean) / global_std
        
        with torch.no_grad():
            z = encoder(data.to(DEVICE)).cpu().numpy()
        
        all_z.append(z)
        all_labels.extend([domain_id] * n)
    
    z_all = np.concatenate(all_z)
    labels = np.array(all_labels)
    
    print(f"  t-SNE on {len(z_all)} samples, dim={z_all.shape[1]}...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    z_2d = tsne.fit_transform(z_all)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    domain_names = ['K562', 'RPE1', 'Jurkat']
    for d_id, d_name in enumerate(domain_names):
        mask = labels == d_id
        ax.scatter(z_2d[mask, 0], z_2d[mask, 1],
                   c=DOMAIN_COLORS[d_name], label=d_name,
                   s=8, alpha=0.5, edgecolors='none')
    
    ax.legend(fontsize=12, markerscale=3)
    ax.set_title(f't-SNE Latent Space: {exp_name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('t-SNE 1')
    ax.set_ylabel('t-SNE 2')
    ax.grid(alpha=0.2)
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Done: t-SNE -> {output_path}")


# ============================================================================
# Main
# ============================================================================
def main():
    args = parse_args()
    
    print("=" * 60)
    print("MMD-AAE Result Visualization")
    print("=" * 60)
    
    # t-SNE only
    if args.tsne:
        tsne_path = os.path.join(VIS_DIR, f'tsne_{args.tsne}.png')
        plot_tsne(args.tsne, tsne_path)
        return
    
    # Load JSON
    results = load_latest_json(args.json)
    
    if not results:
        print("ERROR: JSON is empty")
        return
    
    print(f"  {len(results)} experiment results loaded\n")
    
    # Generate all charts
    print("Generating charts...")
    
    plot_metric_bars(results, os.path.join(VIS_DIR, 'metric_bars.png'))
    plot_radar(results, os.path.join(VIS_DIR, 'radar.png'))
    plot_lambda_vs_metrics(results, os.path.join(VIS_DIR, 'lambda_scatter.png'))
    plot_summary_card(results, os.path.join(VIS_DIR, 'summary_card.png'))
    
    # t-SNE for all
    if args.all_tsne:
        print("\nGenerating t-SNE plots...")
        for r in results:
            name = r['exp_name']
            tsne_path = os.path.join(VIS_DIR, f'tsne_{name}.png')
            try:
                plot_tsne(name, tsne_path)
            except Exception as e:
                print(f"  WARNING: {name}: {e}")
    
    print(f"\nAll charts saved to: {VIS_DIR}")
    print("  - metric_bars.png    : Metric bar chart")
    print("  - radar.png          : Radar comparison")
    print("  - lambda_scatter.png : Lambda effect scatter")
    print("  - summary_card.png   : Ranking summary")


if __name__ == "__main__":
    main()

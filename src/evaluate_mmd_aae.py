#!/usr/bin/env python
"""
evaluate_mmd_aae.py - 量化评估 MMD-AAE 域对齐效果

指标说明:
  1. Domain Classification Acc: 训练一个分类器判断 z 来自哪个域
     - 33.3% = 完美对齐 (随机猜测)
     - 100% = 完全分离
  2. Silhouette Score (by domain): 衡量域内紧凑度 vs 域间分离度
     - +1 = 完全分离
     - 0 = 随机混合 (我们的目标)
     - -1 = 完全交叉
  3. MMD 值: 域间分布差异的直接度量
     - 0 = 完美对齐
  4. CORAL Distance: 二阶统计量对齐度量
  5. Reconstruction MSE: 重建质量 (不应太差)

使用方法:
    cd ~/state/src

    # 评估单个实验
    python evaluate_mmd_aae.py --exp_name v2_m10_ep50

    # 对比多个实验
    python evaluate_mmd_aae.py --compare v2_m10_ep50 v2_m20_ep50 v2_r05_m20_ep50

    # 列出所有实验并全部评估
    python evaluate_mmd_aae.py --all
"""

import os
import sys
import argparse
import json
import torch
import torch.nn as nn
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# 配置
# ============================================================================
BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/mmd_aae"
OUTPUT_DIR = f"{BASE_DIR}/evaluations"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DOMAIN_CONFIGS = [
    {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
    {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
    {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser(description='MMD-AAE 量化评估')
    parser.add_argument('--exp_name', type=str, default=None, help='单个实验名')
    parser.add_argument('--compare', type=str, nargs='+', help='对比多个实验')
    parser.add_argument('--all', action='store_true', help='评估所有实验')
    parser.add_argument('--samples', type=int, default=2000, help='每域采样数')
    parser.add_argument('--checkpoint_file', type=str, default='best_model.pt')
    return parser.parse_args()


# ============================================================================
# 模型 (自动架构检测)
# ============================================================================
def detect_architecture(state_dict):
    info = {'use_layernorm': False, 'input_dim': None, 'latent_dim': None,
            'hidden_dim': None, 'mid_dim': None}
    
    # 统一 key: 去掉 net. 前缀
    # encoder.net.0.weight → encoder.0.weight
    # encoder.0.weight → encoder.0.weight (不变)
    normalized = {}
    for key in state_dict.keys():
        nk = key.replace('encoder.net.', 'encoder.').replace('decoder.net.', 'decoder.')
        normalized[nk] = key  # 映射到原始 key
    
    for nk, orig in normalized.items():
        if nk == 'encoder.0.weight':
            info['hidden_dim'] = state_dict[orig].shape[0]
            info['input_dim'] = state_dict[orig].shape[1]
        if nk == 'encoder.4.weight':
            info['mid_dim'] = state_dict[orig].shape[0]
        if nk == 'encoder.8.weight':
            info['latent_dim'] = state_dict[orig].shape[0]
    
    if info['latent_dim'] is None:
        for nk, orig in normalized.items():
            if nk == 'encoder.6.weight':
                info['latent_dim'] = state_dict[orig].shape[0]
    
    # 检测 BatchNorm vs LayerNorm
    for nk, orig in normalized.items():
        if 'encoder.1.' in nk and 'weight' in nk:
            bn_key_normalized = nk.replace('weight', 'running_mean')
            # 检查原始 key 空间中是否有 running_mean
            has_running_mean = any('running_mean' in k and ('encoder.net.1.' in k or 'encoder.1.' in k) 
                                  for k in state_dict.keys())
            info['use_layernorm'] = not has_running_mean
            break
    
    return info


def build_model_from_checkpoint(state_dict, info):
    NormLayer = nn.LayerNorm if info['use_layernorm'] else nn.BatchNorm1d
    mid = info.get('mid_dim') or info['hidden_dim'] // 2
    
    encoder = nn.Sequential(
        nn.Linear(info['input_dim'], info['hidden_dim']),
        NormLayer(info['hidden_dim']),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(info['hidden_dim'], mid),
        NormLayer(mid),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(mid, info['latent_dim']),
    )
    
    decoder = nn.Sequential(
        nn.Linear(info['latent_dim'], mid),
        NormLayer(mid),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(mid, info['hidden_dim']),
        NormLayer(info['hidden_dim']),
        nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(info['hidden_dim'], info['input_dim']),
    )
    
    model = nn.ModuleDict({'encoder': encoder, 'decoder': decoder})
    
    # 处理 key 前缀: encoder.net.0.weight → 0.weight
    enc_state = {}
    for k, v in state_dict.items():
        if k.startswith('encoder.'):
            new_key = k.replace('encoder.net.', '').replace('encoder.', '')
            enc_state[new_key] = v
    
    dec_state = {}
    for k, v in state_dict.items():
        if k.startswith('decoder.'):
            new_key = k.replace('decoder.net.', '').replace('decoder.', '')
            dec_state[new_key] = v
    
    model['encoder'].load_state_dict(enc_state)
    model['decoder'].load_state_dict(dec_state)
    
    return model


class SimpleH5Dataset(Dataset):
    def __init__(self, h5_path, max_samples=None, use_normalize=True,
                 global_mean=None, global_std=None):
        self.h5_path = h5_path
        self.use_normalize = use_normalize
        self.global_mean = global_mean
        self.global_std = global_std
        with h5py.File(h5_path, 'r') as f:
            total = f['X'].shape[0]
            self.num_samples = min(total, max_samples) if max_samples else total
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
            if self.global_mean is not None:
                # v3: z-score 标准化
                counts = (counts - self.global_mean) / self.global_std
            else:
                # v2: L2 归一化
                norm = counts.norm(p=2)
                if norm > 0:
                    counts = counts / norm
        return counts


# ============================================================================
# 量化指标
# ============================================================================

def metric_domain_classification(z_all, domains_all, n_domains=3):
    """
    训练一个简单 MLP 分类器判断 z 的域标签
    返回准确率 (33% = 完美对齐, 100% = 完全分离)
    """
    from sklearn.model_selection import cross_val_score
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    
    scaler = StandardScaler()
    z_scaled = scaler.fit_transform(z_all)
    
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42,
                        early_stopping=True, validation_fraction=0.15)
    scores = cross_val_score(clf, z_scaled, domains_all, cv=5, scoring='accuracy')
    
    return {
        'domain_cls_acc': float(scores.mean()),
        'domain_cls_std': float(scores.std()),
        'domain_cls_random': 1.0 / n_domains,
    }


def metric_silhouette(z_all, domains_all):
    """
    Silhouette Score: 衡量域聚类的紧凑/分离程度
    +1 = 完全按域分离, 0 = 随机混合, -1 = 交叉
    """
    from sklearn.metrics import silhouette_score
    
    # 在大数据上随机子采样以加速
    n = len(z_all)
    if n > 5000:
        idx = np.random.choice(n, 5000, replace=False)
        z_sub, d_sub = z_all[idx], domains_all[idx]
    else:
        z_sub, d_sub = z_all, domains_all
    
    score = silhouette_score(z_sub, d_sub, metric='euclidean')
    return {'silhouette_score': float(score)}


def metric_mmd(z_all, domains_all, n_domains=3):
    """
    计算所有域对之间的 MMD 值
    """
    def rbf_mmd(x, y, sigmas=[0.1, 1.0, 10.0]):
        xx = np.dot(x, x.T)
        yy = np.dot(y, y.T)
        xy = np.dot(x, y.T)
        
        rx = np.diag(xx)[:, None] * np.ones((1, len(x)))
        ry = np.diag(yy)[:, None] * np.ones((1, len(y)))
        
        dxx = rx.T + rx - 2 * xx
        dyy = ry.T + ry - 2 * yy
        dxy = rx.T + np.ones((len(x), 1)) @ ry.T[0:1, :] - 2 * xy
        
        mmd = 0.0
        for sigma in sigmas:
            gamma = 1.0 / (2 * sigma ** 2)
            mmd += np.exp(-gamma * dxx).mean() + np.exp(-gamma * dyy).mean() - 2 * np.exp(-gamma * dxy).mean()
        return mmd / len(sigmas)
    
    results = {}
    domain_names = [d['name'] for d in DOMAIN_CONFIGS[:n_domains]]
    mmd_values = []
    
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            zi = z_all[domains_all == i][:1000]  # cap for speed
            zj = z_all[domains_all == j][:1000]
            if len(zi) > 10 and len(zj) > 10:
                mmd_val = rbf_mmd(zi, zj)
                pair_name = f"{domain_names[i]}_vs_{domain_names[j]}"
                results[f'mmd_{pair_name}'] = float(mmd_val)
                mmd_values.append(mmd_val)
    
    results['mmd_mean'] = float(np.mean(mmd_values)) if mmd_values else 0.0
    return results


def metric_coral(z_all, domains_all, n_domains=3):
    """
    CORAL Distance: 二阶统计量 (协方差矩阵) 差异
    """
    domain_names = [d['name'] for d in DOMAIN_CONFIGS[:n_domains]]
    results = {}
    coral_values = []
    
    for i in range(n_domains):
        for j in range(i + 1, n_domains):
            zi = z_all[domains_all == i]
            zj = z_all[domains_all == j]
            if len(zi) > 10 and len(zj) > 10:
                ci = np.cov(zi, rowvar=False)
                cj = np.cov(zj, rowvar=False)
                diff = ci - cj
                coral_dist = np.sqrt((diff ** 2).sum()) / (4 * zi.shape[1] ** 2)
                pair_name = f"{domain_names[i]}_vs_{domain_names[j]}"
                results[f'coral_{pair_name}'] = float(coral_dist)
                coral_values.append(coral_dist)
    
    results['coral_mean'] = float(np.mean(coral_values)) if coral_values else 0.0
    return results


def metric_reconstruction(model, datasets, use_normalize, device):
    """
    重建 MSE: 衡量信息保留质量
    """
    model.eval()
    total_mse = 0.0
    total_count = 0
    
    for ds in datasets:
        loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=0)
        n_batches = 0
        for batch in loader:
            x = batch.to(device)
            with torch.no_grad():
                z = model['encoder'](x)
                x_recon = model['decoder'](z)
            mse = ((x - x_recon) ** 2).mean().item()
            total_mse += mse
            n_batches += 1
            if n_batches >= 10:  # 只评估前 10 个 batch
                break
        total_count += n_batches
    
    return {'recon_mse': total_mse / max(total_count, 1)}


# ============================================================================
# 主评估流程
# ============================================================================

def evaluate_experiment(exp_name, args):
    """评估单个实验，返回所有指标"""
    
    checkpoint_path = os.path.join(CHECKPOINT_DIR, exp_name, args.checkpoint_file)
    
    if not os.path.exists(checkpoint_path):
        print(f"  ❌ 跳过 {exp_name}: {checkpoint_path} 不存在")
        return None
    
    print(f"\n{'='*60}")
    print(f"评估实验: {exp_name}")
    print(f"{'='*60}")
    
    # 加载模型
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint['model_state_dict']
    info = detect_architecture(state_dict)
    
    version = checkpoint.get('version', 'v1')
    use_normalize = info['use_layernorm'] or version in ('v2', 'v3')
    
    # v3 保存了 global_mean/std
    global_mean = checkpoint.get('global_mean', None)
    global_std = checkpoint.get('global_std', None)
    if global_mean is not None:
        global_mean = global_mean.cpu()
        global_std = global_std.cpu()
    
    print(f"  Version: {version}")
    print(f"  Epoch: {checkpoint.get('epoch', '?')}")
    print(f"  Latent dim: {info['latent_dim']}")
    print(f"  Norm: {'LayerNorm' if info['use_layernorm'] else 'BatchNorm'}")
    if global_mean is not None:
        print(f"  Input norm: z-score (global mean/std)")
    
    if 'lambdas' in checkpoint:
        l = checkpoint['lambdas']
        print(f"  Lambda: r={l.get('recon')}, m={l.get('mmd')}, a={l.get('adv')}")
    
    try:
        model = build_model_from_checkpoint(state_dict, info)
        model = model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"  ❌ 模型加载失败: {e}")
        return None
    
    # 提取隐空间
    print(f"\n  提取隐空间 (每域 {args.samples} 样本)...")
    
    all_z = []
    all_domains = []
    datasets = []
    
    for domain_id, domain in enumerate(DOMAIN_CONFIGS):
        if not os.path.exists(domain['path']):
            print(f"  ⚠️ 跳过 {domain['name']}: 文件不存在")
            continue
        
        ds = SimpleH5Dataset(domain['path'], max_samples=args.samples,
                             use_normalize=use_normalize,
                             global_mean=global_mean, global_std=global_std)
        datasets.append(ds)
        loader = DataLoader(ds, batch_size=256, shuffle=True, num_workers=0)
        
        z_list = []
        count = 0
        with torch.no_grad():
            for batch in loader:
                x = batch.to(DEVICE)
                z = model['encoder'](x)
                z_list.append(z.cpu().numpy())
                count += x.size(0)
                if count >= args.samples:
                    break
        
        z_domain = np.concatenate(z_list)[:args.samples]
        all_z.append(z_domain)
        all_domains.extend([domain_id] * len(z_domain))
        print(f"    {domain['name']}: {len(z_domain)} 样本")
    
    if not all_z:
        print("  ❌ 无数据")
        return None
    
    z_all = np.concatenate(all_z)
    d_all = np.array(all_domains)
    n_domains = len(set(d_all))
    
    print(f"  总计: {len(z_all)} 样本, latent_dim={z_all.shape[1]}")
    
    # 计算所有指标
    print("\n  计算指标...")
    
    results = {
        'exp_name': exp_name,
        'version': version,
        'epoch': checkpoint.get('epoch', -1),
        'latent_dim': info['latent_dim'],
        'norm_type': 'LayerNorm' if info['use_layernorm'] else 'BatchNorm',
        'lambdas': checkpoint.get('lambdas', {}),
        'total_samples': len(z_all),
    }
    
    # 1. Domain Classification
    print("    [1/5] Domain Classification Accuracy...")
    results.update(metric_domain_classification(z_all, d_all, n_domains))
    print(f"         Acc = {results['domain_cls_acc']:.4f} ± {results['domain_cls_std']:.4f} (random={results['domain_cls_random']:.3f})")
    
    # 2. Silhouette Score
    print("    [2/5] Silhouette Score...")
    results.update(metric_silhouette(z_all, d_all))
    print(f"         Score = {results['silhouette_score']:.4f} (0 = 完美混合)")
    
    # 3. MMD
    print("    [3/5] MMD Values...")
    results.update(metric_mmd(z_all, d_all, n_domains))
    print(f"         Mean MMD = {results['mmd_mean']:.6f}")
    
    # 4. CORAL
    print("    [4/5] CORAL Distance...")
    results.update(metric_coral(z_all, d_all, n_domains))
    print(f"         Mean CORAL = {results['coral_mean']:.6f}")
    
    # 5. Reconstruction MSE
    print("    [5/5] Reconstruction MSE...")
    results.update(metric_reconstruction(model, datasets, use_normalize, DEVICE))
    print(f"         MSE = {results['recon_mse']:.6f}")
    
    return results


def print_comparison_table(all_results):
    """打印多实验对比表格"""
    
    if not all_results:
        print("\n没有可对比的结果")
        return
    
    print("\n")
    print("=" * 120)
    print("📊 实验对比汇总")
    print("=" * 120)
    
    # 表头
    headers = ['实验名', 'Version', 'Epoch', 'Latent', 'λ_mmd',
               'DomainClsAcc↓', 'Silhouette↓', 'MMD↓', 'CORAL↓', 'ReconMSE']
    
    col_widths = [25, 8, 6, 7, 7, 15, 13, 12, 12, 12]
    
    header_line = ""
    for h, w in zip(headers, col_widths):
        header_line += f"{h:<{w}}"
    print(header_line)
    print("-" * 120)
    
    # 数据行
    for r in sorted(all_results, key=lambda x: x.get('domain_cls_acc', 1.0)):
        lambdas = r.get('lambdas', {})
        line = (
            f"{r['exp_name']:<25}"
            f"{r.get('version', '?'):<8}"
            f"{r.get('epoch', '?'):<6}"
            f"{r.get('latent_dim', '?'):<7}"
            f"{lambdas.get('mmd', '?'):<7}"
            f"{r.get('domain_cls_acc', 0):<15.4f}"
            f"{r.get('silhouette_score', 0):<13.4f}"
            f"{r.get('mmd_mean', 0):<12.6f}"
            f"{r.get('coral_mean', 0):<12.6f}"
            f"{r.get('recon_mse', 0):<12.6f}"
        )
        print(line)
    
    print("-" * 120)
    
    # 指标解读
    print("\n📖 指标解读:")
    print("  DomainClsAcc ↓ : 域分类准确率, 越低越好 (33.3% = 完美对齐, 100% = 完全分离)")
    print("  Silhouette  ↓ : 轮廓系数, 越接近 0 越好 (0 = 混合, +1 = 分离)")
    print("  MMD         ↓ : 分布差异, 越低越好 (0 = 完美对齐)")
    print("  CORAL       ↓ : 协方差差异, 越低越好")
    print("  ReconMSE      : 重建误差, 不应太大 (信息保留)")
    
    # 找最佳实验
    best = min(all_results, key=lambda x: x.get('domain_cls_acc', 1.0))
    print(f"\n🏆 最佳域对齐: {best['exp_name']} (DomainClsAcc = {best['domain_cls_acc']:.4f})")


def main():
    args = parse_args()
    
    print("=" * 60)
    print("MMD-AAE 量化评估")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {DEVICE}")
    print("=" * 60)
    
    # 确定要评估的实验列表
    exp_names = []
    
    if args.all:
        if os.path.exists(CHECKPOINT_DIR):
            exp_names = sorted([d for d in os.listdir(CHECKPOINT_DIR) 
                               if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
        print(f"\n发现 {len(exp_names)} 个实验")
    elif args.compare:
        exp_names = args.compare
    elif args.exp_name:
        exp_names = [args.exp_name]
    else:
        # 默认评估所有
        if os.path.exists(CHECKPOINT_DIR):
            exp_names = sorted([d for d in os.listdir(CHECKPOINT_DIR) 
                               if os.path.isdir(os.path.join(CHECKPOINT_DIR, d))])
        if exp_names:
            print(f"\n未指定实验，自动评估所有 ({len(exp_names)} 个)")
        else:
            print("❌ 未找到任何实验。请使用 --exp_name 指定。")
            return
    
    # 逐个评估
    all_results = []
    for exp_name in exp_names:
        result = evaluate_experiment(exp_name, args)
        if result is not None:
            all_results.append(result)
    
    # 打印对比表
    if len(all_results) > 0:
        print_comparison_table(all_results)
    
    # 保存结果
    output_path = os.path.join(OUTPUT_DIR, f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 结果已保存: {output_path}")


if __name__ == "__main__":
    main()

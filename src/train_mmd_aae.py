#!/usr/bin/env python
"""
train_mmd_aae.py - MMD-AAE 训练脚本 v3 (训练坍缩修复版)

修复历史:
  v1: 初版 (BatchNorm, latent_dim=512, 单核MMD)
  v2: LayerNorm, log1p+L2, latent_dim=64, median heuristic
  v3: ★ 修复训练坍缩:
      - L2 归一化导致值太小 (~1/√18080≈0.007), MSE 无梯度信号
      - 改用 log1p + 全局标准化 (z-score) 替代 L2
      - 增加学习率 warmup 防止早期坍缩
      - 增加诊断输出 (z 统计量, 梯度范数)

使用方法:
    cd ~/state/src
    python train_mmd_aae.py
    python train_mmd_aae.py --lambda_mmd 10.0 --epochs 50 --exp_name v3_test
"""

import os
import sys
import h5py
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)


# ============================================================================
# 参数解析
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='MMD-AAE Training v3')
    
    parser.add_argument('--exp_name', type=str, default=None)
    
    # Lambda
    parser.add_argument('--lambda_recon', type=float, default=1.0)
    parser.add_argument('--lambda_mmd', type=float, default=0.0,
                        help='MMD 权重. 设 0 则自动校准')
    parser.add_argument('--lambda_adv', type=float, default=0.1)
    parser.add_argument('--auto_lambda', action='store_true', default=True,
                        help='自动校准 lambda (默认开启). 用 --no_auto_lambda 关闭')
    parser.add_argument('--no_auto_lambda', dest='auto_lambda', action='store_false')
    parser.add_argument('--mmd_target_ratio', type=float, default=1.0,
                        help='MMD 相对于 Recon 的目标比例')
    
    # 训练阶段
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--recon_warmup', type=int, default=20,
                        help='纯重建预热 epoch 数 (先训练好 autoencoder)')
    parser.add_argument('--rampup_epochs', type=int, default=20,
                        help='MMD/ADV 权重线性上升的 epoch 数')
    parser.add_argument('--lambda_var', type=float, default=1.0,
                        help='z 方差正则化权重 (防止 encoder 坍缩)')
    
    # 模型
    parser.add_argument('--latent_dim', type=int, default=64)
    
    return parser.parse_args()


# ============================================================================
# Dataset
# ============================================================================
def compute_global_stats(h5_paths, n_sample=5000):
    """
    从所有域采样计算全局 mean/std，用于 z-score 标准化
    这比 L2 归一化好得多: L2 把 18080 维压到 ||x||=1，每个分量 ~0.007，MSE 几乎无梯度
    """
    log.info("计算全局归一化统计量 (log1p → z-score)...")
    all_data = []
    for path in h5_paths:
        with h5py.File(path, 'r') as f:
            n = f['X'].shape[0]
            idx = np.random.choice(n, min(n_sample, n), replace=False)
            idx.sort()
            data = f['X'][idx]
            all_data.append(data)
    
    combined = np.concatenate(all_data, axis=0).astype(np.float32)
    combined = np.log1p(combined)
    
    mean = combined.mean(axis=0)
    std = combined.std(axis=0)
    std[std < 1e-6] = 1.0  # 避免除零
    
    log.info(f"  采样 {len(combined)} 个细胞")
    log.info(f"  log1p 后: mean={mean.mean():.4f}, std={std.mean():.4f}")
    log.info(f"  每个基因值范围: [{mean.min():.3f}, {mean.max():.3f}]")
    
    return torch.tensor(mean, dtype=torch.float32), torch.tensor(std, dtype=torch.float32)


class SimpleH5Dataset(Dataset):
    def __init__(self, h5_path, domain_id=0, domain_name="unknown",
                 global_mean=None, global_std=None):
        self.h5_path = h5_path
        self.domain_id = domain_id
        self.domain_name = domain_name
        self.global_mean = global_mean
        self.global_std = global_std
        
        with h5py.File(h5_path, 'r') as f:
            self.shape = f['X'].shape
            self.num_cells = self.shape[0]
            self.num_genes = self.shape[1]
        
        self._file = None
    
    def _get_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, 'r')
        return self._file
    
    def __len__(self):
        return self.num_cells
    
    def __getitem__(self, idx):
        f = self._get_file()
        counts = torch.tensor(f['X'][idx], dtype=torch.float32)
        
        # ★ v3 修复: log1p + z-score 标准化 (替代 L2 归一化)
        counts = torch.log1p(counts)
        if self.global_mean is not None:
            counts = (counts - self.global_mean) / self.global_std
        
        return counts, self.domain_id
    
    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None


def collate_fn(batch):
    counts = torch.stack([item[0] for item in batch])
    domains = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return counts, domains


class ParallelZipLoader:
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


# ============================================================================
# MMD (median heuristic)
# ============================================================================
def compute_mmd_multi_kernel(x, y):
    """
    多核 MMD + median heuristic 自动选择 sigma
    """
    # 计算点积
    xx = torch.mm(x, x.t())
    yy = torch.mm(y, y.t())
    xy = torch.mm(x, y.t())
    
    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)
    
    dxx = rx.t() + rx - 2. * xx
    dyy = ry.t() + ry - 2. * yy
    dxy = rx.t().expand(x.size(0), y.size(0)) + ry.expand(x.size(0), y.size(0)) - 2. * xy
    
    # ★ 修复 4: median heuristic 自动选择 sigma
    all_dists = torch.cat([dxx.reshape(-1), dyy.reshape(-1), dxy.reshape(-1)])
    median_dist = all_dists.median().detach()
    median_dist = torch.clamp(median_dist, min=1e-6)
    
    # 使用 median 的不同倍率作为 sigma
    sigmas = [median_dist * f for f in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0]]
    
    mmd = x.new_zeros(1)
    
    for sigma in sigmas:
        gamma = 1.0 / (2 * sigma)
        XX = torch.exp(-gamma * dxx)
        YY = torch.exp(-gamma * dyy)
        XY = torch.exp(-gamma * dxy)
        mmd = mmd + XX.mean() + YY.mean() - 2. * XY.mean()
    
    return mmd / len(sigmas)


# ============================================================================
# 模型 (★ 修复 1: LayerNorm 替换 BatchNorm)
# ============================================================================
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=64, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),        # ★ LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),               # ★ LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, latent_dim),      # ★ 更小的 latent_dim
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim=1024, output_dim=18080, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.LayerNorm(256),               # ★ LayerNorm
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden_dim),
            nn.LayerNorm(hidden_dim),        # ★ LayerNorm
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


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class MMD_AAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=64, num_domains=3, dropout=0.2):
        super().__init__()
        
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout)
        self.discriminator = DomainDiscriminator(latent_dim, hidden_dim // 4, num_domains)
        
        self.num_domains = num_domains
        self.latent_dim = latent_dim
    
    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        domain_logits = self.discriminator(z)
        return x_recon, z, domain_logits
    
    def compute_loss(self, x, domain_labels, weight_recon=1.0, weight_mmd=10.0, weight_adv=0.5,
                     weight_var=1.0, grl_alpha=1.0):
        x_recon, z, _ = self.forward(x)
        
        # 1. 重建损失
        recon_loss = nn.functional.mse_loss(x_recon, x)
        
        # 2. MMD 损失
        mmd_losses = []
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                mask_i = domain_labels == i
                mask_j = domain_labels == j
                if mask_i.sum() > 1 and mask_j.sum() > 1:
                    mmd_ij = compute_mmd_multi_kernel(z[mask_i], z[mask_j])
                    mmd_losses.append(mmd_ij)
        
        if len(mmd_losses) > 0:
            mmd_loss = torch.stack(mmd_losses).mean()
        else:
            mmd_loss = z.new_zeros(1).squeeze()
        
        # 3. 对抗损失
        z_grl = GradientReversalFunction.apply(z, grl_alpha)
        domain_logits = self.discriminator(z_grl)
        adv_loss = nn.functional.cross_entropy(domain_logits, domain_labels)
        
        # 4. ★ z 正则化: 匹配 z → N(0, 1)
        #    类似 VAE 的 KL 约束，防止坍缩 (var→0) 和爆炸 (var→∞)
        z_mean = z.mean(dim=0)
        z_var = z.var(dim=0)
        # KL(q||p) where q=N(z_mean, z_var), p=N(0,1)
        var_loss = 0.5 * (z_var + z_mean**2 - 1 - torch.log(z_var + 1e-8)).mean()
        
        # 总损失
        total_loss = (weight_recon * recon_loss + weight_mmd * mmd_loss + 
                      weight_adv * adv_loss + weight_var * var_loss)
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'mmd': mmd_loss,
            'adv': adv_loss,
            'var': var_loss,
            'z_var': z_var.mean(),
        }


# ============================================================================
# 训练
# ============================================================================
def train_epoch(model, train_loader, optimizer, device, epoch, lambdas, log_interval=50):
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_mmd = 0
    total_adv = 0
    total_var = 0
    num_batches = 0
    
    for batch_idx, domain_batches in enumerate(train_loader):
        all_counts = []
        all_domains = []
        for domain_id, (counts, domains) in enumerate(domain_batches):
            all_counts.append(counts)
            all_domains.append(torch.full((counts.size(0),), domain_id, dtype=torch.long))
        
        x = torch.cat(all_counts, dim=0).to(device)
        domain_labels = torch.cat(all_domains, dim=0).to(device)
        
        perm = torch.randperm(x.size(0))
        x = x[perm]
        domain_labels = domain_labels[perm]
        
        optimizer.zero_grad()
        
        losses = model.compute_loss(
            x, domain_labels,
            weight_recon=lambdas['recon'],
            weight_mmd=lambdas['mmd'],
            weight_adv=lambdas['adv'],
            weight_var=lambdas.get('var', 0.0),
        )
        losses['total'].backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        total_recon += losses['recon'].item()
        total_mmd += losses['mmd'].item()
        total_adv += losses['adv'].item()
        total_var += losses.get('z_var', torch.tensor(0.0)).item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            phase_str = ""
            if lambdas['mmd'] == 0 and lambdas['adv'] == 0:
                phase_str = "[WARMUP] "
            log.info(
                f"{phase_str}Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {losses['total'].item():.8f} "
                f"(Recon: {losses['recon'].item():.8f}, "
                f"MMD: {losses['mmd'].item():.8f}, "
                f"Adv: {losses['adv'].item():.8f}, "
                f"z_var: {losses.get('z_var', torch.tensor(0.0)).item():.6f})"
            )
    
    # ★ 每个 epoch 输出诊断信息
    model.eval()
    with torch.no_grad():
        z = model.encoder(x)
        z_np = z.cpu().numpy()
        per_dim_std = z_np.std(axis=0)
        log.info(f"  [诊断] z 统计: mean={z_np.mean():.4f}, std={z_np.std():.4f}, "
                 f"min={z_np.min():.4f}, max={z_np.max():.4f}")
        log.info(f"  [诊断] z 各维度 std: mean={per_dim_std.mean():.6f}, "
                 f"min={per_dim_std.min():.6f}, max={per_dim_std.max():.6f}")
        x_np = x.cpu().numpy()
        log.info(f"  [诊断] x 输入: mean={x_np.mean():.4f}, std={x_np.std():.4f}")
    
    # 梯度范数
    total_grad = 0
    n_params = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad += p.grad.norm().item()
            n_params += 1
    if n_params > 0:
        log.info(f"  [诊断] 平均梯度范数: {total_grad / n_params:.6f}")
    
    return {
        'loss': total_loss / num_batches,
        'recon': total_recon / num_batches,
        'mmd': total_mmd / num_batches,
        'adv': total_adv / num_batches,
        'z_var': total_var / num_batches,
    }


def main():
    args = parse_args()
    
    BASE_DIR = "/media/mldadmin/home/s125mdg34_03/state"
    
    LAMBDAS = {
        'recon': args.lambda_recon,
        'mmd': args.lambda_mmd,
        'adv': args.lambda_adv,
    }
    
    if args.exp_name:
        EXP_NAME = args.exp_name
    else:
        EXP_NAME = f"v3_r{LAMBDAS['recon']}_m{LAMBDAS['mmd']}_a{LAMBDAS['adv']}"
    
    CHECKPOINT_DIR = f"{BASE_DIR}/checkpoints/mmd_aae/{EXP_NAME}"
    
    DOMAIN_CONFIGS = [
        {"name": "K562",   "path": f"{BASE_DIR}/competition_support_set/k562.h5"},
        {"name": "RPE1",   "path": f"{BASE_DIR}/competition_support_set/rpe1.h5"},
        {"name": "Jurkat", "path": f"{BASE_DIR}/competition_support_set/jurkat.h5"},
    ]
    
    BATCH_SIZE = args.batch_size
    HIDDEN_DIM = 1024
    LATENT_DIM = args.latent_dim
    LEARNING_RATE = args.lr
    NUM_EPOCHS = args.epochs
    NUM_WORKERS = 2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    # 显示配置
    log.info("=" * 60)
    log.info("MMD-AAE 训练 v3 (3阶段训练)")
    log.info("=" * 60)
    log.info(f"实验名称: {EXP_NAME}")
    log.info(f"检查点目录: {CHECKPOINT_DIR}")
    log.info("-" * 60)
    log.info(">>> 修复内容 <<<")
    log.info("  [1] BatchNorm → LayerNorm")
    log.info("  [2] 输入 log1p + z-score 归一化")
    log.info(f"  [3] latent_dim = {LATENT_DIM} (原 512)")
    log.info("  [4] MMD sigma = median heuristic")
    log.info("-" * 60)
    log.info(">>> Lambda 权重 <<<")
    log.info(f"  λ_recon = {LAMBDAS['recon']}")
    log.info(f"  λ_mmd   = {LAMBDAS['mmd']}")
    log.info(f"  λ_adv   = {LAMBDAS['adv']}")
    log.info("-" * 60)
    log.info(f"Device: {DEVICE}")
    log.info(f"Batch Size: {BATCH_SIZE} x 3 = {BATCH_SIZE * 3}")
    log.info(f"Learning Rate: {LEARNING_RATE}")
    log.info(f"Epochs: {NUM_EPOCHS}")
    
    # ★ v3: 先计算全局 mean/std
    all_h5_paths = [d['path'] for d in DOMAIN_CONFIGS]
    global_mean, global_std = compute_global_stats(all_h5_paths)
    
    # DataLoader
    log.info("\n创建 DataLoader...")
    
    loaders = []
    datasets = []
    domain_names = []
    input_dim = None
    
    for i, domain in enumerate(DOMAIN_CONFIGS):
        dataset = SimpleH5Dataset(
            domain["path"], domain_id=i, domain_name=domain["name"],
            global_mean=global_mean, global_std=global_std
        )
        datasets.append(dataset)
        
        if input_dim is None:
            input_dim = dataset.num_genes
        
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=NUM_WORKERS,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True if NUM_WORKERS > 0 else False,
        )
        loaders.append(loader)
        domain_names.append(domain["name"])
        log.info(f"  {domain['name']}: {dataset.num_cells} cells")
    
    train_loader = ParallelZipLoader(loaders, domain_names)
    log.info(f"总迭代数/epoch: {len(train_loader)}")
    
    # 模型
    log.info("\n创建模型...")
    model = MMD_AAE(
        input_dim=input_dim,
        hidden_dim=HIDDEN_DIM,
        latent_dim=LATENT_DIM,
        num_domains=len(DOMAIN_CONFIGS),
    ).to(DEVICE)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"模型参数量: {num_params:,}")
    
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10,
                                                      min_lr=1e-5)
    
    # ============================================================
    # Phase 1: 纯重建预热 (训练好 autoencoder)
    # ============================================================
    recon_warmup = args.recon_warmup
    rampup_epochs = args.rampup_epochs
    target_lambda_mmd = LAMBDAS['mmd']  # 可能是0 (待自动校准) 或用户指定值
    
    log.info("\n" + "=" * 60)
    log.info(f"Phase 1: 纯重建预热 ({recon_warmup} epochs)")
    log.info(f"  只训练 Recon + z方差正则化, 不加 MMD/ADV")
    log.info("=" * 60)
    
    best_loss = float('inf')
    
    warmup_lambdas = {
        'recon': LAMBDAS['recon'],
        'mmd': 0.0,
        'adv': 0.0,
        'var': args.lambda_var,
    }
    
    for epoch in range(1, recon_warmup + 1):
        log.info(f"\n=== [Phase 1] Epoch {epoch}/{recon_warmup} ===")
        
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, epoch, warmup_lambdas)
        
        log.info(
            f"Epoch {epoch} 完成: "
            f"Loss={train_metrics['loss']:.8f}, "
            f"Recon={train_metrics['recon']:.8f}, "
            f"z_var={train_metrics['z_var']:.6f}"
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"Learning Rate: {current_lr:.6f}")
    
    log.info(f"\n✅ Phase 1 完成! Recon={train_metrics['recon']:.8f}, z_var={train_metrics['z_var']:.6f}")
    
    # ============================================================
    # Phase 2: 自动 Lambda 校准 (在训练好的 AE 上测量)
    # ============================================================
    if args.auto_lambda and target_lambda_mmd == 0.0:
        log.info("\n" + "=" * 60)
        log.info("Phase 2: 自动 Lambda 校准")
        log.info("  在训练好的 autoencoder 上测量各 loss 量级")
        log.info("=" * 60)
        
        model.train()
        recon_sum, mmd_sum, adv_sum = 0.0, 0.0, 0.0
        n_cal = 0
        
        for batch_idx, domain_batches in enumerate(train_loader):
            all_counts, all_domains = [], []
            for domain_id, (counts, domains) in enumerate(domain_batches):
                all_counts.append(counts)
                all_domains.append(torch.full((counts.size(0),), domain_id, dtype=torch.long))
            
            x = torch.cat(all_counts, dim=0).to(DEVICE)
            domain_labels = torch.cat(all_domains, dim=0).to(DEVICE)
            
            with torch.no_grad():
                losses = model.compute_loss(x, domain_labels,
                                          weight_recon=1.0, weight_mmd=1.0, weight_adv=1.0, weight_var=0.0)
            
            recon_sum += losses['recon'].item()
            mmd_sum += losses['mmd'].item()
            adv_sum += losses['adv'].item()
            n_cal += 1
            
            if n_cal >= 100:
                break
        
        avg_recon = recon_sum / n_cal
        avg_mmd = mmd_sum / n_cal
        avg_adv = adv_sum / n_cal
        
        log.info(f"\n  原始量级 (在训练好的 AE 上):")
        log.info(f"    Recon = {avg_recon:.10f}")
        log.info(f"    MMD   = {avg_mmd:.10f}")
        log.info(f"    Adv   = {avg_adv:.10f}")
        
        if avg_mmd > 1e-15:
            target_lambda_mmd = (avg_recon / avg_mmd) * args.mmd_target_ratio
        else:
            target_lambda_mmd = 1000.0
            log.info("  ⚠️ MMD 仍然太小，使用默认 lambda_mmd=1000")
        
        LAMBDAS['mmd'] = target_lambda_mmd
        
        log.info(f"\n  ★ 自动校准 λ_mmd = {LAMBDAS['mmd']:.2f}")
        log.info(f"    校准后等效: Recon={avg_recon:.8f}, "
                 f"MMD_weighted={LAMBDAS['mmd'] * avg_mmd:.8f}")
        log.info("=" * 60)
    elif target_lambda_mmd == 0.0:
        LAMBDAS['mmd'] = 10.0
        target_lambda_mmd = 10.0
    
    # ============================================================
    # Phase 3: 域对齐 (线性 ramp-up MMD/ADV)
    # ============================================================
    align_epochs = NUM_EPOCHS - recon_warmup
    
    log.info("\n" + "=" * 60)
    log.info(f"Phase 3: 域对齐训练 ({align_epochs} epochs)")
    log.info(f"  λ_mmd: 0 → {target_lambda_mmd:.2f} (线性 ramp-up {rampup_epochs} epochs)")
    log.info(f"  λ_adv: 0 → {LAMBDAS['adv']}")
    log.info("=" * 60)
    
    for epoch in range(1, align_epochs + 1):
        global_epoch = recon_warmup + epoch
        
        # 线性 ramp-up
        if epoch <= rampup_epochs:
            ramp = epoch / rampup_epochs
        else:
            ramp = 1.0
        
        epoch_lambdas = {
            'recon': LAMBDAS['recon'],
            'mmd': target_lambda_mmd * ramp,
            'adv': LAMBDAS['adv'] * ramp,
            'var': args.lambda_var,
        }
        
        log.info(f"\n=== [Phase 3] Epoch {global_epoch}/{NUM_EPOCHS} (ramp={ramp:.2f}) ===")
        log.info(f"  λ: recon={epoch_lambdas['recon']}, mmd={epoch_lambdas['mmd']:.2f}, "
                 f"adv={epoch_lambdas['adv']:.3f}, var={epoch_lambdas['var']}")
        
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, global_epoch, epoch_lambdas)
        
        log.info(
            f"Epoch {global_epoch} 完成: "
            f"Loss={train_metrics['loss']:.8f}, "
            f"Recon={train_metrics['recon']:.8f}, "
            f"MMD={train_metrics['mmd']:.8f}, "
            f"Adv={train_metrics['adv']:.8f}, "
            f"z_var={train_metrics['z_var']:.6f}"
        )
        
        scheduler.step(train_metrics['recon'])
        current_lr = optimizer.param_groups[0]['lr']
        log.info(f"Learning Rate: {current_lr:.6f}")
        
        checkpoint_data = {
            'epoch': global_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_metrics['loss'],
            'lambdas': {**LAMBDAS, 'mmd': target_lambda_mmd},
            'exp_name': EXP_NAME,
            'latent_dim': LATENT_DIM,
            'version': 'v3',
            'global_mean': global_mean,
            'global_std': global_std,
        }
        
        if train_metrics['loss'] < best_loss:
            best_loss = train_metrics['loss']
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, 'best_model.pt'))
            log.info(f"保存最佳模型")
        
        if global_epoch % 20 == 0:
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f'model_epoch_{global_epoch}.pt'))
            log.info(f"保存检查点: epoch {global_epoch}")
    
    # 保存最终模型
    torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, 'final_model.pt'))
    
    log.info("\n" + "=" * 60)
    log.info("✅ 训练完成!")
    log.info(f"实验: {EXP_NAME}")
    log.info(f"Lambda: recon={LAMBDAS['recon']}, mmd={target_lambda_mmd:.2f}, adv={LAMBDAS['adv']}")
    log.info(f"最佳 Loss: {best_loss:.4f}")
    log.info(f"检查点: {CHECKPOINT_DIR}")
    log.info("=" * 60)
    
    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()


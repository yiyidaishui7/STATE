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
    parser.add_argument('--lambda_mmd', type=float, default=10.0)
    parser.add_argument('--lambda_adv', type=float, default=0.5)
    
    # 训练
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率 warmup 的 epoch 数')
    
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
    
    def compute_loss(self, x, domain_labels, weight_recon=1.0, weight_mmd=10.0, weight_adv=0.5, grl_alpha=1.0):
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
        
        # 总损失
        total_loss = weight_recon * recon_loss + weight_mmd * mmd_loss + weight_adv * adv_loss
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'mmd': mmd_loss,
            'adv': adv_loss,
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
            weight_adv=lambdas['adv']
        )
        losses['total'].backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        total_recon += losses['recon'].item()
        total_mmd += losses['mmd'].item()
        total_adv += losses['adv'].item()
        num_batches += 1
        
        if batch_idx % log_interval == 0:
            log.info(
                f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                f"Loss: {losses['total'].item():.4f} "
                f"(Recon: {losses['recon'].item():.4f}, "
                f"MMD: {losses['mmd'].item():.4f}, "
                f"Adv: {losses['adv'].item():.4f})"
            )
    
    # ★ v3: 每个 epoch 输出诊断信息
    model.eval()
    with torch.no_grad():
        # 取最后一个 batch 做诊断
        z = model.encoder(x)
        z_np = z.cpu().numpy()
        log.info(f"  [诊断] z 统计: mean={z_np.mean():.4f}, std={z_np.std():.4f}, "
                 f"min={z_np.min():.4f}, max={z_np.max():.4f}")
        log.info(f"  [诊断] z 各维度 std: mean={z_np.std(axis=0).mean():.4f}, "
                 f"min={z_np.std(axis=0).min():.4f}")
        # 检查 x 输入统计
        x_np = x.cpu().numpy()
        log.info(f"  [诊断] x 输入: mean={x_np.mean():.4f}, std={x_np.std():.4f}, "
                 f"max={x_np.max():.4f}")
    
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
        EXP_NAME = f"v2_r{LAMBDAS['recon']}_m{LAMBDAS['mmd']}_a{LAMBDAS['adv']}"
    
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
    log.info("MMD-AAE 训练 v2 (架构修复版)")
    log.info("=" * 60)
    log.info(f"实验名称: {EXP_NAME}")
    log.info(f"检查点目录: {CHECKPOINT_DIR}")
    log.info("-" * 60)
    log.info(">>> 修复内容 <<<")
    log.info("  [1] BatchNorm → LayerNorm")
    log.info("  [2] 输入 log1p + L2 归一化")
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
    
    # 优化器 (★ v3: 带 warmup 的学习率调度)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    warmup_epochs = args.warmup_epochs
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs  # 线性 warmup
        else:
            # cosine decay
            progress = (epoch - warmup_epochs) / max(NUM_EPOCHS - warmup_epochs, 1)
            return 0.5 * (1 + np.cos(np.pi * progress))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # 保存全局统计量到 checkpoint
    global_stats = {'mean': global_mean, 'std': global_std}
    
    # 训练
    log.info("\n开始训练...")
    log.info(f"学习率 warmup: {warmup_epochs} epochs")
    log.info("-" * 60)
    
    best_loss = float('inf')
    
    for epoch in range(1, NUM_EPOCHS + 1):
        log.info(f"\n=== Epoch {epoch}/{NUM_EPOCHS} ===")
        
        train_metrics = train_epoch(model, train_loader, optimizer, DEVICE, epoch, LAMBDAS)
        
        log.info(
            f"Epoch {epoch} 完成: "
            f"Loss={train_metrics['loss']:.4f}, "
            f"Recon={train_metrics['recon']:.4f}, "
            f"MMD={train_metrics['mmd']:.4f}, "
            f"Adv={train_metrics['adv']:.4f}"
        )
        
        scheduler.step()
        log.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_metrics['loss'],
            'lambdas': LAMBDAS,
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
        
        if epoch % 10 == 0:
            torch.save(checkpoint_data, os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch}.pt'))
            log.info(f"保存检查点: epoch {epoch}")
    
    log.info("\n" + "=" * 60)
    log.info("✅ 训练完成!")
    log.info(f"实验: {EXP_NAME}")
    log.info(f"Lambda: recon={LAMBDAS['recon']}, mmd={LAMBDAS['mmd']}, adv={LAMBDAS['adv']}")
    log.info(f"最佳 Loss: {best_loss:.4f}")
    log.info(f"检查点: {CHECKPOINT_DIR}")
    log.info("=" * 60)
    
    for ds in datasets:
        ds.close()


if __name__ == "__main__":
    main()

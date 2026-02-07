"""
MMD-AAE 模型定义
基于论文: "Domain Generalization with Adversarial Feature Learning"

架构:
- Encoder: 将基因表达映射到隐空间
- Decoder: 重建输入
- Domain Discriminator: 对抗性域分类器
- Task Classifier: 下游任务分类器 (可选)
- MMD Loss: 最小化域间分布差异
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_mmd(x, y, kernel='rbf', sigma=None):
    """
    计算 Maximum Mean Discrepancy (MMD)
    
    Args:
        x: (N, D) 第一个分布的样本
        y: (M, D) 第二个分布的样本
        kernel: 'rbf' 或 'linear'
        sigma: RBF 核的带宽参数
    
    Returns:
        MMD^2 值
    """
    if kernel == 'linear':
        return _mmd_linear(x, y)
    elif kernel == 'rbf':
        if sigma is None:
            # 使用中值启发式
            sigma = median_heuristic(torch.cat([x, y], dim=0))
        return _mmd_rbf(x, y, sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")


def _mmd_linear(x, y):
    """线性核 MMD"""
    xx = torch.mm(x, x.t()).mean()
    yy = torch.mm(y, y.t()).mean()
    xy = torch.mm(x, y.t()).mean()
    return xx + yy - 2 * xy


def _mmd_rbf(x, y, sigma):
    """RBF 核 MMD"""
    xx = _rbf_kernel(x, x, sigma)
    yy = _rbf_kernel(y, y, sigma)
    xy = _rbf_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2 * xy.mean()


def _rbf_kernel(x, y, sigma):
    """计算 RBF 核矩阵"""
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    
    x = x.unsqueeze(1)  # (N, 1, D)
    y = y.unsqueeze(0)  # (1, M, D)
    
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    
    kernel_matrix = torch.exp(-torch.sum((tiled_x - tiled_y) ** 2, dim=2) / (2 * sigma ** 2))
    return kernel_matrix


def median_heuristic(x):
    """使用中值启发式估计 sigma"""
    pairwise_distances = torch.cdist(x, x)
    median = torch.median(pairwise_distances[pairwise_distances > 0])
    return median


class Encoder(nn.Module):
    """编码器：将高维基因表达映射到低维隐空间"""
    
    def __init__(self, input_dim, hidden_dim=1024, latent_dim=512, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    """解码器：从隐空间重建输入"""
    
    def __init__(self, latent_dim, hidden_dim=1024, output_dim=18080, dropout=0.25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, z):
        return self.net(z)


class DomainDiscriminator(nn.Module):
    """域判别器：预测样本来自哪个域"""
    
    def __init__(self, latent_dim, hidden_dim=512, num_domains=3):
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


class GradientReversal(torch.autograd.Function):
    """梯度反转层：用于对抗训练"""
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class MMD_AAE(nn.Module):
    """
    MMD-AAE 完整模型
    
    组件:
    - encoder: 将输入映射到隐空间
    - decoder: 重建输入
    - discriminator: 域判别器 (对抗)
    
    损失:
    - reconstruction: 重建损失
    - mmd: MMD 损失 (域对齐)
    - adversarial: 对抗损失 (让 encoder 欺骗 discriminator)
    """
    
    def __init__(
        self,
        input_dim,
        hidden_dim=1024,
        latent_dim=512,
        num_domains=3,
        dropout=0.25,
        # 损失权重
        weight_recon=1.0,
        weight_mmd=1.0,
        weight_adv=0.1,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_domains = num_domains
        
        # 网络组件
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim, dropout)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim, dropout)
        self.discriminator = DomainDiscriminator(latent_dim, hidden_dim // 2, num_domains)
        
        # 损失权重
        self.weight_recon = weight_recon
        self.weight_mmd = weight_mmd
        self.weight_adv = weight_adv
    
    def encode(self, x):
        """编码"""
        return self.encoder(x)
    
    def decode(self, z):
        """解码"""
        return self.decoder(z)
    
    def forward(self, x):
        """前向传播"""
        z = self.encode(x)
        x_recon = self.decode(z)
        domain_logits = self.discriminator(z)
        return x_recon, z, domain_logits
    
    def compute_loss(self, batch_data, domain_labels, grl_alpha=1.0):
        """
        计算所有损失
        
        Args:
            batch_data: (N, input_dim) 合并后的 batch
            domain_labels: (N,) 域标签 [0, 1, 2]
            grl_alpha: 梯度反转强度
        
        Returns:
            loss_dict: 包含各个损失项的字典
        """
        # 前向传播
        x_recon, z, domain_logits = self.forward(batch_data)
        
        # 1. 重建损失
        recon_loss = F.mse_loss(x_recon, batch_data)
        
        # 2. MMD 损失 - 成对计算域间 MMD
        mmd_loss = torch.tensor(0.0, device=batch_data.device)
        for i in range(self.num_domains):
            for j in range(i + 1, self.num_domains):
                mask_i = domain_labels == i
                mask_j = domain_labels == j
                if mask_i.sum() > 0 and mask_j.sum() > 0:
                    mmd_loss = mmd_loss + compute_mmd(z[mask_i], z[mask_j], kernel='rbf')
        
        # 归一化 MMD (除以域对数)
        num_pairs = self.num_domains * (self.num_domains - 1) / 2
        mmd_loss = mmd_loss / num_pairs
        
        # 3. 对抗损失 - 让 encoder 欺骗 discriminator
        # 使用梯度反转：discriminator 要正确分类，encoder 要让它分错
        z_grl = GradientReversal.apply(z, grl_alpha)
        domain_logits_grl = self.discriminator(z_grl)
        adv_loss = F.cross_entropy(domain_logits_grl, domain_labels)
        
        # 4. Discriminator 损失 (仅更新 discriminator)
        disc_loss = F.cross_entropy(domain_logits.detach(), domain_labels)
        
        # 总损失
        total_loss = (
            self.weight_recon * recon_loss +
            self.weight_mmd * mmd_loss +
            self.weight_adv * adv_loss
        )
        
        return {
            'total': total_loss,
            'recon': recon_loss,
            'mmd': mmd_loss,
            'adv': adv_loss,
            'disc': disc_loss,
            'z': z,
        }


def count_parameters(model):
    """统计模型参数量"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # 测试模型
    print("测试 MMD-AAE 模型...")
    
    input_dim = 18080  # 基因数
    batch_size = 96    # 32 * 3 domains
    
    model = MMD_AAE(input_dim=input_dim)
    print(f"模型参数量: {count_parameters(model):,}")
    
    # 模拟输入
    x = torch.randn(batch_size, input_dim)
    domain_labels = torch.tensor([0]*32 + [1]*32 + [2]*32)
    
    # 计算损失
    losses = model.compute_loss(x, domain_labels)
    
    print(f"\n损失值:")
    for k, v in losses.items():
        if isinstance(v, torch.Tensor) and v.dim() == 0:
            print(f"  {k}: {v.item():.6f}")
    
    print("\n✓ 模型测试通过!")

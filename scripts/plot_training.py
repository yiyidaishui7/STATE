#!/usr/bin/env python3
"""
训练可视化脚本
绘制loss曲线和其他训练指标
"""

import os
import re
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

def parse_training_log(log_file):
    """
    解析训练日志文件，提取loss和其他指标
    """
    data = {
        'step': [],
        'train_loss': [],
        'val_loss': [],
        'epoch': [],
        'lr': []
    }
    
    with open(log_file, 'r') as f:
        for line in f:
            # 尝试匹配训练步数和loss
            # 格式可能是: Epoch X: step Y/Z | train_loss: X.XXX
            
            # 匹配step和loss
            step_match = re.search(r'step[:\s]+(\d+)', line, re.IGNORECASE)
            train_loss_match = re.search(r'train[_\s]?loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
            val_loss_match = re.search(r'val[_\s]?loss[:\s]+([0-9.]+)', line, re.IGNORECASE)
            epoch_match = re.search(r'epoch[:\s]+(\d+)', line, re.IGNORECASE)
            lr_match = re.search(r'lr[:\s]+([0-9.e-]+)', line, re.IGNORECASE)
            
            if step_match:
                current_step = int(step_match.group(1))
                
                # 如果找到了train_loss，记录
                if train_loss_match:
                    data['step'].append(current_step)
                    data['train_loss'].append(float(train_loss_match.group(1)))
                    data['val_loss'].append(None)
                    data['epoch'].append(int(epoch_match.group(1)) if epoch_match else None)
                    data['lr'].append(float(lr_match.group(1)) if lr_match else None)
                
                # 如果找到了val_loss，更新最后一条记录或添加新记录
                if val_loss_match:
                    val_loss_value = float(val_loss_match.group(1))
                    # 如果最后一条记录是这个step，更新val_loss
                    if data['step'] and data['step'][-1] == current_step:
                        data['val_loss'][-1] = val_loss_value
                    else:
                        # 否则添加新记录
                        data['step'].append(current_step)
                        data['train_loss'].append(None)
                        data['val_loss'].append(val_loss_value)
                        data['epoch'].append(int(epoch_match.group(1)) if epoch_match else None)
                        data['lr'].append(float(lr_match.group(1)) if lr_match else None)
    
    df = pd.DataFrame(data)
    
    # 去除空行
    df = df[df['train_loss'].notna() | df['val_loss'].notna()]
    
    return df

def parse_csv_log(csv_file):
    """
    解析CSV日志文件（如果存在）
    """
    try:
        df = pd.read_csv(csv_file)
        return df
    except Exception as e:
        print(f"无法读取CSV日志: {e}")
        return None

def plot_loss_curves(df, output_dir):
    """
    绘制loss曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：训练和验证loss
    if 'train_loss' in df.columns:
        train_data = df[df['train_loss'].notna()]
        ax1.plot(train_data['step'], train_data['train_loss'], 
                label='Train Loss', alpha=0.7, linewidth=1.5)
    
    if 'val_loss' in df.columns:
        val_data = df[df['val_loss'].notna()]
        ax1.plot(val_data['step'], val_data['val_loss'], 
                label='Validation Loss', alpha=0.8, linewidth=2, marker='o', markersize=3)
    
    ax1.set_xlabel('Step', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 右图：只显示验证loss（放大查看）
    if 'val_loss' in df.columns:
        val_data = df[df['val_loss'].notna()]
        ax2.plot(val_data['step'], val_data['val_loss'], 
                label='Validation Loss', alpha=0.8, linewidth=2, 
                marker='o', markersize=4, color='orange')
        ax2.set_xlabel('Step', fontsize=12)
        ax2.set_ylabel('Validation Loss', fontsize=12)
        ax2.set_title('Validation Loss (Detailed)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: loss_curves.png")
    plt.close()

def plot_loss_smoothed(df, output_dir, window=10):
    """
    绘制平滑的loss曲线
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    if 'train_loss' in df.columns:
        train_data = df[df['train_loss'].notna()].copy()
        # 计算移动平均
        train_data['train_loss_smooth'] = train_data['train_loss'].rolling(window=window, min_periods=1).mean()
        
        # 原始数据（半透明）
        ax.plot(train_data['step'], train_data['train_loss'], 
               alpha=0.2, linewidth=0.5, color='blue', label='Train Loss (raw)')
        # 平滑数据
        ax.plot(train_data['step'], train_data['train_loss_smooth'], 
               alpha=0.8, linewidth=2, color='blue', label=f'Train Loss (smooth, window={window})')
    
    if 'val_loss' in df.columns:
        val_data = df[df['val_loss'].notna()]
        ax.plot(val_data['step'], val_data['val_loss'], 
               alpha=0.9, linewidth=2, marker='o', markersize=4, 
               color='orange', label='Validation Loss')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training Progress (Smoothed)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'loss_curves_smoothed.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: loss_curves_smoothed.png")
    plt.close()

def plot_learning_rate(df, output_dir):
    """
    绘制学习率变化
    """
    if 'lr' not in df.columns or df['lr'].isna().all():
        print("⚠️  未找到学习率数据")
        return
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    lr_data = df[df['lr'].notna()]
    ax.plot(lr_data['step'], lr_data['lr'], linewidth=2, color='green')
    
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # 使用对数坐标
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'learning_rate.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: learning_rate.png")
    plt.close()

def plot_validation_metrics(df, output_dir):
    """
    绘制验证集上的各种指标
    """
    # 从dataframe中找出所有可能的指标列
    metric_cols = [col for col in df.columns if 'val_' in col.lower() and col != 'val_loss']
    
    if not metric_cols:
        print("⚠️  未找到其他验证指标")
        return
    
    n_metrics = len(metric_cols)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_metrics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for idx, col in enumerate(metric_cols):
        metric_data = df[df[col].notna()]
        if len(metric_data) > 0:
            axes[idx].plot(metric_data['step'], metric_data[col], 
                          linewidth=2, marker='o', markersize=4)
            axes[idx].set_xlabel('Step', fontsize=11)
            axes[idx].set_ylabel(col, fontsize=11)
            axes[idx].set_title(col, fontsize=12, fontweight='bold')
            axes[idx].grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'validation_metrics.png'), dpi=300, bbox_inches='tight')
    print(f"✓ 保存: validation_metrics.png")
    plt.close()

def generate_summary_stats(df, output_dir):
    """
    生成训练统计摘要
    """
    summary = []
    summary.append("=" * 60)
    summary.append("训练统计摘要")
    summary.append("=" * 60)
    
    # 训练步数
    if 'step' in df.columns:
        max_step = df['step'].max()
        summary.append(f"\n总训练步数: {max_step:,}")
    
    # 训练loss统计
    if 'train_loss' in df.columns:
        train_losses = df['train_loss'].dropna()
        if len(train_losses) > 0:
            summary.append(f"\n训练Loss:")
            summary.append(f"  - 初始: {train_losses.iloc[0]:.4f}")
            summary.append(f"  - 最终: {train_losses.iloc[-1]:.4f}")
            summary.append(f"  - 最小: {train_losses.min():.4f}")
            summary.append(f"  - 平均: {train_losses.mean():.4f}")
            summary.append(f"  - 标准差: {train_losses.std():.4f}")
    
    # 验证loss统计
    if 'val_loss' in df.columns:
        val_losses = df['val_loss'].dropna()
        if len(val_losses) > 0:
            summary.append(f"\n验证Loss:")
            summary.append(f"  - 初始: {val_losses.iloc[0]:.4f}")
            summary.append(f"  - 最终: {val_losses.iloc[-1]:.4f}")
            summary.append(f"  - 最小: {val_losses.min():.4f} (step {df[df['val_loss'] == val_losses.min()]['step'].iloc[0]})")
            summary.append(f"  - 平均: {val_losses.mean():.4f}")
            summary.append(f"  - 标准差: {val_losses.std():.4f}")
    
    # 保存摘要
    summary_text = "\n".join(summary)
    print("\n" + summary_text)
    
    with open(os.path.join(output_dir, 'training_summary.txt'), 'w') as f:
        f.write(summary_text)
    
    print(f"\n✓ 保存: training_summary.txt")

def main():
    parser = argparse.ArgumentParser(description='可视化训练过程')
    parser.add_argument('--log-dir', type=str, required=True, 
                       help='训练日志目录')
    parser.add_argument('--smooth-window', type=int, default=10,
                       help='平滑窗口大小')
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    
    # 创建plots目录
    plots_dir = log_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("训练可视化")
    print("=" * 60)
    print(f"日志目录: {log_dir}")
    print(f"输出目录: {plots_dir}")
    print()
    
    # 尝试读取不同格式的日志
    df = None
    
    # 1. 尝试CSV格式
    csv_files = list(log_dir.glob('**/metrics.csv')) + list(log_dir.glob('**/version_*/metrics.csv'))
    if csv_files:
        print(f"找到CSV日志: {csv_files[0]}")
        df = parse_csv_log(csv_files[0])
    
    # 2. 如果没有CSV，尝试解析文本日志
    if df is None or len(df) == 0:
        log_file = log_dir / 'training.log'
        if log_file.exists():
            print(f"解析文本日志: {log_file}")
            df = parse_training_log(log_file)
        else:
            print("❌ 未找到日志文件")
            return
    
    if df is None or len(df) == 0:
        print("❌ 未能解析出任何数据")
        return
    
    print(f"✓ 成功解析 {len(df)} 条记录")
    print()
    
    # 保存解析后的数据
    df.to_csv(plots_dir / 'parsed_metrics.csv', index=False)
    print(f"✓ 保存: parsed_metrics.csv")
    print()
    
    # 生成各种图表
    print("生成图表...")
    print()
    
    plot_loss_curves(df, plots_dir)
    plot_loss_smoothed(df, plots_dir, window=args.smooth_window)
    plot_learning_rate(df, plots_dir)
    plot_validation_metrics(df, plots_dir)
    
    # 生成统计摘要
    generate_summary_stats(df, plots_dir)
    
    print()
    print("=" * 60)
    print("✓ 所有图表已生成")
    print(f"查看结果: ls {plots_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
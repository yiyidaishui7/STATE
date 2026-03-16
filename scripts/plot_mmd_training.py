#!/usr/bin/env python3
"""
plot_mmd_training.py — MMD-AAE 训练过程可视化

解析 Lightning CSV logs（metrics.csv），绘制：
  1. train/val loss 曲线
  2. MMD loss 趋势
  3. ADV loss 趋势
  4. Alignment alpha 调度（warmup 可视化）
  5. 学习率调度

使用方法：

    # 从 Lightning log 目录读取（自动找 metrics.csv）
    python scripts/plot_mmd_training.py \\
        --log_dir /path/to/lightning_logs/mmd_aae_pretrain \\
        --output /path/to/output

    # 或直接指定 CSV 文件
    python scripts/plot_mmd_training.py \\
        --csv /path/to/lightning_logs/version_0/metrics.csv \\
        --output /path/to/output
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    p = argparse.ArgumentParser()
    group = p.add_mutually_exclusive_group(required=True)
    group.add_argument("--log_dir", help="Lightning log 目录（含 version_X/metrics.csv）")
    group.add_argument("--csv", help="直接指定 metrics.csv 文件")
    p.add_argument("--output", default=None, help="输出图片目录")
    p.add_argument("--smooth", type=int, default=20, help="滑动平均窗口大小")
    return p.parse_args()


def find_metrics_csv(log_dir):
    """在 log_dir 下找最新的 metrics.csv。"""
    csv_files = sorted(Path(log_dir).rglob("metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"未在 {log_dir} 下找到 metrics.csv")
    print(f"找到 {len(csv_files)} 个 metrics.csv，使用最新的：{csv_files[-1]}")
    return str(csv_files[-1])


def load_metrics(csv_path):
    """加载并预处理 Lightning metrics.csv。"""
    df = pd.read_csv(csv_path)
    print(f"加载 {len(df)} 条记录，列：{list(df.columns)}")

    # Lightning CSV 有时用 step 列，有时用 epoch 列
    # 每个 metric 在不同的行记录（其他列为 NaN）
    return df


def smooth(series, window):
    """Pandas rolling mean，保留边缘。"""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def extract_metric(df, col):
    """提取单个 metric 列的 (step, value) 数据（去除 NaN）。"""
    if col not in df.columns:
        return None, None
    valid = df[["step", col]].dropna(subset=[col])
    if len(valid) == 0:
        return None, None
    return valid["step"].values, valid[col].values


# ============================================================================
# 绘图函数
# ============================================================================

def plot_loss_overview(df, smooth_w, output_dir):
    """图1: Train Loss + Val Loss"""
    step_train, val_train = extract_metric(df, "trainer/train_loss")
    step_val, val_val = extract_metric(df, "validation/val_loss")

    if step_train is None and step_val is None:
        # 兼容不带前缀的列名
        step_train, val_train = extract_metric(df, "train_loss")
        step_val, val_val = extract_metric(df, "val_loss")

    if step_train is None and step_val is None:
        print("⚠️  未找到 train/val loss 数据")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：两者叠加
    ax = axes[0]
    if step_train is not None:
        ax.plot(step_train, val_train, alpha=0.2, color="#3498DB", linewidth=0.5)
        s = pd.Series(val_train)
        ax.plot(step_train, smooth(s, smooth_w).values,
                color="#3498DB", linewidth=2, label="Train Loss")
    if step_val is not None:
        ax.plot(step_val, val_val, color="#E74C3C", linewidth=2,
                marker="o", markersize=3, label="Val Loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 右图：仅 val loss
    ax = axes[1]
    if step_val is not None:
        ax.plot(step_val, val_val, color="#E74C3C", linewidth=2,
                marker="o", markersize=4, label="Val Loss")
        best_idx = np.argmin(val_val)
        ax.axhline(val_val[best_idx], color="gray", linestyle="--", alpha=0.5)
        ax.annotate(f"Best={val_val[best_idx]:.4f}",
                    xy=(step_val[best_idx], val_val[best_idx]),
                    xytext=(step_val[best_idx] + 0.02 * step_val[-1], val_val[best_idx] * 1.01),
                    fontsize=9, color="#E74C3C")
    ax.set_xlabel("Step")
    ax.set_ylabel("Val Loss")
    ax.set_title("Validation Loss (Detail)", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(output_dir, "01_loss_curves.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存: {out}")


def plot_mmd_alignment(df, smooth_w, output_dir):
    """图2: MMD loss + ADV loss + Alignment Alpha（3 合 1）"""
    step_mmd, val_mmd = extract_metric(df, "trainer/mmd_loss")
    step_adv, val_adv = extract_metric(df, "trainer/adv_loss")
    step_alpha, val_alpha = extract_metric(df, "trainer/alignment_alpha")

    has_any = any(v is not None for v in [step_mmd, step_adv, step_alpha])
    if not has_any:
        print("⚠️  未找到 MMD 对齐 loss 数据（trainer/mmd_loss, trainer/adv_loss, trainer/alignment_alpha）")
        print("    这可能是因为：(a) 训练仍在 warmup 阶段, 或 (b) 日志未保存这些指标")
        return

    n_plots = sum(v is not None for v in [step_mmd, step_adv, step_alpha])
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    ax_idx = 0

    if step_mmd is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(step_mmd, val_mmd, alpha=0.2, color="#9B59B6", linewidth=0.5)
        s = pd.Series(val_mmd)
        ax.plot(step_mmd, smooth(s, smooth_w).values,
                color="#9B59B6", linewidth=2, label="MMD Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("MMD Loss")
        ax.set_title("Pairwise MMD Loss\n(K562↔RPE1↔Jurkat)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        # 期望趋势：随训练降低
        ax.text(0.02, 0.95, "↓ 越低 = 域间分布越接近",
                transform=ax.transAxes, fontsize=9, color="#9B59B6", verticalalignment="top")

    if step_adv is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(step_adv, val_adv, alpha=0.2, color="#E74C3C", linewidth=0.5)
        s = pd.Series(val_adv)
        ax.plot(step_adv, smooth(s, smooth_w).values,
                color="#E74C3C", linewidth=2, label="ADV Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("ADV Loss")
        ax.set_title("Adversarial Domain Loss\n(GRL 对抗)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if step_alpha is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(step_alpha, val_alpha, color="#F39C12", linewidth=2, label="Alignment α")
        ax.set_xlabel("Step")
        ax.set_ylabel("Alpha")
        ax.set_title("Alignment Schedule (α)\nWarmup → Linear Ramp → 1.0", fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        # 标注 warmup 结束
        warmup_end = step_alpha[np.searchsorted(val_alpha, 0.01)]
        ax.axvline(warmup_end, color="gray", linestyle="--", alpha=0.6,
                   label=f"Warmup end (~step {warmup_end})")

    plt.tight_layout()
    out = os.path.join(output_dir, "02_mmd_alignment_losses.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存: {out}")


def plot_full_dashboard(df, smooth_w, output_dir):
    """图3: 综合 Dashboard（4 图 2×2）"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    metrics_to_plot = [
        ("trainer/train_loss",       "Train Loss",        "#3498DB", gs[0, 0]),
        ("validation/val_loss",      "Val Loss",          "#E74C3C", gs[0, 1]),
        ("trainer/mmd_loss",         "MMD Loss",          "#9B59B6", gs[1, 0]),
        ("trainer/alignment_alpha",  "Alignment α",       "#F39C12", gs[1, 1]),
    ]

    for col, title, color, slot in metrics_to_plot:
        ax = fig.add_subplot(slot)
        step, val = extract_metric(df, col)
        if step is None:
            ax.text(0.5, 0.5, f"No data\n({col})",
                    ha="center", va="center", transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(title, fontweight="bold")
            continue

        ax.plot(step, val, alpha=0.2, color=color, linewidth=0.5)
        s = pd.Series(val)
        ax.plot(step, smooth(s, smooth_w).values, color=color, linewidth=2)
        ax.set_xlabel("Step", fontsize=10)
        ax.set_ylabel(title, fontsize=10)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.grid(True, alpha=0.3)

        # 在图上注释最终值
        final_val = smooth(s, smooth_w).iloc[-1]
        ax.text(0.98, 0.05, f"Final: {final_val:.4f}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("STATE+MMD-AAE Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    out = os.path.join(output_dir, "03_training_dashboard.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✓ 保存: {out}")


def print_training_summary(df):
    """打印关键训练统计。"""
    print("\n" + "=" * 60)
    print("📈 训练统计摘要")
    print("=" * 60)

    if "step" in df.columns:
        print(f"  总步数:     {int(df['step'].max()):,}")

    for col, name in [
        ("trainer/train_loss",      "Train Loss"),
        ("validation/val_loss",     "Val Loss"),
        ("trainer/mmd_loss",        "MMD Loss"),
        ("trainer/adv_loss",        "ADV Loss"),
        ("trainer/alignment_alpha", "Alignment α"),
    ]:
        step, val = extract_metric(df, col)
        if val is None:
            continue
        print(f"\n  {name}:")
        print(f"    初始: {val[0]:.6f}")
        print(f"    最终: {val[-1]:.6f}")
        print(f"    最小: {val.min():.6f}")
        if name in ("MMD Loss", "ADV Loss"):
            # 期望值变化方向
            trend = "↓ 降低" if val[-1] < val[0] else "↑ 升高（不理想）"
            print(f"    趋势: {trend}")

    print("=" * 60)


def main():
    args = parse_args()

    # 找 CSV 文件
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_metrics_csv(args.log_dir)

    if not os.path.exists(csv_path):
        print(f"❌ 文件不存在: {csv_path}")
        return

    # 输出目录
    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "plots_mmd")
    os.makedirs(output_dir, exist_ok=True)
    print(f"输出目录: {output_dir}")

    # 加载数据
    df = load_metrics(csv_path)

    # 打印摘要
    print_training_summary(df)

    # 生成图表
    print("\n生成图表...")
    plot_loss_overview(df, args.smooth, output_dir)
    plot_mmd_alignment(df, args.smooth, output_dir)
    plot_full_dashboard(df, args.smooth, output_dir)

    print(f"\n✅ 所有图表已生成 → {output_dir}")


if __name__ == "__main__":
    main()

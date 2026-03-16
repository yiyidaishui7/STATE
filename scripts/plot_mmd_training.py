#!/usr/bin/env python3
"""
plot_mmd_training.py - MMD-AAE Training Visualization

Parses Lightning CSV logs (metrics.csv) and plots:
  1. train/val loss curves
  2. MMD loss trend
  3. ADV loss trend
  4. Alignment alpha schedule (warmup visualization)
  5. Learning rate schedule

Usage:

    # Read from Lightning log directory (auto-finds metrics.csv)
    python scripts/plot_mmd_training.py \\
        --log_dir /path/to/lightning_logs/mmd_aae_pretrain \\
        --output /path/to/output

    # Or specify CSV file directly
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
    group.add_argument("--log_dir", help="Lightning log directory (containing version_X/metrics.csv)")
    group.add_argument("--csv", help="Path to metrics.csv file directly")
    p.add_argument("--output", default=None, help="Output directory for plots")
    p.add_argument("--smooth", type=int, default=20, help="Rolling average window size")
    return p.parse_args()


def find_metrics_csv(log_dir):
    """Find the latest metrics.csv under log_dir."""
    csv_files = sorted(Path(log_dir).rglob("metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics.csv found under {log_dir}")
    print(f"Found {len(csv_files)} metrics.csv file(s), using latest: {csv_files[-1]}")
    return str(csv_files[-1])


def load_metrics(csv_path):
    """Load and preprocess Lightning metrics.csv."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records, columns: {list(df.columns)}")
    return df


def smooth(series, window):
    """Pandas rolling mean, preserving edges."""
    return series.rolling(window=window, min_periods=1, center=True).mean()


def extract_metric(df, col):
    """Extract (step, value) arrays for a single metric column (drops NaN)."""
    if col not in df.columns:
        return None, None
    valid = df[["step", col]].dropna(subset=[col])
    if len(valid) == 0:
        return None, None
    return valid["step"].values, valid[col].values


# ============================================================================
# Plot functions
# ============================================================================

def plot_loss_overview(df, smooth_w, output_dir):
    """Plot 1: Train Loss + Val Loss"""
    step_train, val_train = extract_metric(df, "trainer/train_loss")
    step_val, val_val = extract_metric(df, "validation/val_loss")

    if step_train is None and step_val is None:
        # Fallback to unprefixed column names
        step_train, val_train = extract_metric(df, "train_loss")
        step_val, val_val = extract_metric(df, "val_loss")

    if step_train is None and step_val is None:
        print("WARNING: No train/val loss data found")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: both overlaid
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

    # Right: val loss only
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
    print(f"Saved: {out}")


def plot_mmd_alignment(df, smooth_w, output_dir):
    """Plot 2: MMD loss + ADV loss + Alignment Alpha (3-in-1)"""
    step_mmd, val_mmd = extract_metric(df, "trainer/mmd_loss")
    step_adv, val_adv = extract_metric(df, "trainer/adv_loss")
    step_alpha, val_alpha = extract_metric(df, "trainer/alignment_alpha")

    has_any = any(v is not None for v in [step_mmd, step_adv, step_alpha])
    if not has_any:
        print("WARNING: No MMD alignment loss data found (trainer/mmd_loss, trainer/adv_loss, trainer/alignment_alpha)")
        print("  Possible reasons: (a) training still in warmup phase, or (b) metrics not logged")
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
        ax.set_title("Pairwise MMD Loss\n(K562<->RPE1<->Jurkat)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.text(0.02, 0.95, "Lower = domains more aligned",
                transform=ax.transAxes, fontsize=9, color="#9B59B6", verticalalignment="top")

    if step_adv is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(step_adv, val_adv, alpha=0.2, color="#E74C3C", linewidth=0.5)
        s = pd.Series(val_adv)
        ax.plot(step_adv, smooth(s, smooth_w).values,
                color="#E74C3C", linewidth=2, label="ADV Loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("ADV Loss")
        ax.set_title("Adversarial Domain Loss\n(GRL adversarial)", fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)

    if step_alpha is not None:
        ax = axes[ax_idx]; ax_idx += 1
        ax.plot(step_alpha, val_alpha, color="#F39C12", linewidth=2, label="Alignment alpha")
        ax.set_xlabel("Step")
        ax.set_ylabel("Alpha")
        ax.set_title("Alignment Schedule (alpha)\nWarmup -> Linear Ramp -> 1.0", fontweight="bold")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)
        warmup_end = step_alpha[np.searchsorted(val_alpha, 0.01)]
        ax.axvline(warmup_end, color="gray", linestyle="--", alpha=0.6,
                   label=f"Warmup end (~step {warmup_end})")

    plt.tight_layout()
    out = os.path.join(output_dir, "02_mmd_alignment_losses.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def plot_full_dashboard(df, smooth_w, output_dir):
    """Plot 3: Combined Dashboard (2x2 grid)"""
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    metrics_to_plot = [
        ("trainer/train_loss",       "Train Loss",        "#3498DB", gs[0, 0]),
        ("validation/val_loss",      "Val Loss",          "#E74C3C", gs[0, 1]),
        ("trainer/mmd_loss",         "MMD Loss",          "#9B59B6", gs[1, 0]),
        ("trainer/alignment_alpha",  "Alignment alpha",   "#F39C12", gs[1, 1]),
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

        final_val = smooth(s, smooth_w).iloc[-1]
        ax.text(0.98, 0.05, f"Final: {final_val:.4f}",
                ha="right", va="bottom", transform=ax.transAxes,
                fontsize=9, color=color,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.suptitle("STATE+MMD-AAE Training Dashboard", fontsize=16, fontweight="bold", y=0.98)
    out = os.path.join(output_dir, "03_training_dashboard.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


def print_training_summary(df):
    """Print key training statistics."""
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)

    if "step" in df.columns:
        print(f"  Total steps:  {int(df['step'].max()):,}")

    for col, name in [
        ("trainer/train_loss",      "Train Loss"),
        ("validation/val_loss",     "Val Loss"),
        ("trainer/mmd_loss",        "MMD Loss"),
        ("trainer/adv_loss",        "ADV Loss"),
        ("trainer/alignment_alpha", "Alignment alpha"),
    ]:
        step, val = extract_metric(df, col)
        if val is None:
            continue
        print(f"\n  {name}:")
        print(f"    Initial: {val[0]:.6f}")
        print(f"    Final:   {val[-1]:.6f}")
        print(f"    Min:     {val.min():.6f}")
        if name in ("MMD Loss", "ADV Loss"):
            trend = "decreased (good)" if val[-1] < val[0] else "increased (not ideal)"
            print(f"    Trend:   {trend}")

    print("=" * 60)


def main():
    args = parse_args()

    if args.csv:
        csv_path = args.csv
    else:
        csv_path = find_metrics_csv(args.log_dir)

    if not os.path.exists(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return

    output_dir = args.output
    if output_dir is None:
        output_dir = str(Path(csv_path).parent / "plots_mmd")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    df = load_metrics(csv_path)

    print_training_summary(df)

    print("\nGenerating plots...")
    plot_loss_overview(df, args.smooth, output_dir)
    plot_mmd_alignment(df, args.smooth, output_dir)
    plot_full_dashboard(df, args.smooth, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
plot_topk_curve.py — 画 Top-k PCC 折线图（run4 数据，k=3~50）

数据来源：2026-05-13 run4 终端输出，已记录在 docs/deg_experiments_0513.md
不需要重新跑实验，直接用已有数据。

用法：
    python scripts/plot_topk_curve.py
    python scripts/plot_topk_curve.py --output figures/topk_curve.pdf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── run4 数据（来自 docs/deg_experiments_0513.md） ──────────────────────────

MMD_DATA = {
    3: +0.1143, 4: +0.0827, 5: +0.1028, 6: +0.0557, 7: +0.0442,
    8: +0.0400, 9: +0.0333, 10: +0.0549, 11: +0.0368, 12: +0.0560,
    13: +0.0612, 14: +0.0499, 15: +0.0431, 16: +0.0436, 17: +0.0416,
    18: +0.0410, 19: +0.0384, 20: +0.0358, 21: +0.0431, 22: +0.0479,
    23: +0.0469, 24: +0.0510, 25: +0.0541, 26: +0.0563, 27: +0.0577,
    28: +0.0562, 29: +0.0523, 30: +0.0474, 31: +0.0485, 32: +0.0496,
    33: +0.0514, 34: +0.0539, 35: +0.0520, 36: +0.0512, 37: +0.0510,
    38: +0.0518, 39: +0.0507, 40: +0.0500, 41: +0.0518, 42: +0.0483,
    43: +0.0519, 44: +0.0533, 45: +0.0523, 46: +0.0526, 47: +0.0543,
    48: +0.0557, 49: +0.0557, 50: +0.0563,
}

BASELINE_DATA = {
    3: -0.0194, 4: +0.0030, 5: +0.0190, 6: +0.0082, 7: +0.0250,
    8: +0.0358, 9: +0.0204, 10: +0.0328, 11: +0.0514, 12: +0.0396,
    13: +0.0335, 14: +0.0381, 15: +0.0401, 16: +0.0440, 17: +0.0266,
    18: +0.0217, 19: +0.0219, 20: +0.0277, 21: +0.0264, 22: +0.0244,
    23: +0.0305, 24: +0.0362, 25: +0.0406, 26: +0.0340, 27: +0.0348,
    28: +0.0348, 29: +0.0311, 30: +0.0350, 31: +0.0362, 32: +0.0419,
    33: +0.0356, 34: +0.0300, 35: +0.0294, 36: +0.0316, 37: +0.0304,
    38: +0.0284, 39: +0.0280, 40: +0.0327, 41: +0.0339, 42: +0.0371,
    43: +0.0335, 44: +0.0305, 45: +0.0316, 46: +0.0315, 47: +0.0293,
    48: +0.0272, 49: +0.0273, 50: +0.0243,
}

# ── 主函数 ──────────────────────────────────────────────────────────────────

def plot(output: str | None = None):
    ks = sorted(MMD_DATA.keys())
    mmd_vals = [MMD_DATA[k] for k in ks]
    bl_vals  = [BASELINE_DATA[k] for k in ks]

    fig, ax = plt.subplots(figsize=(10, 5))

    # 两条折线
    ax.plot(ks, mmd_vals,  color="#2563EB", linewidth=2,   label="STATE+MMD",  zorder=3)
    ax.plot(ks, bl_vals,   color="#EA580C", linewidth=2,   label="Baseline",   zorder=3)

    # y=0 参考线
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5, zorder=1)

    # ── 三阶段背景色 ──
    ax.axvspan(3,  5.5, alpha=0.08, color="#2563EB", zorder=0)   # 信号峰
    ax.axvspan(5.5, 15.5, alpha=0.05, color="gray",  zorder=0)   # 衰减区
    ax.axvspan(15.5, 50,  alpha=0.06, color="#16A34A", zorder=0) # 平台区

    # phase labels
    ax.text(4.2,  0.125, "Signal peak\n  k=3~5",   fontsize=9, color="#1D4ED8", ha="center", va="bottom")
    ax.text(10.5, 0.125, "Decay\nk=6~15",           fontsize=9, color="#555",    ha="center", va="bottom")
    ax.text(33,   0.110, "Plateau  k=16~50",        fontsize=9, color="#15803D", ha="center", va="bottom")

    # k=3 callout
    ax.annotate(
        "k=3\nMMD  = +0.114\nBase = -0.019",
        xy=(3, MMD_DATA[3]),
        xytext=(7, 0.105),
        fontsize=8,
        color="#1D4ED8",
        arrowprops=dict(arrowstyle="->", color="#1D4ED8", lw=1.2),
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#2563EB", lw=1),
    )

    ax.set_xlabel("k  (top-k genes by |Wilcoxon score|)", fontsize=11)
    ax.set_ylabel("Mean Pearson r (PCC)", fontsize=11)
    ax.set_title("Top-k PCC curve  (run4, HepG2 zero-shot, 67 perturbations)", fontsize=12)
    ax.set_xlim(2, 51)
    ax.set_xticks([3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150, bbox_inches="tight")
        print(f"已保存：{output}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default=None, help="输出路径，如 figures/topk_curve.pdf")
    args = p.parse_args()
    plot(args.output)

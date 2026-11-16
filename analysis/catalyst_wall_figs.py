"""
catalyst_wall_figs.py
---------------------
Figure for catalyst_wall_stability.py: are large groups more stable near the wall?

Usage:
    python catalyst_wall_figs.py <out>      # <out> holds wall_stability/ and receives fig7
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import catalyst_radial_stats as crs
from catalyst_radial_figs import COL, CONDS, GRID, INK, INK2, INK3, MARK, NAME, SURFACE, grid, save  # also sets rcParams

BINS = ["<0.55", "0.55-0.70", "0.70-0.78", "0.78-0.85", ">=0.85"]
BIN_TEXT = ["<0.55", "0.55–0.70", "0.70–0.78", "0.78–0.85", "≥0.85"]
REG = {"inner": ("内圈", "#52514e", "-"), "gap": ("外圈内侧", "#eb9834", "--"), "wall": ("贴墙层", "#1baf7a", "-")}


def profile(ax, P, col, ylabel, title, scale=1.0, legend=False):
    for c in CONDS:
        t = P[P.cond == c].pivot(index="trial", columns="bin", values=col)[BINS].to_numpy(float) * scale
        m, s = t.mean(0), t.std(0, ddof=1) / np.sqrt(len(t))
        ax.fill_between(range(5), m - s, m + s, color=COL[c], alpha=0.16, lw=0)
        ax.plot(range(5), m, color=COL[c], lw=2, marker=MARK[c], ms=6, mec=SURFACE, mew=1, label=NAME[c])
    ax.axvspan(2.5, 4.3, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xlim(-0.3, 4.3)
    ax.set_xticks(range(5), BIN_TEXT, fontsize=8.5)
    ax.set_xlabel("group 质心的 r / R（灰底为外圈）")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    grid(ax)
    if legend:
        ax.legend(frameon=False, loc="best")


def forest(ax, C):
    rows = [("decay_hazard", "衰减率"), ("decay_hazard_6_7", "衰减率（只算 size 6–7）"), ("split_hazard", "分裂率（按大小标准化）")]
    y, ticks, labels = 0, [], []
    for me, lab in rows:
        ticks.append(y)
        labels.append(lab)
        for off, c in zip((0.24, 0, -0.24), CONDS):
            r = C[(C.window == "t>=100") & (C.contrast == "wall vs inner") & (C.metric == me) & (C.cond == c)].iloc[0]
            ax.plot([r.lo, r.hi], [y + off] * 2, color=COL[c], lw=2, solid_capstyle="round")
            ax.plot(r.effect, y + off, MARK[c], color=COL[c], ms=7, mec=SURFACE, mew=1)
        y -= 1
    ax.axvline(1, color=INK3, lw=1)
    ax.set_xscale("log")
    ax.set_xticks([0.2, 0.3, 0.5, 1.0], ["0.2", "0.3", "0.5", "1"])
    ax.minorticks_off()
    ax.set_yticks(ticks, labels)
    ax.set_ylim(y + 0.5, 0.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("贴墙层 / 内圈 的比值，< 1 表示贴墙层更稳定（95% CI）")
    ax.set_title("c  贴墙层相对内圈的风险比")
    grid(ax, "x")


def main():
    out = sys.argv[1]
    wd = os.path.join(out, "wall_stability")
    P = pd.read_csv(os.path.join(wd, "radial_profile_per_trial.csv"))
    P = P[P.window == "t>=100"]
    C = pd.read_csv(os.path.join(wd, "contrasts.csv"))
    I = pd.read_csv(os.path.join(wd, "formation_lifetimes.csv"))
    I = I[I.window == "t>=100"]

    fig, axs = plt.subplots(2, 3, figsize=(13, 7.6), constrained_layout=True)
    profile(axs[0, 0], P, "decay_hazard", "次 / (group·min)", "a  衰减率随位置的变化", legend=True)
    axs[0, 0].set_ylim(bottom=0)
    profile(axs[0, 1], P, "retention30_6_9", "30 s 后仍在一起的成员（%）", "b  size 6–9 group 的成员留存", scale=100)
    forest(axs[0, 2], C)
    for ax, c in zip(axs[1], CONDS):
        for reg, (lab, col, ls) in REG.items():
            s = I[(I.cond == c) & (I.reg == reg)]
            x, y = crs.km_curve(s.dur.to_numpy(), s.obs.to_numpy())
            ax.step(x, y, where="post", color=col, ls=ls, lw=2, label=f"{lab} (n={len(s)})")
        ax.axhline(0.5, color=INK3, lw=1, ls=":")
        ax.set_xscale("log")
        ax.set_xlim(1, 400)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("形成后存活时间（s）")
        ax.set_ylabel("存活比例")
        ax.set_title({"baseline": "d", "farthest": "e", "nearest": "f"}[c] + f"  按形成位置分：{NAME[c]}（5 trial 合并）")
        grid(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)
    save(fig, out, "fig7_wall_stability")


if __name__ == "__main__":
    main()

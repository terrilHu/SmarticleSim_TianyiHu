"""
catalyst_fixed_combined.py
--------------------------
Size 3+ and size 6+ shares on one chart, from the tables catalyst_fixed_dose_curves.py
already wrote (no recomputation):
    <dir>/catalyst_fixed_ge3_<bases>_last<L>.csv   (--size-ge 3)
    <dir>/catalyst_fixed_<bases>_last<L>.csv        (default, size >= 6)

Colour = baseline, solid line + filled marker = size 6+, dashed line + open marker = size 3+.
Point = mean over trials, whisker = 95% CI; 0% = each baseline, 100% = the shared all-catalyst run.

--layout overlay   one axes (default)
--layout stacked   two panels sharing the x axis, same y scale

Output: <dir>/catalyst_fixed_level_ge3_ge6_<bases>_last<L>[_stacked][_bare].png / .svg

Usage:
    python catalyst_fixed_combined.py DIR --groups a816 a834 a862 [--layout stacked] [--bare]
"""

import argparse
import os

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from catalyst_dose_curves import GRID, INK, INK2, PAIR_COLORS, SURFACE, X_TICKS

METRICS = [  # (file tag, label, line style, open marker)
    ("_ge3", "size 3+", (0, (4, 2.5)), True),
    ("", "size 6+", "-", False),
]


def spread(ys, gap):
    """Move label heights apart to at least gap, keeping each pair centred."""
    order = np.argsort(ys)
    v = list(np.asarray(ys, float)[order])
    for _ in range(50):
        moved = False
        for a in range(len(v) - 1):
            d = v[a + 1] - v[a]
            if d < gap - 1e-9:
                v[a] -= (gap - d) / 2
                v[a + 1] += (gap - d) / 2
                moved = True
        if not moved:
            break
    out = np.empty(len(v))
    out[order] = v
    return out


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def draw(ax, tables, bases, metrics, bare, label_left=True):
    for tagm, mlabel, ls, hollow in metrics:
        t = tables[tagm]
        lab = []
        for j, (base, col) in enumerate(zip(bases, PAIR_COLORS)):
            c = t[t.baseline == base].sort_values("amount_pct")
            pinned = (c.amount_pct == 0) | (c.amount_pct >= 100)
            x = c.amount_pct.to_numpy() + np.where(pinned, 0, (j - (len(bases) - 1) / 2) * 0.8)
            y = c.strategy_share_pct
            ax.errorbar(x, y, yerr=[y - c.strategy_ci95_lo, c.strategy_ci95_hi - y], color=col, lw=2, ls=ls,
                        elinewidth=1.1, capsize=3, marker="o", ms=7.5, mfc=SURFACE if hollow else col,
                        mec=col if hollow else SURFACE, mew=1.6 if hollow else 1.5, zorder=3 + j)
            lab.append((y.iloc[0], base))
        if label_left and not bare:
            ys = spread([v for v, _ in lab], 0.045 * np.ptp(ax.get_ylim()))
            for yl, (_, base) in zip(ys, lab):
                ax.text(-1.6, yl, base, ha="right", va="center", fontsize=12, color=INK)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--groups", nargs="+", required=True)
    ap.add_argument("--last", type=float, default=100.0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--layout", choices=["overlay", "stacked"], default="overlay")
    ap.add_argument("--bare", action="store_true", help="also write a curve-only copy (<name>_bare)")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="fixed y limits, e.g. to match another set of baselines")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

    tag = "_".join(args.groups)
    tables = {m[0]: pd.read_csv(os.path.join(args.dir, f"catalyst_fixed{m[0]}_{tag}_last{args.last:g}.csv"))
              for m in METRICS}
    n_tr = int(tables[""].n.iloc[0])
    ymax = max(t.strategy_ci95_hi.max() for t in tables.values())
    ylim = tuple(args.ylim) if args.ylim else (0, np.ceil(ymax / 5) * 5 + 2)

    for bare in ([False, True] if args.bare else [False]):
        if args.layout == "overlay":
            fig, ax = plt.subplots(figsize=(9.0, 6.6), facecolor=SURFACE, layout="constrained")
            axes = [ax]
            style(ax)
            ax.set_ylim(*ylim)
            draw(ax, tables, args.groups, METRICS, bare)
            ax.set_ylabel("share of robots in groups of that size (%)")
            if not bare:
                # name the two bundles where they run apart (50-75%)
                for tagm, mlabel, _, _ in METRICS:
                    t = tables[tagm]
                    y = t[t.amount_pct == 50].strategy_share_pct.max()
                    ax.text(52, y + 0.035 * np.ptp(ylim), mlabel, fontsize=13, color=INK2, fontweight="bold")
        else:
            fig, axes = plt.subplots(2, 1, figsize=(9.0, 9.0), sharex=True, facecolor=SURFACE, layout="constrained")
            for ax, m in zip(axes, METRICS):
                style(ax)
                ax.set_ylim(*ylim)
                draw(ax, tables, args.groups, [m], bare)
                ax.set_ylabel(f"share in {m[1]} (%)")
            ax = axes[-1]
        for a in axes:
            a.set_xlim(-3 if bare else -13, 105)
        name = f"catalyst_fixed_level_ge3_ge6_{tag}_last{args.last:g}" + ("_stacked" if args.layout == "stacked" else "")
        if bare:
            axes[-1].set_xticks(X_TICKS, [""] * len(X_TICKS))
            name += "_bare"
        else:
            axes[-1].set_xticks(X_TICKS, [f"{p}%\n({round(p * args.n / 100)})" for p in X_TICKS])
            axes[-1].tick_params(axis="x", length=4, labelsize=13)
            axes[-1].set_xlabel(f"catalysts: % of robots (number, of {args.n})")
            axes[0].set_title(f"Size 3+ and size 6+ share against fixed catalyst amount (-852), last {args.last:g} s\n"
                              f"point = mean over {n_tr} trials, whisker = 95% CI; 0% = baseline;\n"
                              "100% = all robots catalysts, one run shared by every curve", loc="left", fontsize=12)
            handles = [Line2D([], [], color=c, lw=2, marker="o", ms=7, mec=SURFACE, label=b)
                       for b, c in zip(args.groups, PAIR_COLORS)]
            if args.layout == "overlay":
                handles += [Line2D([], [], color=INK2, lw=2, ls=ls, marker="o", ms=7,
                                   mfc=SURFACE if hollow else INK2, mec=INK2 if hollow else SURFACE, label=lab)
                            for _, lab, ls, hollow in reversed(METRICS)]
            fig.legend(handles=handles, frameon=False, loc="outside lower center", ncol=len(handles), fontsize=13,
                       handlelength=2.4, columnspacing=1.4)
        out = os.path.join(args.dir, name)
        fig.savefig(out + ".png", dpi=150, facecolor=SURFACE)
        fig.savefig(out + ".svg", facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out}.png / .svg", flush=True)


if __name__ == "__main__":
    main()

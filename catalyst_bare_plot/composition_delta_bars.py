"""
composition_delta_bars.py
-------------------------
Panel C of strategy_hypotheses.py (change in group-size composition,
strategy - baseline) for several baseline/strategy pairs in one chart.

Reads <root>/<base>_vs_<strat>/hypotheses_<base>_vs_<strat>_<window>_trials.csv
for each requested baseline and, per size class (1-2 / 3-5 / 6+ with the
default --grouped 3 --large 6), plots the mean paired difference in the share
of robots, whisker = 95% t-interval over the seed-matched trials, dots = the
individual pairs. Within each pair the three classes sum to 0.

Usage:
    python composition_delta_bars.py <root> a016 a034 a062 [--window last100] [--name NAME]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# categorical slots 3, 4, 7 of the dataviz reference palette: blue / orange already
# mean baseline / strategy in the other figures, so pairs get other hues
PAIR_COLORS = ["#1baf7a", "#eda100", "#6250d6", "#e87ba4", "#008300"]


def deltas(root, base, window, grouped, large, exp_root):
    hits = glob.glob(os.path.join(root, f"{base}_vs_*", f"hypotheses_{base}_vs_*_{window}_trials.csv"))
    if len(hits) != 1:
        raise SystemExit(f"{base}: expected one *_{window}_trials.csv under {root}, found {len(hits)}")
    t = pd.read_csv(hits[0])
    strat = [e for e in t.exp.unique() if e != base][0]
    w = t.pivot(index="trial", columns="exp")
    g, l = f"share_ge{grouped}", f"share_ge{large}"
    share = {f"1-{grouped - 1}": lambda e: 1 - w[g][e],
             f"{grouped}-{large - 1}": lambda e: w[g][e] - w[l][e],
             f"{large}+": lambda e: w[l][e]}
    d = pd.DataFrame({k: (f(strat) - f(base)) * 100 for k, f in share.items()})
    try:
        cfg = json.load(open(os.path.join(exp_root, strat, "config_snapshot.json"), encoding="utf-8"))
        cmd = cfg["STRATEGY_SPEC"]["leave_command"]
    except (OSError, KeyError):
        cmd = None
    return strat, cmd, d


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("baselines", nargs="+")
    ap.add_argument("--window", default="last100")
    ap.add_argument("--grouped", type=int, default=3)
    ap.add_argument("--large", type=int, default=6)
    ap.add_argument("--name", default=None)
    ap.add_argument("--exp-root", default=None, help="folder holding the strategy experiments (default: <root>/..)")
    ap.add_argument("--bare", action="store_true",
                    help="bars, whiskers and the y axis only (no p, k/n, title, legend or x labels); "
                         "written as <name>_bare")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="fixed y range, so that several charts share one scale")
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    exp_root = os.path.abspath(args.exp_root) if args.exp_root else os.path.dirname(root)

    pairs = [(b,) + deltas(root, b, args.window, args.grouped, args.large, exp_root) for b in args.baselines]
    classes = list(pairs[0][3].columns)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none",     # keep text as text in the .svg (editable in Illustrator/Inkscape)
                         "font.size": 14,"axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, ax = plt.subplots(figsize=(4.0 + 2.2 * len(classes) * len(pairs) / 3, 6.2), facecolor=SURFACE,
                           layout="constrained")
    # x in data units, y in axes units: the k/n row sits under the x axis
    below = matplotlib.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    k = len(pairs)
    w = 0.8 / k
    rows = []
    for j, ((base, strat, cmd, d), col) in enumerate(zip(pairs, PAIR_COLORS)):
        n = len(d)
        q = stats.t.ppf(0.975, n - 1)
        for i, c in enumerate(classes):
            v = d[c].to_numpy()
            m, ci = v.mean(), q * v.std(ddof=1) / np.sqrt(n)
            p = stats.ttest_1samp(v, 0).pvalue
            rows.append(dict(baseline=base, strategy=strat, strategy_cmd=cmd, size_class=c, mean_pp=m,
                             ci95_lo=m - ci, ci95_hi=m + ci, n_pairs_up=int((v > 0).sum()), paired_t_p=p))
            x = i + (j - (k - 1) / 2) * w
            # 2 px surface gap between neighbouring bars
            ax.bar(x, m, width=w * 0.92, color=col, edgecolor=SURFACE, lw=1,
                   label=f"{base}  (strategy cmd {cmd})" if i == 0 else None)
            ax.errorbar(x, m, yerr=ci, color=INK, capsize=4, lw=1.3)
            if args.bare:
                continue
            p_txt = f"p={p:.3f}" if p >= 0.001 else "p<0.001"
            ax.text(x, (m + ci + 0.6) if m >= 0 else (m - ci - 0.6), p_txt, ha="center",
                    va="bottom" if m >= 0 else "top", fontsize=12, color=INK)
            up = int((v > 0).sum()) if m >= 0 else int((v < 0).sum())
            ax.text(x, -0.025, f"{up}/{n}{'↑' if m >= 0 else '↓'}", transform=below, ha="center", va="top",
                    fontsize=12, color=INK if up == n else INK2, fontweight="bold" if up == n else "normal")
    ax.axhline(0, color=INK2, lw=1)
    ax.set_ylabel("Δ share of robots (pp)\nstrategy − baseline")
    name = args.name or "composition_delta_" + "_".join(args.baselines) + f"_{args.window}"
    if args.bare:
        ax.set_xticks([])
        ax.spines["bottom"].set_visible(False)
        name += "_bare"
    else:
        ax.set_xticks(range(len(classes)), [f"size {c}" for c in classes])
        ax.tick_params(axis="x", length=0, pad=30, labelsize=15)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo - 4, hi + 4)
        fig.legend(frameon=False, loc="outside lower center", ncol=len(pairs), fontsize=13,
                   handlelength=1.2, columnspacing=1.5)
        win = args.window.replace("last", "last ") + " s" if args.window.startswith("last") else args.window
        ax.set_title(f"Δ group-size composition, {win}\n"
                     f"whisker = 95% CI over {len(pairs[0][3])} seed-matched trials · p = paired t-test · "
                     "k/n = trials changing in the bar's direction",
                     loc="left", fontsize=12, pad=22)
    if args.ylim:
        ax.set_ylim(*args.ylim)

    out = os.path.join(root, name)
    fig.savefig(out + ".png", dpi=150, facecolor=SURFACE)
    fig.savefig(out + ".svg", facecolor=SURFACE)
    pd.DataFrame(rows).to_csv(out + ".csv", index=False)
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print(f"wrote {out}.png / .svg / .csv")


if __name__ == "__main__":
    main()

"""
hypotheses_overview.py
----------------------
Put the strategy_hypotheses.py results of several baseline/strategy pairs side
by side. Reads every <root>/*/hypotheses_<base>_vs_<strat>_<window>.csv and
writes, into <root>:

    hypotheses_overview_<window>.csv   one row per pair x metric
    hypotheses_overview_<window>.png   one row per pair, one panel per key metric:
                                       paired difference with 95% CI, p, pairs up

The strategy's gait command is read from <strat>/config_snapshot.json
(STRATEGY_SPEC.leave_command), with experiment directories looked up next to
<root> (i.e. <root>/..).

Usage:
    python hypotheses_overview.py <root> [--window last100]
"""

import argparse
import glob
import json
import os
import re

import numpy as np
import pandas as pd

INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]
PANELS = [("share_ge6", 100, "H1   Δ % robots in groups ≥6", SERIES[1]),
          ("large_form_per_min", 1, "H1   Δ groups growing to ≥6 / min", SERIES[1]),
          ("share_ge3", 100, "H2   Δ % robots in groups ≥3", INK)]


def command_of(exp_dir):
    try:
        cfg = json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))
        return cfg["STRATEGY_SPEC"]["leave_command"]
    except (OSError, KeyError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--window", default="last100")
    args = ap.parse_args()
    root = os.path.abspath(args.root)

    rows = []
    for csv in sorted(glob.glob(os.path.join(root, "*", f"hypotheses_*_{args.window}.csv"))):
        m = re.match(rf"hypotheses_(.+)_vs_(.+)_{args.window}\.csv$", os.path.basename(csv))
        base, strat = m.groups()
        t = pd.read_csv(csv)
        t.insert(0, "strategy_cmd", command_of(os.path.join(os.path.dirname(root), strat)))
        t.insert(0, "strategy", strat)
        t.insert(0, "baseline", base)
        rows.append(t)
    if not rows:
        raise SystemExit(f"no */hypotheses_*_{args.window}.csv under {root}")
    table = pd.concat(rows, ignore_index=True)
    out = os.path.join(root, f"hypotheses_overview_{args.window}")
    table.to_csv(out + ".csv", index=False)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    pairs = table[["baseline", "strategy", "strategy_cmd"]].drop_duplicates().reset_index(drop=True)
    y = np.arange(len(pairs))[::-1]
    fig, axes = plt.subplots(1, len(PANELS), figsize=(5 * len(PANELS) + 1.5, 1.0 + 0.75 * len(pairs)),
                             facecolor=SURFACE, layout="constrained", sharey=True)
    for ax, (metric, scale, title, col) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="x", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axvline(0, color=INK2, lw=1)
        for yi, p in zip(y, pairs.itertuples()):
            r = table[(table.baseline == p.baseline) & (table.metric == metric)]
            if r.empty:
                ax.text(0, yi, "  n/a", va="center", fontsize=9, color=INK3)
                continue
            r = r.iloc[0]
            ax.plot([r.ci95_lo * scale, r.ci95_hi * scale], [yi, yi], color=col, lw=2, solid_capstyle="round")
            ax.plot(r["diff"] * scale, yi, "o", ms=8, color=col, mec=SURFACE, mew=2)
            p_txt = f"p={r.paired_t_p:.3f}" if r.paired_t_p >= 0.001 else "p<0.001"
            ax.annotate(f"{r['diff']*scale:+.1f}  {p_txt}  {int(r.n_pairs_up)}/5↑",
                        (r.ci95_hi * scale, yi), xytext=(6, 0), textcoords="offset points",
                        va="center", fontsize=8.5, color=INK2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("strategy − baseline, 95% CI")
        lo, hi = ax.get_xlim()
        ax.set_xlim(min(lo, 0), hi + 0.45 * (hi - min(lo, 0)))
    axes[0].set_yticks(y, [f"{p.baseline} vs {p.strategy}\n(strategy cmd {p.strategy_cmd})"
                           for p in pairs.itertuples()])
    axes[0].set_ylim(y.min() - 0.6, y.max() + 0.6)
    fig.suptitle(f"strategy vs baseline across pairs, window: {args.window}  "
                 "(p = paired t over 5 seed-matched trials; k/5↑ = pairs that rose)", fontsize=11)
    fig.savefig(out + ".png", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    keep = ["baseline", "strategy", "strategy_cmd", "metric", "baseline_mean", "strategy_mean", "diff",
            "ci95_lo", "ci95_hi", "n_pairs_up", "paired_t_p", "tost_min_margin"]
    pd.set_option("display.width", 250)
    print(table[table.metric.isin([p[0] for p in PANELS])][keep].round(4).to_string(index=False))
    print(f"\nwrote {out}.csv / .png")


if __name__ == "__main__":
    main()

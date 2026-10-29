"""
persistence_compare.py  -  partnership survival of two experiments, side by side.

The identity-free counterpart of group_lifetimes.py's Kaplan-Meier comparison.
Instead of following groups (identity, censoring, pooled non-independent
groups), it follows partnerships: two robots in the same group (size >= 3).
See group_persistence.py for the definitions.

    S(tau)  P(a partnership is unbroken for tau | it exists now)
    q(tau)  share of a robot's current groupmates it is grouped with tau later

Each trial gives one curve; the error band is the across-trial SEM. Both
experiments use the same initial conditions (trial k <-> trial k), so the
test at a fixed lag is a paired one over trials: paired t, with Wilcoxon
signed-rank alongside.

    python persistence_compare.py <exp_A> <exp_B> [--t0 200] [--max-lag 200] [--lag 30]

Writes compare_<A>_vs_<B>_persistence.png/.svg/.csv in the common parent.
"""

import argparse
import glob
import os
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

from group_persistence import BLOCK_S, block_comembership, overlap_curve, sampled_labels
from group_lifetimes import INK, INK2, GRID, SURFACE, SERIES

TINTS = ["#9ec5f4", "#f5b89c"]


def trial_curves(job):
    trial_dir, t0, max_lag = job
    t, C = block_comembership(*sampled_labels(trial_dir))
    lags, q, pair, cont, chance = overlap_curve(t, C, t0, max_lag)
    return os.path.basename(trial_dir), lags, q, pair, cont, chance


def run_experiment(exp_dir, t0, max_lag, jobs, n_trials=None):
    trials = sorted(d for d in glob.glob(os.path.join(exp_dir, "trial_*")) if os.path.isdir(d))[:n_trials]
    with Pool(min(jobs, len(trials))) as p:
        res = p.map(trial_curves, [(d, t0, max_lag) for d in trials])
    names = [r[0] for r in res]
    lags = res[0][1]
    stack = {k: np.array([r[i] for r in res]) for i, k in enumerate(("q", "pair", "cont"), start=2)}
    stack["chance"] = np.array([r[5] for r in res])
    return names, lags, stack


def paired_p(x, y):
    d = x - y
    return stats.ttest_1samp(d, 0).pvalue, stats.wilcoxon(x, y).pvalue


def fmt_p(p):
    return f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs=2)
    ap.add_argument("--t0", type=float, default=200.0, help="only partnerships present at t >= t0 (s) are followed")
    ap.add_argument("--max-lag", type=float, default=200.0)
    ap.add_argument("--lag", type=float, default=30.0, help="lag (s) of the headline comparison")
    ap.add_argument("--jobs", type=int, default=10)
    ap.add_argument("--trials", type=int, default=None, help="use only the first N trials of each experiment")
    ap.add_argument("--out", default=None, help="output directory (default: common parent of the two experiments)")
    args = ap.parse_args()

    exps = [os.path.abspath(e) for e in args.exps]
    names = [os.path.basename(e) for e in exps]
    data = [run_experiment(e, args.t0, args.max_lag, args.jobs, args.trials) for e in exps]
    if data[0][0] != data[1][0]:
        raise SystemExit(f"trial lists differ, cannot pair: {data[0][0]} vs {data[1][0]}")
    lags = data[0][1]
    k_head = int(round(args.lag / BLOCK_S))

    # ---- table: every metric at a few lags, paired over trials ----------------
    rows = []
    for key, label in (("cont", "S: partnership survival"), ("pair", "partnership present again"),
                       ("q", "q: groupmate overlap")):
        for L in sorted({10, args.lag, 60, 120} & set(lags[~np.isnan(data[0][2][key][0])])):
            k = int(round(L / BLOCK_S))
            x, y = data[0][2][key][:, k], data[1][2][key][:, k]
            pt, pw = paired_p(x, y)
            rows.append(dict(metric=label, lag_s=L, **{f"{names[0]}_mean": x.mean(), f"{names[0]}_sem": stats.sem(x),
                                                       f"{names[1]}_mean": y.mean(), f"{names[1]}_sem": stats.sem(y)},
                             diff_mean=(x - y).mean(), diff_sem=stats.sem(x - y), p_paired_t=pt, p_wilcoxon=pw))
    table = pd.DataFrame(rows)

    # ---- figure ------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2, 2, 1]})

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def curve_panel(ax, key, ylabel, title):
        style(ax)
        k = k_head
        x, y = data[0][2][key][:, k], data[1][2][key][:, k]
        p = paired_p(x, y)[0]
        for (n, (_, _, st)), c, tint in zip(zip(names, data), SERIES, TINTS):
            cur = st[key]
            ok = ~np.isnan(cur).any(0)
            for tr in cur:
                ax.plot(lags[ok], tr[ok], color=tint, lw=0.8, alpha=0.6)
            m, se = cur[:, ok].mean(0), stats.sem(cur[:, ok], axis=0)
            ax.fill_between(lags[ok], m - se, m + se, color=c, alpha=0.18, lw=0)
            ax.plot(lags[ok], m, color=c, lw=2, label=f"{n}: {m[k]:.2f} at {args.lag:g} s")
        ax.axvline(args.lag, color=INK2, lw=0.8, ls="--")
        ax.plot([], [], color=INK2, lw=0.8, alpha=0.6, label="single trial")
        ax.fill_between([], [], [], color=INK2, alpha=0.18, lw=0, label="mean ± SEM over trials")
        ax.legend(frameon=False, loc="upper right")
        ax.set_xlim(0, lags[~np.isnan(data[0][2][key][0])][-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("lag τ (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}   paired t at {args.lag:g} s: {fmt_p(p)}", loc="left", color=INK)

    curve_panel(axes[0], "cont", "fraction of partnerships unbroken after τ",
                "Partnership survival S(τ)")
    curve_panel(axes[1], "q", "groupmates still shared after τ",
                "Groupmate overlap q(τ)")

    # paired dots at the headline lag
    ax = axes[2]
    style(ax)
    x, y = data[0][2]["cont"][:, k_head], data[1][2]["cont"][:, k_head]
    for ya, yb in zip(x, y):
        ax.plot([0, 1], [ya, yb], color=GRID, lw=1, zorder=1)
    for i, (v, c) in enumerate(zip((x, y), SERIES)):
        ax.scatter(np.full(len(v), i), v, s=40, color=c, edgecolor=SURFACE, lw=1, zorder=2)
        ax.plot([i - 0.2, i + 0.2], [v.mean()] * 2, color=INK, lw=2, zorder=3)
        ax.annotate(f"{v.mean():.2f}", (i + 0.22, v.mean()), va="center", color=INK)
    pt, pw = paired_p(x, y)
    ax.set_xticks([0, 1], names)
    ax.set_xlim(-0.5, 1.6)
    ax.set_ylabel(f"S({args.lag:g} s), one dot per trial")
    ax.set_title(f"S({args.lag:g} s), paired\nt: {fmt_p(pt)}   Wilcoxon: {fmt_p(pw)}", loc="left", color=INK)

    fig.suptitle(f"{names[0]}  vs  {names[1]}  -  partnerships in groups of >= 3, d65, {BLOCK_S:g} s majority blocks, "
                 f"origins t >= {args.t0:g} s, trials paired by initial condition (n = {len(x)})",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()

    out_dir = args.out or os.path.commonpath(exps)
    out = os.path.join(out_dir, f"compare_{names[0]}_vs_{names[1]}_persistence")
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    table.to_csv(f"{out}.csv", index=False)
    with pd.option_context("display.width", 200):
        print(table.round(4).to_string(index=False))
    print(f"\nwrote {out}.png / .svg / .csv")


if __name__ == "__main__":
    main()

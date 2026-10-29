"""
stability_vs_size.py
--------------------
How group stability depends on group size, separately for every experiment in
a root directory (e.g. the N-scaling runs a862_10 ... a862_1000).

Same blocks and same J as group_persistence.py (d65 grouping, groups >= 3,
2 s majority blocks, origins t >= T0_S). For every robot i in a group at
origin t:

    s_i(t)        = |G_i(t)| + 1, the size of i's group at t
    J_i(t, tau)   = Jaccard overlap of i's groupmates at t and at t+tau
    R_i(t, tau)   = fraction of i's groupmates at t still with it at t+tau
                    (retention; ignores newcomers)

Size is read off the block at t alone, so no group tracking is needed: J
follows robots, not group identities. Samples are binned by s_i(t), giving
"if you are in a group of size s now, how much of it do you still share
tau later" -- q(tau) resolved by current group size. Every (robot, origin)
counts once, so a size-s group enters its bin s times per origin.

Per trial the dependence is summarised by the least-squares slope of J on
log2(s) over (robot, origin) samples -- the change of J per doubling of group
size -- and tested across trials (t test of the per-trial slopes against 0;
the trial is the sample, samples within a trial are not independent).

Outputs (in <root>):
    stability_vs_size.png / .svg
    stability_vs_size_bins.csv      J, R per experiment x lag x size bin (mean, SEM over trials)
    stability_vs_size_trials.csv    per-trial slopes and means
    stability_vs_size_stats.csv     per-experiment slope test

Usage:
    python stability_vs_size.py <root> [--lags 30 50] [--jobs 10]
"""
import argparse
import glob
import os
import re
import sys
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from group_persistence import BLOCK_S, MIN_SIZE, SAMPLE_HZ, T0_S, sampled_labels
from group_lifetimes import INK, INK2, GRID, SURFACE

EDGES = [3, 4, 5, 6, 8, 11, 16, 24, 36, 54, 81, 10**6]          # bin k = [EDGES[k], EDGES[k+1])
MIN_SAMPLES, MIN_TRIALS = 20, 3                                   # a bin is shown only above both


def bin_label(k):
    a, b = EDGES[k], EDGES[k + 1] - 1
    return f"{a}" if a == b else (f"{a}+" if b > 10**5 else f"{a}-{b}")


def bin_center(k):
    a, b = EDGES[k], min(EDGES[k + 1] - 1, 120)
    return np.sqrt(a * b)


def n_of(name):
    m = re.search(r"_(\d+)$", name)
    return int(m.group(1)) if m else name


def block_labels(t, lab):
    """-> (block start times s, (B, N) group label per block, -1 outside groups >= MIN_SIZE).

    Same blocks as group_persistence.block_comembership, kept as labels instead
    of a (B, N, N) matrix -- 250 MB per trial at N = 1000.
    """
    N = lab.shape[1]
    per = int(round(BLOCK_S * SAMPLE_HZ))
    nb = len(t) // per
    out = np.full((nb, N), -1, np.int32)
    for b in range(nb):
        L = lab[b * per:(b + 1) * per]
        co = (L[:, :, None] == L[:, None, :]).sum(0) * 2 > per
        _, c = connected_components(csr_matrix(co), directed=False)
        big = np.bincount(c)[c] >= MIN_SIZE
        out[b, big] = c[big]
    return t[::per][:nb], out


def overlap_by_robot(a, b):
    """Block labels a (origin) and b (origin + tau) -> (size, J, R) of every robot grouped in a."""
    ia = a >= 0
    s = np.bincount(a[ia])[a[ia]]                      # own group size at the origin
    bb = b[ia]
    inb = bb >= 0
    mates_b = np.zeros(len(s), np.int64)
    mates_b[inb] = np.bincount(b[b >= 0])[bb[inb]] - 1
    # robots sharing both i's group at a and i's group at b, i included
    key = a[ia].astype(np.int64) * (b.max() + 2) + bb
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inter = np.where(inb, cnt[inv] - 1, 0)
    union = (s - 1) + mates_b - inter
    return np.c_[s, inter / union, inter / (s - 1)]


def trial_samples(trial_dir, lags):
    """-> (trial name, {lag: DataFrame(size, J, R)}) over every (robot, origin) in a group >= 3."""
    t, B = block_labels(*sampled_labels(trial_dir))
    o0 = np.searchsorted(t, T0_S)
    out = {}
    for L in lags:
        k = int(round(L / BLOCK_S))
        rows = [overlap_by_robot(B[o], B[o + k]) for o in range(o0, len(B) - k)]
        out[L] = pd.DataFrame(np.vstack(rows) if rows else np.empty((0, 3)), columns=["size", "J", "R"])
    return os.path.basename(os.path.normpath(trial_dir)), out


def slope(x, y):
    return np.polyfit(x, y, 1)[0] if len(x) > 2 and np.ptp(x) > 0 else np.nan


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--lags", type=int, nargs="+", default=[30, 50], help="lags tau (s)")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    exps = sorted((d for d in glob.glob(os.path.join(args.root, "*")) if glob.glob(os.path.join(d, "trial_*"))),
                  key=lambda d: n_of(os.path.basename(d)))
    all_bins, all_trials = [], []
    for exp in exps:
        b, t = experiment_rows(exp, args.lags, args.jobs)
        all_bins.append(b)
        all_trials.append(t)
    bins, trials = pd.concat(all_bins), pd.concat(all_trials)
    finish(bins, trials, args)


def experiment_rows(exp, lags, jobs):
    """Per-trial and per-bin rows of one experiment, cached in the experiment directory
    so that an interrupted run resumes where it stopped."""
    name = os.path.basename(exp)
    tag = "-".join(str(L) for L in lags)
    cache = [os.path.join(exp, f"_stability_vs_size_{kind}_lag{tag}.csv") for kind in ("bins", "trials")]
    if all(os.path.exists(c) for c in cache):
        print(f"{name}: cached", flush=True)
        return pd.read_csv(cache[0]), pd.read_csv(cache[1])
    tdirs = sorted(d for d in glob.glob(os.path.join(exp, "trial_*")) if os.path.isdir(d))
    with Pool(min(jobs, len(tdirs))) as p:
        res = p.map(partial(trial_samples, lags=lags), tdirs)
    print(f"{name}: {len(res)} trials", flush=True)
    bins, trials = [], []
    for trial, per_lag in res:
        for L, df in per_lag.items():
            x = np.log2(df["size"])
            trials.append(dict(exp=name, N=n_of(name), trial=trial, lag_s=L, n_samples=len(df),
                               mean_size=df["size"].mean(), q=df.J.mean(), mean_R=df.R.mean(),
                               slope_J=slope(x, df.J), slope_R=slope(x, df.R),
                               rho_J=stats.spearmanr(df["size"], df.J)[0]
                               if len(x) > 2 and np.ptp(x) > 0 and np.ptp(df.J) > 0 else np.nan))
            for k, g in df.groupby(np.digitize(df["size"], EDGES) - 1):
                bins.append(dict(exp=name, N=n_of(name), trial=trial, lag_s=L, size_bin=bin_label(k),
                                 size_center=bin_center(k), n=len(g), J=g.J.mean(), R=g.R.mean()))
    bins, trials = pd.DataFrame(bins), pd.DataFrame(trials)
    bins.to_csv(cache[0], index=False)
    trials.to_csv(cache[1], index=False)
    return bins, trials


def finish(bins, trials, args):
    agg =(bins[bins.n >= MIN_SAMPLES]
           .groupby(["exp", "N", "lag_s", "size_bin", "size_center"])
           .agg(n_trials=("J", "size"), n_samples=("n", "sum"), J_mean=("J", "mean"), J_sem=("J", "sem"),
                R_mean=("R", "mean"), R_sem=("R", "sem"))
           .reset_index().sort_values(["lag_s", "N", "size_center"]))
    agg = agg[agg.n_trials >= MIN_TRIALS]

    srows = []
    for (name, N, L), g in trials.groupby(["exp", "N", "lag_s"]):
        g = g.dropna(subset=["slope_J"])
        if len(g) < 3:
            continue
        srows.append(dict(exp=name, N=N, lag_s=L, n_trials=len(g), mean_size=g.mean_size.mean(), q=g.q.mean(),
                          slope_J=g.slope_J.mean(), slope_J_sem=g.slope_J.sem(),
                          p_slope_J=stats.ttest_1samp(g.slope_J, 0).pvalue,
                          n_trials_slope_J_neg=int((g.slope_J < 0).sum()),
                          slope_R=g.slope_R.mean(), slope_R_sem=g.slope_R.sem(),
                          p_slope_R=stats.ttest_1samp(g.slope_R, 0).pvalue, rho_J=g.rho_J.mean()))
    st = pd.DataFrame(srows).sort_values(["lag_s", "N"])

    out = os.path.join(args.root, "stability_vs_size")
    agg.to_csv(f"{out}_bins.csv", index=False)
    trials.to_csv(f"{out}_trials.csv", index=False)
    st.to_csv(f"{out}_stats.csv", index=False)
    pd.set_option("display.width", 220)
    print(st.round(4).to_string(index=False))
    plot(agg, st, args.lags, out)


def plot(agg, st, lags, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
                         "svg.fonttype": "none"})
    Ns = sorted(agg.N.unique())
    ramp = LinearSegmentedColormap.from_list("blue", ["#9ec5f0", "#2a78d6", "#0b2f61"])
    color = {N: ramp(i / max(1, len(Ns) - 1)) for i, N in enumerate(Ns)}
    fig, axes = plt.subplots(1, len(lags) + 1, figsize=(5.2 * (len(lags) + 1), 4.8), facecolor=SURFACE)

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    for ax, L in zip(axes, lags):
        style(ax)
        for N in Ns:
            g = agg[(agg.N == N) & (agg.lag_s == L)]
            ax.plot(g.size_center, g.J_mean, "-o", color=color[N], lw=2, ms=4, label=f"N = {N}")
            ax.fill_between(g.size_center, g.J_mean - g.J_sem, g.J_mean + g.J_sem, color=color[N], alpha=0.15, lw=0)
        ax.set_xscale("log", base=2)
        smax = agg.size_center.max()
        ticks = [s for s in (3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96) if s <= smax * 1.2]
        ax.set_xlim(2.8, smax * 1.15)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("group size s at origin")
        ax.set_ylabel(f"groupmates still shared after {L} s")
        ax.set_ylim(0, 1)
        ax.set_title(f"q({L} s) by group size, mean ± SEM over trials", loc="left")
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="lower left")

    ax = axes[-1]
    style(ax)
    ax.axhline(0, color=INK2, lw=0.8)
    # one row of significance marks per lag along the bottom, so they never collide with the bars
    lo = (st.slope_J - st.slope_J_sem).min()
    for i, (L, c, fmt) in enumerate(zip(lags, ["#2a78d6", "#eb6834", "#1baf7a"], ["o-", "s--", "^:"])):
        s = st[st.lag_s == L]
        ax.errorbar(s.N, s.slope_J, yerr=s.slope_J_sem, fmt=fmt, color=c, lw=2, ms=5, capsize=3,
                    label=f"q({L} s)")
        for _, r in s.iterrows():
            mark = ("***" if r.p_slope_J < 0.001 else "**" if r.p_slope_J < 0.01
                    else "*" if r.p_slope_J < 0.05 else "n.s.")
            ax.annotate(mark, (r.N, lo), textcoords="offset points", xytext=(0, -14 - 11 * i),
                        ha="center", fontsize=7, color=c)
    ax.set_ylim(bottom=lo - 0.035 - 0.02 * len(lags))
    Nst = sorted(st.N.unique())
    ax.set_xscale("log")
    ax.set_xticks(Nst)
    ax.set_xticklabels([str(n) for n in Nst], fontsize=8)
    ax.minorticks_off()
    ax.set_xlabel("number of robots N")
    ax.set_ylabel("change in J per doubling of group size")
    ax.set_title("Size dependence per experiment\n(slope of J on log2 s, t test of trial slopes vs 0)",
                 loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(f"Group stability vs group size  -  d65 groups of >= 3, 2 s majority blocks, "
                 f"origins t >= {T0_S:g} s", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    for ext in ("png", "svg"):
        fig.savefig(f"{out}.{ext}", dpi=150, facecolor=SURFACE)
    print(f"wrote {out}.png / .svg / _bins.csv / _trials.csv / _stats.csv")


if __name__ == "__main__":
    main()

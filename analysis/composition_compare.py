"""
composition_compare.py
----------------------
Compare group-size composition, mean group size and largest group size across
two or more experiments with the same N.

Reads the per-trial grouping caches written by batch_group_plots.py
(trial_XXXX_groups_d<max_dist>.csv, singletons included) and uses the same
definitions as that script:

    share of bin b at frame t   fraction of robots whose group size falls in b
    mean group size at frame t  sum(s^2) / N -- "grab a random robot, how big
                                is its group" (singletons count as size 1)
    largest group at frame t    size of the biggest group in that frame

Bins come from batch_group_plots.size_bins(N), e.g. 1-2 / 3-5 / 6-13 / 14+ at
N=100. Each trial is reduced to its time average over t >= --t0 (default 50 s,
past the initial clumping transient), so every statistic has one value per
trial; error bars are +-1 SD across trials.

The first experiment is the reference. Every other one is compared with it,
trial by trial (trials are paired by name, i.e. by seed and initial
condition):
  - each scalar (share per bin, mean size, largest size): Wilcoxon signed-rank
  - the composition as a whole: centred log-ratio (CLR) transform per trial,
    then an exact sign-flip permutation test on the norm of the mean paired
    CLR difference (all 2^n flips for n pairs)

Usage:
    python composition_compare.py <ref_exp> <exp2> [<exp3> ...] [--t0 50] [--out-dir DIR] [--name NAME]
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from batch_group_plots import size_bins

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


def trial_summary(groups_csv, n_robots, bins, fps, t0):
    """Time averages over t >= t0 of the per-frame share per bin, mean size and largest size."""
    df = pd.read_csv(groups_csv, usecols=["Step", "size"])
    T = df.Step.max() + 1
    keep = np.arange(T) / fps >= t0
    out = {}
    for (lo, hi), label in zip(*bins):
        sub = df[(df["size"] >= lo) & (df["size"] <= hi)]
        out[f"share_{label}"] = (np.bincount(sub.Step, weights=sub["size"], minlength=T) / n_robots)[keep].mean()
    out["mean_size"] = (np.bincount(df.Step, weights=df["size"] ** 2, minlength=T) / n_robots)[keep].mean()
    largest = df.groupby("Step")["size"].max().reindex(range(T)).to_numpy()
    out["largest_size"] = largest[keep].mean()
    return out


def bins_from_edges(edges, n):
    """Same construction as batch_group_plots.size_bins, for caller-chosen edges."""
    bins, labels, lo = [], [], 1
    for e in edges:
        bins.append((lo, e))
        labels.append(str(lo) if lo == e else f"{lo}-{e}")
        lo = e + 1
    bins.append((lo, max(lo, n)))
    labels.append(f"{lo}+")
    return bins, labels


def load_experiment(exp_dir, t0, max_dist, edges=None):
    cfg = json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))
    n, fps = cfg["N_SMARTICLES"], cfg.get("RENDER_FPS_HEADLESS", 60)
    bins, labels = bins_from_edges(edges, n) if edges else size_bins(n)
    rows = []
    for groups_csv in sorted(glob.glob(os.path.join(exp_dir, "trial_*", f"*_groups_d{max_dist}.csv"))):
        rows.append(dict(trial=os.path.basename(os.path.dirname(groups_csv)),
                         **trial_summary(groups_csv, n, (bins, labels), fps, t0)))
    if not rows:
        raise SystemExit(f"{exp_dir}: no trial_*/*_groups_d{max_dist}.csv -- run batch_group_plots.py on it first")
    return os.path.basename(exp_dir), n, labels, pd.DataFrame(rows)


def clr_permutation_test(ca, cb):
    """Exact sign-flip test on ||mean(CLR(a) - CLR(b))|| over paired trials."""
    def clr(c):
        lc = np.log(np.maximum(c, 1e-6))     # a bin that is empty for a whole trial
        return lc - lc.mean(axis=1, keepdims=True)
    d = clr(ca) - clr(cb)
    obs = np.linalg.norm(d.mean(axis=0))
    null = [np.linalg.norm((np.array(s)[:, None] * d).mean(axis=0))
            for s in itertools.product([1, -1], repeat=len(d))]
    return obs, np.mean(np.array(null) >= obs - 1e-12)


def fmt_p(p):
    return "n/a" if np.isnan(p) else f"{p:.3f}" if p >= 0.001 else "<0.001"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+", help="experiment directories; the first is the reference")
    ap.add_argument("--t0", type=float, default=50.0, help="ignore the first t0 seconds (default 50)")
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--out-dir", default=None, help="default: the experiments' common parent directory")
    ap.add_argument("--name", default=None, help="output file stem (default: compare_<exp1>_vs_..._composition_t<t0>)")
    ap.add_argument("--formats", nargs="+", default=["png"], choices=["png", "svg", "pdf"],
                    help="figure formats (default png); svg/pdf keep text editable")
    ap.add_argument("--edges", nargs="+", type=int, default=None,
                    help="upper edges of the size bins, e.g. '2 5' -> 1-2 / 3-5 / 6+ (default: size_bins(N))")
    ap.add_argument("--labels", nargs="+", default=None,
                    help="display name for each experiment, in the same order (default: directory names)")
    args = ap.parse_args()
    if len(args.exps) < 2:
        raise SystemExit("need at least two experiments")
    if args.labels and len(args.labels) != len(args.exps):
        raise SystemExit(f"--labels has {len(args.labels)} names for {len(args.exps)} experiments")

    exps = [load_experiment(os.path.abspath(e), args.t0, args.max_dist, args.edges) for e in args.exps]
    dir_names = {}
    if args.labels:
        dir_names = {lab: e[0] for lab, e in zip(args.labels, exps)}
        exps = [(lab,) + e[1:] for lab, e in zip(args.labels, exps)]
    ns = {e[1] for e in exps}
    if len(ns) > 1:
        raise SystemExit(f"population sizes differ ({sorted(ns)}); compare same-N experiments")
    n, labels = exps[0][1], exps[0][2]
    ref_name, _, _, ref = exps[0]
    for name, _, _, t in exps[1:]:
        if list(t.trial) != list(ref.trial):
            raise SystemExit(f"{name}: trial names differ from {ref_name}; this script assumes seed-paired trials")
    share_cols = [f"share_{l}" for l in labels]
    metrics = share_cols + ["mean_size", "largest_size"]

    # ---- numbers ------------------------------------------------------------
    rows = []
    for name, _, _, t in exps:
        for col in metrics:
            x = t[col].to_numpy()
            row = {"exp": name, "metric": col, "mean": x.mean(), "sd": x.std(ddof=1),
                   "sem": x.std(ddof=1) / np.sqrt(len(x))}
            if name != ref_name:
                y = ref[col].to_numpy()
                d = x - y
                ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
                row.update({f"diff_vs_{ref_name}": d.mean(), "diff_ci95_lo": d.mean() - ci,
                            "diff_ci95_hi": d.mean() + ci,
                            "wilcoxon_p": stats.wilcoxon(x, y).pvalue if np.any(d != 0) else np.nan})
            rows.append(row)
        if name != ref_name:
            stat, p = clr_permutation_test(t[share_cols].to_numpy(), ref[share_cols].to_numpy())
            rows.append({"exp": name, "metric": "composition (CLR, all bins)",
                         f"diff_vs_{ref_name}": stat, "perm_p": p})
    table = pd.DataFrame(rows)
    if dir_names:
        table.insert(1, "exp_dir", table.exp.map(dir_names))

    out_dir = args.out_dir or os.path.commonpath([os.path.abspath(e) for e in args.exps])
    stem = args.name or "compare_" + "_vs_".join(e[0] for e in exps) + f"_composition_t{args.t0:g}"
    out = os.path.join(out_dir, stem)
    table.to_csv(out + ".csv", index=False)
    pd.concat([t.assign(exp=name, exp_dir=dir_names.get(name, name)) for name, _, _, t in exps]).to_csv(
        out + "_trials.csv", index=False)

    def p_of(name, metric, col="wilcoxon_p"):
        r = table[(table.exp == name) & (table.metric == metric)]
        return r[col].iloc[0] if len(r) else np.nan

    # ---- figure -------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    k = len(exps)
    fig, axes = plt.subplots(1, 3, figsize=(7 + 2.2 * k + 2 * 1.2 * k, 5.6), facecolor=SURFACE, layout="constrained",
                             gridspec_kw={"width_ratios": [2.2 * k, 1.2 * k, 1.2 * k]})

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    # composition: grouped bars, whisker = SD, p vs reference above each non-reference bar
    ax = axes[0]
    style(ax)
    x = np.arange(len(labels))
    w = 0.8 / k
    tops = []
    for j, ((name, _, _, t), c) in enumerate(zip(exps, SERIES)):
        m = t[share_cols].mean().to_numpy() * 100
        sd = t[share_cols].std(ddof=1).to_numpy() * 100
        xs = x + (j - (k - 1) / 2) * w
        label = name + (" (reference)" if j == 0 else f"   CLR p = {fmt_p(p_of(name, 'composition (CLR, all bins)', 'perm_p'))}")
        ax.bar(xs, m, w * 0.9, color=c, label=label, yerr=sd, error_kw=dict(ecolor=INK, lw=1, capsize=2.5))
        tops.append((xs, m + sd, name))
    ymax = max(tt.max() for _, tt, _ in tops)
    for xs, hi, name in tops[1:]:
        for i, l in enumerate(labels):
            ax.annotate(fmt_p(p_of(name, f"share_{l}")), (xs[i], hi[i] + ymax * 0.015),
                        ha="center", va="bottom", fontsize=7.5, color=INK2, rotation=90)
    ax.set_ylim(0, ymax * 1.5)
    ax.set_xticks(x, labels)
    ax.set_xlabel("group size")
    ax.set_ylabel(f"share of robots (%), time average t >= {args.t0:g} s")
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Composition  (bar = mean, whisker = SD; numbers = paired p vs reference)",
                 loc="left", color=INK)

    # mean / largest group size: dots per trial, black = mean +- SD
    for ax, col, title in ((axes[1], "mean_size", "Mean group size (robot's view)"),
                           (axes[2], "largest_size", "Largest group size")):
        style(ax)
        for trial_vals in zip(*[t[col] for _, _, _, t in exps]):
            ax.plot(range(k), trial_vals, color=GRID, lw=1, zorder=1)
        hi_all = max(t[col].max() for _, _, _, t in exps)
        for j, ((name, _, _, t), c) in enumerate(zip(exps, SERIES)):
            m, sd = t[col].mean(), t[col].std(ddof=1)
            ax.scatter(np.full(len(t), j - 0.12), t[col], s=30, color=c, edgecolor=SURFACE, lw=1, zorder=2)
            ax.errorbar(j + 0.14, m, yerr=sd, fmt="o", color=INK, ms=5, capsize=4, lw=1.5, zorder=3)
            ax.annotate(f"{m:.2f}\n±{sd:.2f}", (j + 0.2, m), va="center", fontsize=8.5, color=INK)
            if j:
                ax.annotate(f"p={fmt_p(p_of(name, col))}", (j, hi_all * 1.08), ha="center", fontsize=8.5,
                            color=INK2)
        ax.set_xticks(range(k), [e[0] for e in exps], rotation=20, ha="right")
        ax.set_xlim(-0.5, k - 0.3)
        ax.set_ylim(0, hi_all * 1.18)
        ax.set_ylabel("robots")
        ax.set_title(f"{title}\n(dot = trial, black = mean ± SD)", loc="left", color=INK)

    fig.suptitle(f"N={n}, d{args.max_dist}, {len(ref)} seed-paired trials per experiment, time averages over "
                 f"t >= {args.t0:g} s.  p values: paired Wilcoxon vs {ref_name}", x=0.01, ha="left",
                 color=INK, fontsize=11)
    for fmt in args.formats:
        fig.savefig(f"{out}.{fmt}", dpi=150, facecolor=SURFACE)
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(table.round(4).to_string(index=False))
    print(f"\nwrote {out}.{{{','.join(args.formats)}}} / .csv / _trials.csv")


if __name__ == "__main__":
    main()

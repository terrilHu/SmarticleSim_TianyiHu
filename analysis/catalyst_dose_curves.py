"""
catalyst_dose_curves.py
-----------------------
Change in the share of robots in groups >= --large (the "size 6+" bar of
composition_delta_bars.py) against the amount of catalyst: one curve per
baseline, one chart per --groups list. The baseline itself is the 0% point, so
each curve starts at (0, 0).

Experiments are found by name, <root>/<base>_xend_*, so a new amount appears on
the chart as soon as its folder and grouping caches exist. The amount comes from
config_snapshot.json (2 x n_per_end / N_SMARTICLES), not from the folder name.
The x axis is fixed at 0-100% whatever data exists yet.

For each amount: mean paired difference (strategy - baseline, trials paired by
name), 95% t-interval, paired t-test, over the last --last seconds of the run.

Per-trial shares are kept in <out-dir>/catalyst_dose_trials.csv. A share not in
that file is taken from <root>/hypotheses_compare*/*/*_last<L>_trials.csv if
strategy_hypotheses.py has already computed it, and otherwise from the trial's
groups_d<d>.csv. Use --force after re-running a trial: it recomputes every share
from the groups caches.

Output (in --out-dir):
    catalyst_dose_<bases>_last<L>.png / .svg / .csv   one set per --groups

Usage:
    python catalyst_dose_curves.py <root> --baseline-root <dir> --out-dir DIR
           --groups a016 a034 a062 --groups a816 a834 a862 [--ylim LO HI]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from strategy_hypotheses import trial_metrics

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
PAIR_COLORS = ["#1baf7a", "#eda100", "#6250d6"]     # same pair colours as composition_delta_bars.py
X_TICKS = [0, 12, 24, 36, 50, 75, 100]


def config(exp_dir):
    return json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))


def amount_pct(cfg):
    roles = cfg["STRATEGY_SPEC"]["roles"]
    return 100.0 * sum(2 * r["n_per_end"] for r in roles) / cfg["N_SMARTICLES"]


def compute(exp_dir, large, last, max_dist):
    cfg = config(exp_dir)
    n, fps = cfg["N_SMARTICLES"], cfg.get("RENDER_FPS_HEADLESS", 60)
    rows = []
    for csv in sorted(glob.glob(os.path.join(exp_dir, "trial_*", f"*_groups_d{max_dist}.csv"))):
        m = trial_metrics(csv, n, fps, None, large, 3, last)[0]
        rows.append(dict(exp=os.path.basename(exp_dir), trial=os.path.basename(os.path.dirname(csv)),
                         share=m[f"share_ge{large}"]))
        print(f"  {rows[-1]['exp']}/{rows[-1]['trial']} done", flush=True)
    return rows


class Shares:
    """Per-trial shares by experiment: cache file, then existing comparison tables, then the groups caches."""

    def __init__(self, root, out_dir, large, last, max_dist, force):
        self.large, self.last, self.max_dist = large, last, max_dist
        self.path = os.path.join(out_dir, "catalyst_dose_trials.csv")
        self.table = pd.DataFrame(columns=["exp", "trial", "share"])
        if force:
            return
        tabs = [pd.read_csv(f)[["exp", "trial", f"share_ge{large}"]].rename(columns={f"share_ge{large}": "share"})
                for f in glob.glob(os.path.join(root, "hypotheses_compare*", "*", f"*_last{last:g}_trials.csv"))]
        if os.path.exists(self.path):
            tabs.append(pd.read_csv(self.path))
        if tabs:
            self.table = pd.concat(tabs, ignore_index=True).drop_duplicates(["exp", "trial"], keep="last")

    def get(self, exp_dir):
        name = os.path.basename(exp_dir)
        t = self.table[self.table.exp == name]
        if t.empty:
            t = pd.DataFrame(compute(exp_dir, self.large, self.last, self.max_dist))
            if t.empty:
                raise SystemExit(f"{exp_dir}: no groups_d{self.max_dist}.csv -- run batch_group_plots.py first")
            self.table = pd.concat([self.table, t], ignore_index=True)
            self.table.to_csv(self.path, index=False)
        return t.set_index("trial").share


def curve(base, root, baseline_root, shares):
    """One row per amount of catalyst, including the baseline at 0."""
    b = shares.get(os.path.join(baseline_root, base))
    rows = [dict(baseline=base, strategy=base, amount_pct=0.0, n_catalysts=0, mean_pp=0.0, ci95_lo=0.0, ci95_hi=0.0,
                 k_up=None, n=len(b), paired_t_p=None, baseline_share_pct=b.mean() * 100,
                 strategy_share_pct=b.mean() * 100)]
    cmd = None
    for d in glob.glob(os.path.join(root, f"{base}_xend_*")):
        if not os.path.isdir(d):
            continue
        cfg = config(d)
        cmd = cfg["STRATEGY_SPEC"]["leave_command"]
        s = shares.get(d)
        if set(s.index) != set(b.index):
            raise SystemExit(f"{d}: trials {sorted(s.index)} do not match baseline {sorted(b.index)}")
        v = ((s - b.reindex(s.index)) * 100).to_numpy()
        m, ci = v.mean(), stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
        rows.append(dict(baseline=base, strategy=os.path.basename(d), amount_pct=amount_pct(cfg),
                         n_catalysts=round(amount_pct(cfg) * cfg["N_SMARTICLES"] / 100), mean_pp=m,
                         ci95_lo=m - ci, ci95_hi=m + ci, k_up=int((v > 0).sum()), n=len(v),
                         paired_t_p=stats.ttest_1samp(v, 0).pvalue, baseline_share_pct=b.mean() * 100,
                         strategy_share_pct=s.mean() * 100))
    return cmd, pd.DataFrame(rows).sort_values("amount_pct", ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding the <base>_xend_* experiments")
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--groups", nargs="+", action="append", required=True, help="baselines of one chart; repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--large", type=int, default=6)
    ap.add_argument("--last", type=float, default=100.0)
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--n", type=int, default=500, help="N_SMARTICLES, for the robot counts on the x axis")
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"))
    ap.add_argument("--force", action="store_true", help="recompute every share from the groups caches")
    ap.add_argument("--bare", action="store_true",
                    help="curves, whiskers and the y axis only (x ticks unlabelled; no title, legend, curve or x "
                         "labels); written as <name>_bare")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    shares = Shares(args.root, args.out_dir, args.large, args.last, args.max_dist, args.force)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

    for bases in args.groups:
        fig, ax = plt.subplots(figsize=(9.0, 6.2), facecolor=SURFACE, layout="constrained")
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.axhline(0, color=INK2, lw=1)
        rows = []
        for j, (base, col) in enumerate(zip(bases, PAIR_COLORS)):
            cmd, c = curve(base, args.root, args.baseline_root, shares)
            rows.append(c)
            # nudge each curve sideways so whiskers at the same amount stay apart; all start at the same origin
            x = c.amount_pct.to_numpy() + np.where(c.amount_pct > 0, (j - (len(bases) - 1) / 2) * 0.8, 0)
            ax.errorbar(x, c.mean_pp, yerr=[c.mean_pp - c.ci95_lo, c.ci95_hi - c.mean_pp], color=col, lw=2,
                        elinewidth=1.3, capsize=4, marker="o", ms=8, mec=SURFACE, mew=1.5, zorder=3 + j,
                        label=f"{base}  (strategy cmd {cmd})")
            if not args.bare:
                ax.text(x[-1] + 1.6, c.mean_pp.iloc[-1], base, va="center", fontsize=13, color=INK)
        ax.set_xlim(-3, 105)
        ax.set_ylabel(f"Δ share of robots in size {args.large}+ (pp)\nstrategy − baseline")
        if args.ylim:
            ax.set_ylim(*args.ylim)
        name = f"catalyst_dose_{'_'.join(bases)}_last{args.last:g}"
        if args.bare:
            ax.set_xticks(X_TICKS, [""] * len(X_TICKS))
            name += "_bare"
        else:
            ax.set_xticks(X_TICKS, [f"{p}%\n({round(p * args.n / 100)})" for p in X_TICKS])
            ax.tick_params(axis="x", length=4, labelsize=13)
            ax.set_xlabel(f"catalysts: % of robots (number, of {args.n})")
            n = int(rows[0].n.iloc[0])
            ax.set_title(f"Size {args.large}+ gain against catalyst amount, last {args.last:g} s\n"
                         f"point = mean paired difference over {n} seed-matched trials, whisker = 95% CI; "
                         "0% = baseline", loc="left", fontsize=12)
            fig.legend(frameon=False, loc="outside lower center", ncol=len(bases), fontsize=13, handlelength=1.6,
                       columnspacing=1.5)
        out = os.path.join(args.out_dir, name)
        fig.savefig(out + ".png", dpi=150, facecolor=SURFACE)
        fig.savefig(out + ".svg", facecolor=SURFACE)
        plt.close(fig)
        table = pd.concat(rows, ignore_index=True)
        table.to_csv(out + ".csv", index=False)
        pd.set_option("display.width", 200)
        print(table.round(3).to_string(index=False))
        print(f"wrote {out}.png / .svg / .csv", flush=True)

    # keep everything used, so a later run needs only the new amounts
    shares.table.to_csv(shares.path, index=False)


if __name__ == "__main__":
    main()

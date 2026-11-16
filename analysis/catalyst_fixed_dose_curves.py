"""
catalyst_fixed_dose_curves.py
-----------------------------
catalyst_dose_curves.py for fixed-catalyst runs: the catalysts are a fixed share of
COMMAND_ARRAY (e.g. 60 x -852 among 440 x 816), not robots recruited by a strategy.

Experiments are found as <root>/<base>_<pct>-<cmd>; the amount is the share of
COMMAND_ARRAY entries that are not the base command, read from config_snapshot.json.
A 100% run has no base command left, so it is the same experiment for every base:
a base without its own <base>_100-* folder uses any <root>/*_100-* folder.

Two charts per --groups list, in the style of catalyst_dose_curves.py:
    catalyst_fixed_dose_<bases>_last<L>        Δ share of robots in size >= --large,
                                               strategy - baseline, paired by trial name
    catalyst_fixed_level_<bases>_last<L>       the share itself (mean over trials, 95% CI);
                                               0% = each baseline, 100% = the shared all-catalyst run
--bare adds a copy of each with only the curves, whiskers and y axis (<name>_bare).
--kinds picks which of the two charts to draw.

--size-ge S plots the share of robots in groups of size >= S instead of >= --large, with
its own cache; file names then carry _ge<S>.

Per-trial shares are cached in <out-dir>/catalyst_dose_trials.csv (as in catalyst_dose_curves.py),
or catalyst_ge<S>_trials.csv with --size-ge.

Usage:
    python catalyst_fixed_dose_curves.py <root> --baseline-root <dir> --out-dir DIR
           --groups a816 a834 a862 [--bare]
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import pandas as pd
from scipy import stats

from catalyst_dose_curves import GRID, INK, INK2, PAIR_COLORS, SURFACE, X_TICKS, Shares
from strategy_hypotheses import trial_metrics


def config(exp_dir):
    return json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))


def catalysts(cfg, base_cmd):
    """(amount %, number of catalysts, catalyst commands) from COMMAND_ARRAY."""
    c = Counter(cfg["COMMAND_ARRAY"])
    n_cat = sum(v for k, v in c.items() if k != base_cmd)
    return 100.0 * n_cat / cfg["N_SMARTICLES"], n_cat, sorted(k for k in c if k != base_cmd)


def experiments(root, base):
    """Fixed-catalyst experiment dirs for one base, plus the shared 100% run when the base has none."""
    pat = re.compile(rf"^{re.escape(base)}_(\d+)-(-?\d+)$")
    dirs = [d for d in glob.glob(os.path.join(root, f"{base}_*")) if os.path.isdir(d) and pat.match(os.path.basename(d))]
    if not any(pat.match(os.path.basename(d)).group(1) == "100" for d in dirs):
        full = sorted(d for d in glob.glob(os.path.join(root, "*_100-*")) if os.path.isdir(d))
        dirs += full[:1]
    return dirs


class SizeShares:
    """Per-trial share of robots in groups of size >= size, cached like catalyst_dose_curves.Shares."""

    def __init__(self, out_dir, size, last, max_dist, force):
        self.size, self.last, self.max_dist = size, last, max_dist
        self.path = os.path.join(out_dir, f"catalyst_ge{size}_trials.csv")
        self.table = pd.DataFrame(columns=["exp", "trial", "share"])
        if not force and os.path.exists(self.path):
            self.table = pd.read_csv(self.path)

    def get(self, exp_dir):
        name = os.path.basename(exp_dir)
        t = self.table[self.table.exp == name]
        if t.empty:
            cfg = config(exp_dir)
            n, fps = cfg["N_SMARTICLES"], cfg.get("RENDER_FPS_HEADLESS", 60)
            rows = []
            for csv in sorted(glob.glob(os.path.join(exp_dir, "trial_*", f"*_groups_d{self.max_dist}.csv"))):
                m = trial_metrics(csv, n, fps, None, self.size, self.size, self.last)[0]
                rows.append(dict(exp=name, trial=os.path.basename(os.path.dirname(csv)),
                                 share=m[f"share_ge{self.size}"]))
                print(f"  {name}/{rows[-1]['trial']} done", flush=True)
            if not rows:
                raise SystemExit(f"{exp_dir}: no groups_d{self.max_dist}.csv -- run batch_group_plots.py first")
            t = pd.DataFrame(rows)
            self.table = pd.concat([self.table, t], ignore_index=True)
            self.table.to_csv(self.path, index=False)
        return t.set_index("trial").share


def ci95(v):
    v = np.asarray(v, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


def curve(base, root, baseline_root, shares):
    bdir = os.path.join(baseline_root, base)
    base_cmd = Counter(config(bdir)["COMMAND_ARRAY"]).most_common(1)[0][0]
    b = shares.get(bdir).astype(float)
    bh = ci95(b * 100)
    rows = [dict(baseline=base, strategy=base, amount_pct=0.0, n_catalysts=0, mean_pp=0.0, ci95_lo=0.0, ci95_hi=0.0,
                 k_up=None, n=len(b), paired_t_p=None, baseline_share_pct=b.mean() * 100,
                 strategy_share_pct=b.mean() * 100, strategy_ci95_lo=b.mean() * 100 - bh,
                 strategy_ci95_hi=b.mean() * 100 + bh)]
    cat = set()
    for d in experiments(root, base):
        cfg = config(d)
        pct, n_cat, cmds = catalysts(cfg, base_cmd)
        cat.update(cmds)
        s = shares.get(d).astype(float)
        if set(s.index) != set(b.index):
            raise SystemExit(f"{d}: trials {sorted(s.index)} do not match baseline {sorted(b.index)}")
        v = ((s - b.reindex(s.index)) * 100).to_numpy()
        m, h, sh = v.mean(), ci95(v), ci95(s * 100)
        rows.append(dict(baseline=base, strategy=os.path.basename(d), amount_pct=pct, n_catalysts=n_cat, mean_pp=m,
                         ci95_lo=m - h, ci95_hi=m + h, k_up=int((v > 0).sum()), n=len(v),
                         paired_t_p=stats.ttest_1samp(v, 0).pvalue, baseline_share_pct=b.mean() * 100,
                         strategy_share_pct=s.mean() * 100, strategy_ci95_lo=s.mean() * 100 - sh,
                         strategy_ci95_hi=s.mean() * 100 + sh))
    return base_cmd, sorted(cat), pd.DataFrame(rows).sort_values("amount_pct", ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="directory holding the <base>_<pct>-<cmd> experiments")
    ap.add_argument("--baseline-root", required=True)
    ap.add_argument("--groups", nargs="+", action="append", required=True, help="baselines of one chart; repeatable")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--large", type=int, default=6)
    ap.add_argument("--last", type=float, default=100.0)
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--n", type=int, default=500, help="N_SMARTICLES, for the robot counts on the x axis")
    ap.add_argument("--force", action="store_true", help="recompute every share from the groups caches")
    ap.add_argument("--bare", action="store_true", help="also write curve-only copies (<name>_bare)")
    ap.add_argument("--size-ge", type=int, default=None, metavar="S",
                    help="plot the share of robots in groups of size >= S (own cache, _ge<S> file names)")
    ap.add_argument("--kinds", nargs="+", choices=["dose", "level"], default=["dose", "level"])
    ap.add_argument("--ylim", nargs=2, type=float, default=None, metavar=("LO", "HI"),
                    help="fixed y limits of the level chart, e.g. to match another set of baselines")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    if args.size_ge:
        shares = SizeShares(args.out_dir, args.size_ge, args.last, args.max_dist, args.force)
        size_txt, what, mtag = f"size {args.size_ge}+", "gain", f"_ge{args.size_ge}"
    else:
        shares = Shares(args.root, args.out_dir, args.large, args.last, args.max_dist, args.force)
        size_txt, what, mtag = f"size {args.large}+", "gain", ""
    Size_txt = size_txt[0].upper() + size_txt[1:]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 14, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})

    for bases in args.groups:
        curves = [curve(b, args.root, args.baseline_root, shares) for b in bases]
        table = pd.concat([c for _, _, c in curves], ignore_index=True)
        n_tr = int(table.n.iloc[0])
        cat = sorted({k for _, cmds, _ in curves for k in cmds})
        cat_txt = ", ".join(str(k) for k in cat)
        tag = "_".join(bases)

        kinds = {
            "dose": dict(y="mean_pp", lo="ci95_lo", hi="ci95_hi", zero=True,
                         ylabel=f"Δ share of robots in {size_txt} (pp)\nstrategy − baseline",
                         title=f"{Size_txt} {what} against fixed catalyst amount ({cat_txt}), last {args.last:g} s\n"
                               f"point = mean paired difference over {n_tr} trials, whisker = 95% CI; 0% = baseline"),
            "level": dict(y="strategy_share_pct", lo="strategy_ci95_lo", hi="strategy_ci95_hi", zero=False,
                          ylabel=f"share of robots in {size_txt} (%)",
                          title=f"{Size_txt} share against fixed catalyst amount ({cat_txt}), last {args.last:g} s\n"
                                f"point = mean over {n_tr} trials, whisker = 95% CI; 0% = baseline;\n"
                                "100% = all robots catalysts, one run shared by every curve"),
        }
        for kind, k in ((kd, kinds[kd]) for kd in args.kinds):
            for bare in ([False, True] if args.bare else [False]):
                fig, ax = plt.subplots(figsize=(9.0, 6.2), facecolor=SURFACE, layout="constrained")
                ax.set_facecolor(SURFACE)
                ax.grid(True, axis="y", color=GRID, lw=0.8)
                ax.set_axisbelow(True)
                for s in ("top", "right"):
                    ax.spines[s].set_visible(False)
                if k["zero"]:
                    ax.axhline(0, color=INK2, lw=1)
                labels = []
                for j, (base, col) in enumerate(zip(bases, PAIR_COLORS)):
                    base_cmd, _, c = curves[j]
                    # nudge sideways so whiskers at the same amount stay apart; points that are one shared value
                    # (0% of the Δ chart, 100% of the level chart) stay put
                    pinned = (c.amount_pct == 0) | ((c.amount_pct >= 100) & (not k["zero"]))
                    x = c.amount_pct.to_numpy() + np.where(pinned, 0, (j - (len(bases) - 1) / 2) * 0.8)
                    y = c[k["y"]]
                    ax.errorbar(x, y, yerr=[y - c[k["lo"]], c[k["hi"]] - y], color=col, lw=2, elinewidth=1.3,
                                capsize=4, marker="o", ms=8, mec=SURFACE, mew=1.5, zorder=3 + j,
                                label=f"{base}  (base cmd {base_cmd})")
                    if not bare:
                        # dose chart: label the right end (curves share 0%); level chart: label the left end
                        # (curves share 100%, and the baselines differ)
                        i = len(c) - 1 if k["zero"] else 0
                        labels.append([y.iloc[i], x[i], base])
                ax.set_xlim(-3 if k["zero"] or bare else -13, 105)
                ax.set_ylabel(k["ylabel"])
                if not k["zero"]:
                    ax.set_ylim(*args.ylim) if args.ylim else ax.set_ylim(bottom=0)
                # spread labels that would overlap: at least 4.5% of the y range apart, centred on their points
                gap = 0.045 * np.ptp(ax.get_ylim())
                labels.sort()
                ys = [lb[0] for lb in labels]
                for _ in range(50):
                    moved = False
                    for a in range(len(ys) - 1):
                        d = ys[a + 1] - ys[a]
                        if d < gap - 1e-9:
                            ys[a] -= (gap - d) / 2
                            ys[a + 1] += (gap - d) / 2
                            moved = True
                    if not moved:
                        break
                for (y0, x0, base), yl in zip(labels, ys):
                    if k["zero"]:
                        ax.text(x0 + 1.4, yl, base, ha="left", va="bottom", fontsize=13, color=INK)
                    else:
                        ax.text(x0 - 1.6, yl, base, ha="right", va="center", fontsize=13, color=INK)
                name = f"catalyst_fixed_{kind}{mtag}_{tag}_last{args.last:g}"
                if bare:
                    ax.set_xticks(X_TICKS, [""] * len(X_TICKS))
                    name += "_bare"
                else:
                    ax.set_xticks(X_TICKS, [f"{p}%\n({round(p * args.n / 100)})" for p in X_TICKS])
                    ax.tick_params(axis="x", length=4, labelsize=13)
                    ax.set_xlabel(f"catalysts: % of robots (number, of {args.n})")
                    ax.set_title(k["title"], loc="left", fontsize=12)
                    fig.legend(frameon=False, loc="outside lower center", ncol=len(bases), fontsize=13,
                               handlelength=1.6, columnspacing=1.5)
                out = os.path.join(args.out_dir, name)
                fig.savefig(out + ".png", dpi=150, facecolor=SURFACE)
                fig.savefig(out + ".svg", facecolor=SURFACE)
                plt.close(fig)
                print(f"wrote {out}.png / .svg", flush=True)

        out = os.path.join(args.out_dir, f"catalyst_fixed{mtag}_{tag}_last{args.last:g}.csv")
        table.to_csv(out, index=False)
        pd.set_option("display.width", 220)
        print(table.round(3).to_string(index=False))
        print(f"wrote {out}", flush=True)

    shares.table.to_csv(shares.path, index=False)


if __name__ == "__main__":
    main()

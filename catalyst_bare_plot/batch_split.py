"""
batch_split.py
--------------
For a run_analysis.py output made from merged batches (merge_batches.py),
split each pair's paired differences by batch, to check that the batches agree
before pooling them.

Reads <out_root>/<base>_vs_<strat>/hypotheses_*_<window>_trials.csv and the
strategy's _merge_map.csv (which batch, IC and gait-switch mode each trial
came from), and writes <out_root>/batch_split_<window>.csv: per pair, metric and
batch, the mean paired difference (strategy - baseline), its 95% t-interval,
k/n trials up, and the paired t p-value; plus a Welch test of batch 1 vs batch 2
differences.

Usage:
    python batch_split.py <out_root> --strategy-root <merged strategy root> [--window last100]
"""

import argparse
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

METRICS = ["share_ge6", "share_ge3", "large_given_grp", "mean_size", "large_form_per_min", "large_km_median_s"]


def ci(v):
    n = len(v)
    if n < 2:
        return np.nan, np.nan
    h = stats.t.ppf(0.975, n - 1) * v.std(ddof=1) / np.sqrt(n)
    return v.mean() - h, v.mean() + h


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_root")
    ap.add_argument("--strategy-root", required=True)
    ap.add_argument("--window", default="last100")
    args = ap.parse_args()

    rows = []
    for csv in sorted(glob.glob(os.path.join(args.out_root, "*_vs_*", f"hypotheses_*_{args.window}_trials.csv"))):
        t = pd.read_csv(csv)
        base = os.path.basename(os.path.dirname(csv)).split("_vs_")[0]
        strat = [e for e in t.exp.unique() if e != base][0]
        mp = pd.read_csv(os.path.join(args.strategy_root, strat, "_merge_map.csv")).set_index("trial")
        w = t.pivot(index="trial", columns="exp")
        label = {b: f"{b} (IC {mp[mp.batch == b].init_index.min()}-{mp[mp.batch == b].init_index.max()}, "
                    f"{mp[mp.batch == b].gait_switch_mode.iloc[0]})" for b in mp.batch.unique()}
        for m in METRICS:
            if m not in w:
                continue
            d = (w[m][strat] - w[m][base]).dropna()
            groups = {b: d[d.index.map(mp.batch) == b].to_numpy() for b in mp.batch.unique()}
            for b, v in list(groups.items()) + [("all", d.to_numpy())]:
                lo, hi = ci(v)
                rows.append(dict(baseline=base, strategy=strat, metric=m, batch=label.get(b, b), n=len(v),
                                 diff=v.mean(), ci95_lo=lo, ci95_hi=hi, n_up=int((v > 0).sum()),
                                 paired_t_p=stats.ttest_1samp(v, 0).pvalue if len(v) > 1 else np.nan))
            if len(groups) == 2:
                a, c = groups.values()
                rows[-1]["batch_welch_p"] = stats.ttest_ind(a, c, equal_var=False).pvalue
    out = pd.DataFrame(rows)
    path = os.path.join(args.out_root, f"batch_split_{args.window}.csv")
    out.to_csv(path, index=False)
    pd.set_option("display.width", 250)
    print(out.round(4).to_string(index=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

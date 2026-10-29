"""
scaling_stats_update.py  -  recompute selected conditions and fold them back
into an existing scaling_* result set.

Same window, grouping and error definitions as scaling_stats2.py; see
GROUP_STATS_METHOD.md. The point of this one is incremental: when a couple of
conditions are re-run or added, the other dozen do not need a rerun, and the
rows that are not touched stay exactly as they were.

_scaling_stats_raw.csv is the source of truth. The two presentation CSVs are
regenerated from it, so the derived columns (ci95, drift_sigma, status) are
always computed the same way for every row rather than some rows carrying an
older convention.

    python scaling_stats_update.py <datafile dir> a862c_500 a862c_1000
"""

import glob
import os
import sys

import numpy as np
import pandas as pd

# The simulation modules (config, pos_all_grouping, strategy, gait, ...)
# live one level up. Running this as `python analysis/<name>.py` puts only
# analysis/ on the path, so the project root has to be added explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pos_all_grouping import compute_frame_groups

T_MIN, FPS, MAXD = 400.0, 60.0, 65.0
T95 = 2.262        # t, df = 9


def per_trial(path):
    df = pd.read_csv(path, usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    df = df[df["Step"] >= int(T_MIN * FPS)]
    if df.empty:
        return None
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    step = df["Step"].to_numpy()
    xy = df[["X", "Y"]].to_numpy(float)
    th = df["Theta"].to_numpy(float)
    aid = df["Agent_ID"].to_numpy()
    steps, starts = np.unique(step, return_index=True)
    b = np.append(starts, len(step))
    nf = len(steps)
    largest = np.empty(nf); sw_a = np.empty(nf); sw_m = np.empty(nf)
    for k in range(nf):
        sl = slice(b[k], b[k + 1])
        s = np.array([g["size"] for g in
                      compute_frame_groups(xy[sl], th[sl], aid[sl], MAXD, True)],
                     dtype=np.int64)
        largest[k] = s.max()
        sw_a[k] = (s * s).sum() / s.sum()
        m = s[s >= 2]
        sw_m[k] = (m * m).sum() / m.sum() if m.size else np.nan
    t = steps / FPS
    drift = 100.0 * np.polyfit(t, largest, 1)[0] * 100.0 / largest.mean()
    return largest.mean(), sw_a.mean(), np.nanmean(sw_m), drift, nf


def condition_row(cond_dir):
    N = int(os.path.basename(cond_dir).rsplit("_", 1)[1])
    vals = []
    for tr in sorted(glob.glob(os.path.join(cond_dir, "trial_*"))):
        pa = glob.glob(os.path.join(tr, "*_POS_ALL.csv"))
        if not pa:
            continue
        r = per_trial(pa[0])
        if r:
            vals.append(r)
            print(f"    {os.path.basename(tr)}: largest={r[0]:6.2f} "
                  f"sw={r[1]:5.2f} drift={r[3]:+6.2f}", flush=True)
    if not vals:
        return None
    a = np.array(vals)

    def ms(c):
        v = a[:, c]
        return v.mean(), v.std(ddof=1), v.std(ddof=1) / np.sqrt(len(v))

    lm, lsd, lse = ms(0); am, asd, ase = ms(1); mm, msd, mse = ms(2)
    return dict(N=N, n_trials=len(a), n_frames=int(a[:, 4].mean()),
                largest_mean=lm, largest_sd=lsd, largest_sem=lse,
                largest_over_N=lm / N,
                mean_size_w_mean=am, mean_size_w_sd=asd, mean_size_w_sem=ase,
                mean_size_w_nosing_mean=mm, mean_size_w_nosing_sd=msd,
                mean_size_w_nosing_sem=mse,
                drift_pct_per_100s=a[:, 3].mean(),
                drift_sem=a[:, 3].std(ddof=1) / np.sqrt(len(a)))


def write_outputs(d, root):
    t = (d.drift_pct_per_100s / d.drift_sem).abs()
    status = np.where(t >= 2.0, "lower_bound",
                      np.where(t >= 1.0, "marginal", "steady"))
    pd.DataFrame({
        "N": d.N, "n_trials": d.n_trials, "n_frames_per_trial": d.n_frames,
        "largest_mean": d.largest_mean.round(4),
        "largest_sd": d.largest_sd.round(4),
        "largest_sem": d.largest_sem.round(4),
        "largest_ci95_halfwidth": (d.largest_sem * T95).round(4),
        "largest_over_N": d.largest_over_N.round(5),
        "drift_pct_per_100s": d.drift_pct_per_100s.round(3),
        "drift_sem": d.drift_sem.round(3),
        "drift_sigma": t.round(2), "status": status,
    }).to_csv(os.path.join(root, "scaling_largest_groupsize.csv"), index=False)
    pd.DataFrame({
        "N": d.N, "n_trials": d.n_trials, "n_frames_per_trial": d.n_frames,
        "mean_size_w_mean": d.mean_size_w_mean.round(4),
        "mean_size_w_sd": d.mean_size_w_sd.round(4),
        "mean_size_w_sem": d.mean_size_w_sem.round(4),
        "mean_size_w_ci95_halfwidth": (d.mean_size_w_sem * T95).round(4),
        "nosing_mean": d.mean_size_w_nosing_mean.round(4),
        "nosing_sd": d.mean_size_w_nosing_sd.round(4),
        "nosing_sem": d.mean_size_w_nosing_sem.round(4),
        "nosing_ci95_halfwidth": (d.mean_size_w_nosing_sem * T95).round(4),
    }).to_csv(os.path.join(root, "scaling_mean_groupsize.csv"), index=False)


def main():
    root = os.path.abspath(sys.argv[1])
    conds = sys.argv[2:]
    if not conds:
        print("Give the condition folders to recompute, e.g. a862c_500 a862c_1000")
        return 1

    raw_path = os.path.join(root, "_scaling_stats_raw.csv")
    raw = (pd.read_csv(raw_path) if os.path.isfile(raw_path)
           else pd.DataFrame(columns=["N"]))
    before = set(raw.N.tolist())

    rows = []
    for c in conds:
        d = os.path.join(root, c)
        if not os.path.isdir(d):
            print(f"  [skip] {c}: no such folder")
            continue
        print(f"  {c}:", flush=True)
        r = condition_row(d)
        if r:
            rows.append(r)
            print(f"  -> N={r['N']}: largest {r['largest_mean']:.2f}"
                  f"+-{r['largest_sem']:.2f}  sw {r['mean_size_w_mean']:.2f}"
                  f"+-{r['mean_size_w_sem']:.2f}", flush=True)
    if not rows:
        print("Nothing computed.")
        return 1

    new = pd.DataFrame(rows)
    # Drop the old rows for these N, then append and re-sort: a re-run replaces
    # its row rather than appearing twice, and a new N slots into place.
    keep = raw[~raw.N.isin(new.N)] if len(raw) else raw
    d = pd.concat([keep, new], ignore_index=True).sort_values("N")
    d.to_csv(raw_path, index=False)
    write_outputs(d, root)

    for n in sorted(new.N):
        print(f"  {'updated' if n in before else 'added  '} N={n}")
    print(f"\n-> {raw_path}")
    print(f"-> scaling_largest_groupsize.csv, scaling_mean_groupsize.csv "
          f"({len(d)} rows: {d.N.tolist()})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

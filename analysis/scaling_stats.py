"""
scaling_stats.py  -  largest and mean group size per population, by the
steady-state method.

The scaling counterpart of sweep_stats.py: same definitions, same error
conventions (see GROUP_STATS_METHOD.md), but condition folders are named
<prefix>_<N> and the independent variable is N rather than the gait.

Reads the cached group tables (*_groups_d<maxdist>.csv) when they exist. They
already cover the whole trial, so the steady-state scan gets its full-length
series without re-running the grouping -- which is the expensive part and would
otherwise be repeated for 15 GB of positions. The caches are read in chunks,
two columns at a time and only the sampled steps kept, so a 1000-robot trial
(~0.7 GB of cache) never sits in memory whole; trials run one after another.

Steady-state settings (window, drift, band, merge step, sampling) are imported
from sweep_stats so both experiments are judged the same way. Every steady
segment is kept; a trial's value is the duration-weighted average over them,
and the segments are written to _scaling_steady_segments.csv.

    python analysis/scaling_stats.py <datafile dir>
    python analysis/scaling_stats.py <dir> --fixed      # old t>=400s method
"""

import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

# The simulation modules (pos_all_grouping) live one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from steady_state import steady_states
from sweep_stats import WINDOW_S, MAX_DRIFT, MAX_DEV, SMOOTH_S, MAX_STEP, STRIDE

FPS, MAXD = 60.0, 65.0
T_MIN = 400.0            # --fixed: last 100 s of a 500 s trial
T95 = 2.262              # t, df = 9
CHUNK = 2_000_000        # cache rows per read


def cache_path(trial_dir, max_dist):
    tag = f"{max_dist:g}".replace(".", "p")
    hits = glob.glob(os.path.join(trial_dir, f"*_groups_d{tag}.csv"))
    return hits[0] if hits else None


def series_from_cache(path, stride):
    """-> (t seconds, largest per frame, size-weighted mean, mean excl. singletons)."""
    parts = []
    for c in pd.read_csv(path, usecols=["Step", "size"], chunksize=CHUNK,
                         dtype={"Step": np.int32, "size": np.int32}):
        parts.append(c[c["Step"] % stride == 0] if stride > 1 else c)
    df = pd.concat(parts, ignore_index=True)
    del parts
    if df.empty:
        return None
    df = df.sort_values("Step", kind="mergesort")
    step = df["Step"].to_numpy()
    s = df["size"].to_numpy().astype(np.int64)
    steps, starts = np.unique(step, return_index=True)
    b = np.append(starts, len(step))
    n = len(steps)
    largest = np.empty(n); sw_a = np.empty(n); sw_m = np.empty(n)
    for k in range(n):
        c = s[b[k]:b[k + 1]]
        largest[k] = c.max()
        sw_a[k] = (c * c).sum() / c.sum()
        m = c[c >= 2]
        sw_m[k] = (m * m).sum() / m.sum() if m.size else np.nan
    return steps / FPS, largest, sw_a, sw_m


def series_from_pos_all(trial_dir, max_dist, stride):
    """Fallback when no group cache exists: group the frames here."""
    from pos_all_grouping import compute_frame_groups
    hits = glob.glob(os.path.join(trial_dir, "*_POS_ALL.csv"))
    if not hits:
        return None
    df = pd.read_csv(hits[0], usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    if stride > 1:
        df = df[df["Step"] % stride == 0]
    if df.empty:
        return None
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    step = df["Step"].to_numpy(); xy = df[["X", "Y"]].to_numpy(float)
    th = df["Theta"].to_numpy(float); aid = df["Agent_ID"].to_numpy()
    steps, starts = np.unique(step, return_index=True)
    b = np.append(starts, len(step))
    n = len(steps)
    largest = np.empty(n); sw_a = np.empty(n); sw_m = np.empty(n)
    for k in range(n):
        sl = slice(b[k], b[k + 1])
        s = np.array([g["size"] for g in
                      compute_frame_groups(xy[sl], th[sl], aid[sl], max_dist, True)],
                     dtype=np.int64)
        largest[k] = s.max()
        sw_a[k] = (s * s).sum() / s.sum()
        m = s[s >= 2]
        sw_m[k] = (m * m).sum() / m.sum() if m.size else np.nan
    return steps / FPS, largest, sw_a, sw_m


def per_trial(trial_dir, method, max_dist, stride):
    """
    -> ((largest, mean, mean_nosing, coverage, n_samples), segments) or None.
    Steady: values are duration-weighted over all steady segments, coverage is
    the steady fraction; segments is empty for --fixed.
    """
    cp = cache_path(trial_dir, max_dist)
    r = (series_from_cache(cp, stride) if cp
         else series_from_pos_all(trial_dir, max_dist, stride))
    if r is None:
        return None
    t, largest, sw_a, sw_m = r

    if method == "fixed":
        sel = t >= T_MIN
        if not sel.any():
            return None
        drift = (100.0 * np.polyfit(t[sel], largest[sel], 1)[0] * 100.0
                 / largest[sel].mean())
        return (largest[sel].mean(), sw_a[sel].mean(),
                float(np.nanmean(sw_m[sel])), drift, int(sel.sum())), []

    segs = steady_states(t, largest, sw_a, window_s=WINDOW_S,
                         max_drift=MAX_DRIFT, max_dev=MAX_DEV,
                         cv_smooth_s=SMOOTH_S, max_step=MAX_STEP,
                         extra={"nosing": sw_m})
    if not segs:
        return None
    w = np.array([g["dur"] for g in segs])
    avg = lambda k: float(np.average([g[k] for g in segs], weights=w))
    frac = float(w.sum() / (t[-1] - t[0] + np.median(np.diff(t))))
    return (avg("largest"), avg("mean"), avg("nosing"), frac, len(t)), segs


def main():
    ap = argparse.ArgumentParser(
        description="Largest and mean group size per population N.")
    ap.add_argument("root", help="datafile dir holding the <prefix>_<N> folders")
    ap.add_argument("--fixed", action="store_true",
                    help="use the old fixed t>=400s window instead of steady segments")
    ap.add_argument("--max-dist", type=float, default=MAXD)
    ap.add_argument("--stride", type=int, default=STRIDE)
    ap.add_argument("--prefix", default=None,
                    help="only conditions starting with this (default: all that end in _<number>)")
    args = ap.parse_args()

    method = "fixed" if args.fixed else "steady"
    root = os.path.abspath(args.root)

    conds = []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        if not os.path.isdir(d):
            continue
        base = os.path.basename(d)
        tail = base.rsplit("_", 1)[-1]
        if not tail.isdigit():
            continue
        if args.prefix and not base.startswith(args.prefix):
            continue
        conds.append((int(tail), d))
    conds.sort()
    if not conds:
        print(f"No <prefix>_<N> condition folders under {root}")
        return 1

    print(f"{len(conds)} conditions, method={method}"
          + (f" (all steady segments; window {WINDOW_S:g}s, "
             f"drift <= {MAX_DRIFT:.0%}, {SMOOTH_S:g}s mean within "
             f"+-{MAX_DEV:.0%}, merge step <= {MAX_STEP:.0%}, "
             f"{FPS/args.stride:g} Hz)" if method == "steady"
             else f" (t >= {T_MIN:g}s)"))

    per, rows, seg_rows = [], [], []
    for N, d in conds:
        vals, nseg = [], []
        for tr in sorted(glob.glob(os.path.join(d, "trial_*"))):
            if not os.path.isdir(tr):
                continue
            res = per_trial(tr, method, args.max_dist, args.stride)
            if res is None:
                print(f"    {os.path.basename(tr)}: no usable window, skipped",
                      flush=True)
                continue
            r, segs = res
            nseg.append(len(segs))
            for k, g in enumerate(segs):
                seg_rows.append(dict(N=N, trial=os.path.basename(tr), seg=k,
                                     t0=g["t0"], t1=g["t1"], dur=g["dur"],
                                     largest=g["largest"], mean=g["mean"],
                                     nosing=g["nosing"],
                                     largest_cv=g["largest_cv"]))
            vals.append(r)
            per.append(dict(N=N, trial=os.path.basename(tr), largest=r[0],
                            mean_size_w=r[1], mean_size_w_nosing=r[2],
                            **({"steady_frac": r[3]} if method == "steady"
                               else {"drift_pct_per_100s": r[3]})))
        if not vals:
            continue
        a = np.array(vals)

        def ms(c):
            v = a[:, c]
            return v.mean(), v.std(ddof=1), v.std(ddof=1) / np.sqrt(len(v))

        lm, lsd, lse = ms(0); am, asd, ase = ms(1); mm, msd, mse = ms(2)
        cvv, cvs = a[:, 3].mean(), a[:, 3].std(ddof=1) / np.sqrt(len(a))
        rows.append(dict(N=N, n_trials=len(a), n_samples=int(a[:, 4].mean()),
                         largest_mean=lm, largest_sd=lsd, largest_sem=lse,
                         largest_over_N=lm / N,
                         mean_mean=am, mean_sd=asd, mean_sem=ase,
                         nosing_mean=mm, nosing_sd=msd, nosing_sem=mse,
                         wincov=cvv, wincov_sem=cvs,
                         segs_per_trial=float(np.mean(nseg))))
        print(f"  N={N:<5d} largest {lm:6.2f}+-{lse:.2f}   "
              f"mean {am:5.2f}+-{ase:.2f}   "
              + (f"steady {cvv:.0%}" if method == "steady"
                 else f"drift {cvv:+.1f}%/100s"), flush=True)

    d = pd.DataFrame(rows).sort_values("N")
    if method == "steady":
        status = np.where(d["wincov"] < 0.3, "sparse_steady",
                          np.where(d["wincov"] < 0.5, "marginal", "steady"))
        cov_cols = {"steady_frac": d["wincov"].round(3),
                    "steady_frac_sem": d["wincov_sem"].round(3),
                    "segs_per_trial": d["segs_per_trial"].round(2)}
    else:
        sig = (d["wincov"] / d["wincov_sem"]).abs()
        status = np.where(sig >= 2.0, "lower_bound",
                          np.where(sig >= 1.0, "marginal", "steady"))
        cov_cols = {"drift_pct_per_100s": d["wincov"].round(3),
                    "drift_sem": d["wincov_sem"].round(3),
                    "drift_sigma": sig.round(2)}

    pd.DataFrame({
        "N": d.N, "n_trials": d.n_trials, "n_samples_per_trial": d.n_samples,
        "largest_mean": d.largest_mean.round(4),
        "largest_sd": d.largest_sd.round(4),
        "largest_sem": d.largest_sem.round(4),
        "largest_ci95_halfwidth": (d.largest_sem * T95).round(4),
        "largest_over_N": d.largest_over_N.round(5),
        "method": method, **cov_cols, "status": status,
    }).to_csv(os.path.join(root, "scaling_largest_groupsize.csv"), index=False)

    pd.DataFrame({
        "N": d.N, "n_trials": d.n_trials, "n_samples_per_trial": d.n_samples,
        "mean_size_w_mean": d.mean_mean.round(4),
        "mean_size_w_sd": d.mean_sd.round(4),
        "mean_size_w_sem": d.mean_sem.round(4),
        "mean_size_w_ci95_halfwidth": (d.mean_sem * T95).round(4),
        "nosing_mean": d.nosing_mean.round(4),
        "nosing_sd": d.nosing_sd.round(4),
        "nosing_sem": d.nosing_sem.round(4),
        "nosing_ci95_halfwidth": (d.nosing_sem * T95).round(4),
        "method": method,
    }).to_csv(os.path.join(root, "scaling_mean_groupsize.csv"), index=False)

    pd.DataFrame(per).to_csv(
        os.path.join(root, "_scaling_per_trial.csv"), index=False)
    outs = ("scaling_largest_groupsize.csv, scaling_mean_groupsize.csv, "
            "_scaling_per_trial.csv")
    if seg_rows:
        sg = pd.DataFrame(seg_rows)
        sg.round(4).to_csv(os.path.join(root, "_scaling_steady_segments.csv"),
                           index=False)
        # Per N, each trial first reduced to its own mean segment length, as in
        # plot_sweep_heatmaps.duration_table.
        pt = (sg.groupby(["N", "trial"])
                .agg(seg_dur=("dur", "mean"), n_segments=("dur", "size"),
                     steady_s=("dur", "sum")).reset_index())
        g = pt.groupby("N")
        du = pd.DataFrame({"n_trials": g.size(),
                           "seg_dur_mean": g.seg_dur.mean(),
                           "seg_dur_sd": g.seg_dur.std(),
                           "n_segments_mean": g.n_segments.mean(),
                           "steady_s_mean": g.steady_s.mean()})
        du["seg_dur_sem"] = du.seg_dur_sd / np.sqrt(du.n_trials)
        du["seg_dur_ci95_halfwidth"] = du.seg_dur_sem * T95
        du.reset_index().round(3).to_csv(
            os.path.join(root, "scaling_steady_duration.csv"), index=False)
        outs += ", _scaling_steady_segments.csv, scaling_steady_duration.csv"
    print(f"\n-> {outs}  in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

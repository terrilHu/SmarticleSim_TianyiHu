"""
sweep_stats.py  -  per-gait group-size statistics for the 1014 sweep.

Same definitions and window as scaling_stats*.py; see GROUP_STATS_METHOD.md.
What differs here is the design: every condition uses the same ten initial
conditions, so this also writes the per-(pattern, ic) values, which is what any
paired comparison between two gaits needs.

Groups only the t>=400s window straight from POS_ALL. The full group table
would cost five times as much and write tens of GB that nothing else reads.

    python sweep_stats.py datafile/1014_sweep
"""

import glob
import math
import os
import sys

import numpy as np
import pandas as pd

# The simulation modules (config, pos_all_grouping, strategy, gait, ...)
# live one level up. Running this as `python analysis/<name>.py` puts only
# analysis/ on the path, so the project root has to be added explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pos_all_grouping import compute_frame_groups
from steady_state import steady_states

FPS, MAXD = 60.0, 65.0

# --- fixed-window method (the original) -------------------------------------
T_MIN = 400.0            # last 100s of a 500s trial

# --- steady-state method (the default) --------------------------------------
# Slide a window over the WHOLE trial and keep every stretch where grouping has
# settled. The fixed window assumed the last 100s was steady; measured, 13 of
# 25 conditions still had a residual trend inside it, and a trial that settles
# early and is later disturbed gets scored on the disturbance.
#
# A trial can settle at more than one level (plateau, regroup, new plateau).
# All of those steady states count: every segment is written out, and a
# trial's value is the time average over ALL its steady time -- not the
# highest segment, which favoured short transient bumps once segments got
# shorter.
WINDOW_S = 60.0          # short enough to resolve plateaus of about a minute
# Trend change across a window. The largest group is an integer, and at sizes
# 7-15 one robot joining or leaving is 7-14%; with the frame-to-frame jitter the
# fitted slope alone is uncertain by ~4% over 60s. A tighter cap (3% was
# tried) rejects flat windows at random and any window holding a single
# one-robot step.
MAX_DRIFT = 0.08
# Band: the SMOOTH_S moving average may not stray more than MAX_DEV of the
# window mean anywhere in the window. The drift test only sees the slope, so a
# window that rises and falls back passes it; a cap on the worst point also
# stops a step in the last few seconds of a window, which an average spread
# (cv) barely registers.
MAX_DEV = 0.15
SMOOTH_S = 10.0
# Steady windows are merged into one segment only while their means stay within
# MAX_STEP of the segment's; otherwise two plateaus whose windows touch (10 then
# 15) come back as one segment averaging a level that never occurred.
MAX_STEP = 0.08
STRIDE = 30              # sample at 60/30 = 2 Hz; ample for a 100s window, and
                         # ~6x cheaper than grouping every frame


def frame_series(path, stride=1, t_min=None):
    """-> (t seconds, largest per frame, size-weighted mean per frame)."""
    df = pd.read_csv(path, usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    if t_min is not None:
        df = df[df["Step"] >= int(t_min * FPS)]
    if stride > 1:
        df = df[df["Step"] % stride == 0]
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
    return steps / FPS, largest, sw_a, sw_m


def per_trial_steady(path):
    """
    -> ((largest, mean, nosing, steady_frac, n_samples), segments)

    The three values are time averages over all of the trial's steady time,
    i.e. its segments weighted by duration; `segments` is every steady segment
    (steady_state.steady_states), for the per-segment table. All three
    quantities are averaged over the same spans.
    """
    r = frame_series(path, stride=STRIDE)
    if r is None:
        return None
    t, largest, sw_a, sw_m = r
    segs = steady_states(t, largest, sw_a, window_s=WINDOW_S,
                         max_drift=MAX_DRIFT, max_dev=MAX_DEV,
                         cv_smooth_s=SMOOTH_S, max_step=MAX_STEP,
                         extra={"nosing": sw_m})
    if not segs:
        return (np.nan, np.nan, np.nan, 0.0, len(t)), []
    w = np.array([g["dur"] for g in segs])
    avg = lambda k: float(np.average([g[k] for g in segs], weights=w))
    frac = float(w.sum() / (t[-1] - t[0] + np.median(np.diff(t))))
    return (avg("largest"), avg("mean"), avg("nosing"), frac, len(t)), segs


def per_trial(path):
    """-> (largest, sw_all, sw_multi, drift_pct_per_100s, n_frames)."""
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


def decode(pattern):
    """aXYZ -> (Y, Z, amplitude deg, frequency Hz). X is the phase slot and
    plays no part in either quantity."""
    from gait import AMPLI_TABLE, FREQ_TABLE
    y, z = int(pattern[-2]), int(pattern[-1])
    return y, z, round(math.degrees(AMPLI_TABLE[y - 1])), FREQ_TABLE[z - 1]


def ic_map(root, pattern):
    """trial_id -> ic_idx, from the condition's own summary csv."""
    f = os.path.join(root, f"{pattern}_summary.csv")
    if not os.path.isfile(f):
        return {}
    d = pd.read_csv(f)
    if not {"trial_id", "ic_idx"} <= set(d.columns):
        return {}
    return dict(zip(d["trial_id"], d["ic_idx"]))


def variance_split(per_trial_df, col):
    """
    Share of the variance in `col` explained by the initial condition, from the
    pattern x ic layout. High share means the pairing is doing real work: that
    much of the trial-to-trial scatter is common to every condition and cancels
    in a paired comparison, while an unpaired test would carry it twice.
    """
    p = per_trial_df.pivot_table(index="ic_idx", columns="pattern", values=col)
    p = p.dropna()
    if p.shape[0] < 2 or p.shape[1] < 2:
        return float("nan")
    g = p.to_numpy()
    grand = g.mean()
    ss_tot = ((g - grand) ** 2).sum()
    ss_ic = g.shape[1] * ((g.mean(axis=1) - grand) ** 2).sum()
    return float(ss_ic / ss_tot) if ss_tot > 0 else float("nan")


def occurrence(sg, pt):
    """
    How often, how long and at what level each condition sat in a steady
    state, from the per-segment table. Segment levels are weighted by duration
    so a one-minute plateau does not count as much as a five-minute one.
    """
    out = []
    for pat, g in sg.groupby("pattern", sort=False):
        w = g.dur.to_numpy()
        n_tr = int((pt.pattern == pat).sum())
        row = dict(pattern=pat, Y=g.Y.iloc[0], Z=g.Z.iloc[0],
                   cmd_ampli_deg=g.cmd_ampli_deg.iloc[0],
                   cmd_freq_hz=g.cmd_freq_hz.iloc[0],
                   n_trials=n_tr, n_segments=len(g),
                   segs_per_trial=len(g) / max(1, n_tr),
                   seg_dur_median_s=float(np.median(w)),
                   seg_dur_min_s=float(w.min()),
                   steady_frac=float(pt.loc[pt.pattern == pat,
                                            "steady_frac"].mean()))
        for k, lab in (("largest", "largest"), ("mean", "mean_size_w")):
            v = g[k].to_numpy()
            m = np.average(v, weights=w)
            row[f"{lab}_seg_wmean"] = m
            row[f"{lab}_seg_wsd"] = float(np.sqrt(np.average((v - m) ** 2,
                                                            weights=w)))
            row[f"{lab}_seg_min"] = float(v.min())
            row[f"{lab}_seg_max"] = float(v.max())
        row["largest_cv_in_seg_median"] = float(g.largest_cv.median())
        out.append(row)
    return pd.DataFrame(out).sort_values(["Y", "Z"]).round(4)


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    method = "fixed" if "--fixed" in sys.argv else "steady"
    root = os.path.abspath(args[0] if args
                           else os.path.join("datafile", "1014_sweep"))
    print(f"method: {method}"
          + (f" (all steady segments; window {WINDOW_S:g}s, "
             f"drift <= {MAX_DRIFT:.0%}, "
             f"{SMOOTH_S:g}s mean within +-{MAX_DEV:.0%}, "
             f"merge step <= {MAX_STEP:.0%}, "
             f"sampled at {FPS/STRIDE:g} Hz)" if method == "steady"
             else f" (t >= {T_MIN:g}s, every frame)"))
    # Any aXYZ folder, not just a8YZ: the same sweep is run at other
    # initial-phase slots (a0YZ is the desynchronised version), and the X digit
    # does not change how any of this is computed.
    conds = sorted(d for d in glob.glob(os.path.join(root, "a[0-9][0-9][0-9]"))
                   if os.path.isdir(d))
    if not conds:
        print(f"No condition folders under {root}")
        return 1
    print(f"{len(conds)} conditions under {root}")

    per, rows, seg_rows = [], [], []
    for ci, d in enumerate(conds, 1):
        pat = os.path.basename(d)
        y, z, adeg, fhz = decode(pat)
        ics = ic_map(root, pat)
        vals = []
        for tr in sorted(glob.glob(os.path.join(d, "trial_*"))):
            pa = glob.glob(os.path.join(tr, "*_POS_ALL.csv"))
            if not pa:
                continue
            tid = int(os.path.basename(tr).split("_")[1])
            if method == "steady":
                r, segs = per_trial_steady(pa[0]) or (None, [])
                for j, g in enumerate(segs):
                    seg_rows.append(dict(pattern=pat, Y=y, Z=z,
                                         cmd_ampli_deg=adeg, cmd_freq_hz=fhz,
                                         trial_id=tid, ic_idx=ics.get(tid, -1),
                                         seg=j, **g))
            else:
                r = per_trial(pa[0])
            if not r or not np.isfinite(r[0]):
                if r:
                    print(f"  [{ci}/{len(conds)}] {pat} trial {tid}: "
                          f"no steady segment, skipped", flush=True)
                continue
            vals.append(r)
            per.append(dict(pattern=pat, Y=y, Z=z, cmd_ampli_deg=adeg,
                            cmd_freq_hz=fhz, trial_id=tid,
                            ic_idx=ics.get(tid, -1),
                            largest=r[0], mean_size_w=r[1],
                            mean_size_w_nosing=r[2],
                            **({"steady_frac": r[3]} if method == "steady"
                               else {"drift_pct_per_100s": r[3]})))
            print(f"  [{ci}/{len(conds)}] {pat} trial {tid}: "
                  f"largest={r[0]:6.2f} sw={r[1]:5.2f}", flush=True)
        if not vals:
            continue
        a = np.array(vals)

        def ms(c):
            v = a[:, c]
            return v.mean(), v.std(ddof=1), v.std(ddof=1) / np.sqrt(len(v))

        lm, lsd, lse = ms(0); am, asd, ase = ms(1); mm, msd, mse = ms(2)
        dm = a[:, 3].mean(); dse = a[:, 3].std(ddof=1) / np.sqrt(len(a))
        rows.append(dict(pattern=pat, Y=y, Z=z, cmd_ampli_deg=adeg,
                         cmd_freq_hz=fhz, n_trials=len(a),
                         n_frames=int(a[:, 4].mean()),
                         largest_mean=lm, largest_sd=lsd, largest_sem=lse,
                         mean_size_w_mean=am, mean_size_w_sd=asd,
                         mean_size_w_sem=ase,
                         nosing_mean=mm, nosing_sd=msd, nosing_sem=mse,
                         drift_pct_per_100s=dm, drift_sem=dse))
        print(f"  {pat}: largest {lm:.2f}+-{lse:.2f}  sw {am:.2f}+-{ase:.2f}",
              flush=True)

    d = pd.DataFrame(rows).sort_values(["Y", "Z"])
    pt = pd.DataFrame(per)
    T95 = 2.262  # t, df = 9
    if method == "steady":
        # How much of the trial was steady at all. A cell resting on very
        # little steady time is flagged: its peak is then one short stretch
        # rather than a settled state the system held.
        t = d.drift_pct_per_100s          # reused column: carries steady_frac
        status = np.where(t < 0.3, "sparse_steady",
                          np.where(t < 0.5, "marginal", "steady"))
    else:
        t = (d.drift_pct_per_100s / d.drift_sem).abs()
        status = np.where(t >= 2.0, "lower_bound",
                          np.where(t >= 1.0, "marginal", "steady"))

    pd.DataFrame({
        "pattern": d.pattern, "Y": d.Y, "Z": d.Z,
        "cmd_ampli_deg": d.cmd_ampli_deg, "cmd_freq_hz": d.cmd_freq_hz,
        "n_trials": d.n_trials, "n_frames_per_trial": d.n_frames,
        "largest_mean": d.largest_mean.round(4),
        "largest_sd": d.largest_sd.round(4),
        "largest_sem": d.largest_sem.round(4),
        "largest_ci95_halfwidth": (d.largest_sem * T95).round(4),
        "method": method,
        ("steady_frac" if method == "steady" else "drift_pct_per_100s"):
            d.drift_pct_per_100s.round(3),
        ("steady_frac_sem" if method == "steady" else "drift_sem"):
            d.drift_sem.round(3),
        "status": status,
    }).to_csv(os.path.join(root, "sweep_largest_groupsize.csv"), index=False)

    pd.DataFrame({
        "pattern": d.pattern, "Y": d.Y, "Z": d.Z,
        "cmd_ampli_deg": d.cmd_ampli_deg, "cmd_freq_hz": d.cmd_freq_hz,
        "n_trials": d.n_trials, "n_frames_per_trial": d.n_frames,
        "mean_size_w_mean": d.mean_size_w_mean.round(4),
        "mean_size_w_sd": d.mean_size_w_sd.round(4),
        "mean_size_w_sem": d.mean_size_w_sem.round(4),
        "mean_size_w_ci95_halfwidth": (d.mean_size_w_sem * T95).round(4),
        "nosing_mean": d.nosing_mean.round(4),
        "nosing_sd": d.nosing_sd.round(4),
        "nosing_sem": d.nosing_sem.round(4),
        "nosing_ci95_halfwidth": (d.nosing_sem * T95).round(4),
        "method": method,
    }).to_csv(os.path.join(root, "sweep_mean_groupsize.csv"), index=False)

    pt.to_csv(os.path.join(root, "_sweep_per_trial.csv"), index=False)
    if method == "steady" and seg_rows:
        sg = pd.DataFrame(seg_rows)
        sg.round(4).to_csv(os.path.join(root, "_sweep_steady_segments.csv"),
                           index=False)
        occurrence(sg, pt).to_csv(
            os.path.join(root, "sweep_steady_occurrence.csv"), index=False)

    print("\nvariance explained by the initial condition "
          "(how much the pairing buys):")
    for c, lab in (("largest", "largest group"),
                   ("mean_size_w", "mean group size")):
        print(f"  {lab:18s} {variance_split(pt, c):.1%}")
    print(f"\n-> sweep_largest_groupsize.csv, sweep_mean_groupsize.csv, "
          f"_sweep_per_trial.csv"
          + (", _sweep_steady_segments.csv, sweep_steady_occurrence.csv"
             if method == "steady" else "") + f"  in {root}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

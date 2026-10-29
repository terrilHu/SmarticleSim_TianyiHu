"""
plot_largest_steady.py  -  largest group size over time, with the steady
segments that sweep_stats counts shaded in.

The shading is not re-derived with its own settings: it uses sweep_stats'
WINDOW_S / MAX_DRIFT / MAX_DEV / SMOOTH_S / MAX_STEP on the same 2 Hz samples
(Step % STRIDE == 0) that sweep_stats judges, so what is shaded is exactly
what enters _sweep_steady_segments.csv. The plotted line is per frame.

    python analysis/plot_largest_steady.py ".../synchronized/a822" ".../a862"
    python analysis/plot_largest_steady.py ".../a822/trial_0005" --formats png
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pos_all_grouping import compute_frame_groups
from steady_state import steady_states
from sweep_stats import (FPS, MAXD, STRIDE, WINDOW_S, MAX_DRIFT, MAX_DEV,
                         SMOOTH_S, MAX_STEP)
from batch_group_plots import C_LARGEST, roll_mean, smooth_window

C_STEADY = "#fdd49e"
C_LEVEL = "#d94801"


def largest_series(trial):
    """-> (steps, largest per frame), from batch_group_plots' cache if present."""
    for p in glob.glob(os.path.join(trial, "*_series_d65_*.npz")):
        if p.endswith("_nosing.npz"):
            continue
        with np.load(p) as z:
            return np.round(z["secs"] * FPS).astype(int), z["largest"]
    pa = glob.glob(os.path.join(trial, "*_POS_ALL.csv"))[0]
    df = pd.read_csv(pa, usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    step = df["Step"].to_numpy()
    xy = df[["X", "Y"]].to_numpy(float)
    th = df["Theta"].to_numpy(float)
    aid = df["Agent_ID"].to_numpy()
    steps, starts = np.unique(step, return_index=True)
    b = np.append(starts, len(step))
    largest = np.array([max(g["size"] for g in compute_frame_groups(
        xy[b[k]:b[k + 1]], th[b[k]:b[k + 1]], aid[b[k]:b[k + 1]], MAXD, True))
        for k in range(len(steps))], dtype=float)
    return steps, largest


def plot_trial(args):
    trial, formats = args
    steps, largest = largest_series(trial)
    secs = steps / FPS
    sel = steps % STRIDE == 0
    segs = steady_states(secs[sel], largest[sel], np.ones(sel.sum()),
                         window_s=WINDOW_S, max_drift=MAX_DRIFT,
                         max_dev=MAX_DEV, cv_smooth_s=SMOOTH_S,
                         max_step=MAX_STEP)
    name = os.path.basename(os.path.normpath(trial))
    exp = os.path.basename(os.path.dirname(os.path.normpath(trial)))

    fig, ax = plt.subplots(figsize=(11, 4.4))
    for i, g in enumerate(segs):
        ax.axvspan(g["t0"], g["t1"], color=C_STEADY, alpha=0.55, lw=0, zorder=0,
                   label="steady (counted)" if i == 0 else None)
        ax.hlines(g["largest"], g["t0"], g["t1"], colors=C_LEVEL, lw=1.3,
                  linestyles=(0, (5, 3)), zorder=3)
        ax.text(0.5 * (g["t0"] + g["t1"]), 25.3,
                f"{g['t0']:.0f}–{g['t1']:.0f} s\nmean {g['largest']:.2f}",
                ha="center", va="top", fontsize=8, color="#a63603", zorder=4)
    ax.plot(secs, largest, lw=0.8, color=C_LARGEST, alpha=0.3)
    ax.plot(secs, roll_mean(largest, smooth_window(secs, 2.0)), lw=2.2,
            color=C_LARGEST, label="largest group")
    ax.set_xlim(secs[0], secs[-1])
    ax.set_ylim(0, 26)
    ax.grid(alpha=0.25)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("largest group size")
    dur = sum(g["dur"] for g in segs)
    ax.set_title(f"Largest group size over time  [{exp}/{name}]", fontsize=12,
                 pad=18)
    ax.text(0.5, 1.012,
            f"steady: {len(segs)} segment(s), {dur / (secs[-1] - secs[0]):.0%} "
            f"of the trial  |  {WINDOW_S:g}s window, drift ≤ {MAX_DRIFT:.0%}, "
            f"{SMOOTH_S:g}s mean within ±{MAX_DEV:.0%}, merge step ≤ {MAX_STEP:.0%}",
            transform=ax.transAxes, ha="center", fontsize=8, color="#555")
    # bottom right: the segment labels sit along the top, and group size is
    # never near zero late in a trial
    ax.legend(frameon=False, loc="lower right", fontsize=9, ncol=2)
    fig.tight_layout()
    out = []
    for f in formats:
        p = os.path.join(trial, f"{name}_largest_steady.{f}")
        fig.savefig(p, dpi=150)
        out.append(p)
    plt.close(fig)
    return trial, segs, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("targets", nargs="+",
                    help="condition folders (every trial_* inside) or trial folders")
    ap.add_argument("--formats", nargs="+", default=["png", "svg"],
                    choices=["png", "svg", "pdf"])
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    trials = []
    for t in args.targets:
        if glob.glob(os.path.join(t, "*_POS_ALL.csv")):
            trials.append(t)
        else:
            trials += sorted(d for d in glob.glob(os.path.join(t, "trial_*"))
                             if glob.glob(os.path.join(d, "*_POS_ALL.csv")))
    with Pool(max(1, min(args.jobs, len(trials)))) as pool:
        for trial, segs, out in pool.imap(plot_trial,
                                          [(t, args.formats) for t in trials]):
            print(trial, [(g["t0"], g["t1"], round(g["largest"], 2)) for g in segs])
            for p in out:
                print("  ->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

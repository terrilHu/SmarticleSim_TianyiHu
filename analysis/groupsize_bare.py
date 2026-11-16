"""
groupsize_bare.py
-----------------
The per-trial groupsize figure of batch_group_plots.py / real_group_plots.py with
only the largest group (thin raw trace + smoothed line) and the per-robot mean
group size, and no text at all: no title, axis labels, tick labels or legend.
Every figure from one call shares the same y range and ticks (and x range), so
hardware and simulation panels can be laid side by side.

Inputs, any mix of:
    a hardware directory with group<stamp>.csv (+ command<stamp>.csv)  -> same
        series and t = 0 as real_group_plots.py (main-command window)
    a simulation trial directory with trial_XXXX_series_d<d>_b*.npz     -> the
        series batch_group_plots.py cached for its groupsize figure

Output, next to each original figure: <stamp>_groupsize_bare.png/.svg in
<hardware dir>/group_plots/, trial_XXXX_groupsize_bare.png/.svg in the trial dir.

Usage:
    python analysis/groupsize_bare.py <hw_dir> <sim_trial_dir> [...] [--ylim 1 13] [--xmax 500]
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_group_plots as bgp                       # noqa: E402
from real_group_plots import find_real_trials, real_trial_series   # noqa: E402


def collect(paths, max_dist):
    """-> [(series, output path stem)]"""
    out = []
    for p in paths:
        p = os.path.normpath(p)
        real = find_real_trials(p)
        if real:
            for stamp, gcsv, ccsv in real:
                sr, _ = real_trial_series(gcsv, ccsv, "command")
                out.append((sr, os.path.join(p, "group_plots", f"{stamp}_groupsize_bare")))
            continue
        tag = f"{max_dist:g}".replace(".", "p")
        npz = sorted(glob.glob(os.path.join(p, f"*_series_d{tag}_b*.npz")))
        npz = [f for f in npz if not f.endswith("_nosing.npz")]
        if not npz:
            raise SystemExit(f"{p}: neither group*.csv nor a *_series_d{tag}_b*.npz cache")
        if len(npz) > 1:
            print(f"[note] {p}: {len(npz)} series caches, using {os.path.basename(npz[0])}")
        sr = bgp.read_series_cache(npz[0])
        out.append((sr, os.path.join(p, f"{os.path.basename(p)}_groupsize_bare")))
    return out


def draw(sr, stem, ylim, yticks, xlim, smooth_s):
    secs = sr["secs"]
    win = bgp.smooth_window(secs, smooth_s)
    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(secs, sr["largest"], lw=0.8, color=bgp.C_LARGEST, alpha=0.30)
    ax.plot(secs, bgp.roll_mean(sr["largest"], win), lw=2.2, color=bgp.C_LARGEST)
    ax.plot(secs, bgp.roll_mean(sr["mean_size"], win), lw=2.2, color=bgp.C_MEANSZ)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_yticks(yticks)
    ax.grid(alpha=0.25)
    ax.tick_params(labelbottom=False, labelleft=False)
    fig.tight_layout()
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(f"{stem}.{ext}", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--ylim", nargs=2, type=float, default=None,
                    help="shared y range (default: integers around every series' range)")
    ap.add_argument("--ystep", type=float, default=2)
    ap.add_argument("--xmax", type=float, default=None, help="shared x range end (default: longest trial)")
    ap.add_argument("--max-dist", type=float, default=65)
    ap.add_argument("--smooth-s", type=float, default=2.0)
    a = ap.parse_args()

    items = collect(a.paths, a.max_dist)
    if a.ylim:
        ylim = tuple(a.ylim)
    else:
        lo = min(min(sr["largest"].min(), sr["mean_size"].min()) for sr, _ in items)
        hi = max(max(sr["largest"].max(), sr["mean_size"].max()) for sr, _ in items)
        ylim = (np.floor(lo) - 0.5, np.ceil(hi) + 0.5)
    start = np.ceil(ylim[0] / a.ystep) * a.ystep
    yticks = np.arange(start, ylim[1] + 1e-9, a.ystep)
    xlim = (0.0, a.xmax or max(float(sr["secs"][-1]) for sr, _ in items))
    print(f"shared y {ylim}, ticks {yticks.tolist()}, x {xlim}")
    for sr, stem in items:
        draw(sr, stem, ylim, yticks, xlim, a.smooth_s)
        print(f"  largest peak {sr['largest'].max():.0f}, t_end {sr['secs'][-1]:.1f}s -> {stem}.png/.svg")


if __name__ == "__main__":
    main()

"""
plot_trial.py  -  one overview figure for a single trial.

`batch_group_plots.py` already draws per-trial figures, and it accepts a single
trial folder. This is for the other job: looking at one run closely. It puts
everything on **one shared time axis** so the panels can be read against each
other, and it uses the per-trial files the batch script ignores -- coverage,
alignment, the order parameters and the gait log -- which only mean anything
before they are averaged away.

It also shades the steady segments `steady_state.py` finds, so you can see
which part of the trial the sweep statistics were actually taken from.

    python analysis/plot_trial.py ".../synchronized/a826/trial_0000"
    python analysis/plot_trial.py ".../a826" --trial 3
    python analysis/plot_trial.py ".../a826/trial_0000" --formats png svg
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The simulation modules (pos_all_grouping, gait) live one level up.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pos_all_grouping import compute_frame_groups
from steady_state import steady_segments
from batch_group_plots import (SIZE_COLORS, CAT_COLORS, C_LARGEST, C_MEANSZ,
                               C_ALIGN, C_OTHER, size_bins, roll_mean,
                               smooth_window, describe_command, stacked_bands,
                               find_pos_all, read_snapshot)

MAXD_DEFAULT = 65.0
C_STEADY = "#e8eef7"      # the shaded steady stretches
MUTED = "#55534e"


def trial_files(folder):
    """-> dict of the per-trial files present, keyed by their suffix."""
    out = {}
    for p in glob.glob(os.path.join(folder, "*")):
        b = os.path.basename(p)
        for key in ("POS_ALL", "coverage", "alignment", "GRs_ORDs_Mdist",
                    "gait_log", "impacts"):
            if b.endswith(f"_{key}.csv"):
                out[key] = p
    return out


def group_series(pos_all, max_dist, stride, bins):
    """
    Per-frame grouping for one trial. -> dict of equal-length series.

    Strided: the panels are a few thousand pixels wide at most, and grouping
    every frame of a 30k-frame trial costs 6x more for detail no one can see.
    """
    df = pd.read_csv(pos_all, usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    if stride > 1:
        df = df[df["Step"] % stride == 0]
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    step = df["Step"].to_numpy()
    xy = df[["X", "Y"]].to_numpy(float)
    th = df["Theta"].to_numpy(float)
    aid = df["Agent_ID"].to_numpy()
    steps, starts = np.unique(step, return_index=True)
    b = np.append(starts, len(step))
    n = len(steps)

    largest = np.empty(n); mean_sz = np.empty(n); align = np.empty(n)
    n_groups = np.empty(n); n_multi = np.empty(n); n_seen = np.empty(n)
    frac = np.zeros((n, len(bins)))
    for k in range(n):
        sl = slice(b[k], b[k + 1])
        gs = compute_frame_groups(xy[sl], th[sl], aid[sl], max_dist, True)
        s = np.array([g["size"] for g in gs], dtype=np.int64)
        tot = s.sum()
        largest[k] = s.max()
        mean_sz[k] = (s * s).sum() / tot
        align[k] = sum(g["size"] * g["alignment"] for g in gs) / max(tot, 1)
        n_groups[k] = len(gs)
        n_multi[k] = int((s >= 2).sum())
        n_seen[k] = tot
        for g in gs:
            for i, (lo, hi) in enumerate(bins):
                if lo <= g["size"] <= hi:
                    frac[k, i] += g["size"] / max(tot, 1)
                    break
    return {"steps": steps, "largest": largest, "mean_size": mean_sz,
            "alignment": align, "n_groups": n_groups, "n_multi": n_multi,
            "n_seen": n_seen, "frac": frac,
            "largest_frac": largest / np.maximum(n_seen, 1)}


def gait_shares(gait_log, secs):
    """-> (share array (T,k), command list) forward-filled onto `secs`."""
    df = pd.read_csv(gait_log).sort_values("time", kind="mergesort")
    cmds = sorted(df["command"].unique().tolist())
    slot = {c: i for i, c in enumerate(cmds)}
    counts = np.zeros(len(cmds), dtype=int)
    cur = {}
    ev = df[["time", "robot_id", "command"]].to_numpy()
    out = np.zeros((len(secs), len(cmds)))
    j = 0
    for i, t in enumerate(secs):
        while j < len(ev) and ev[j][0] <= t:
            s = slot[int(ev[j][2])]
            prev = cur.get(ev[j][1])
            if prev is not None:
                counts[prev] -= 1
            counts[s] += 1
            cur[ev[j][1]] = s
            j += 1
        tot = counts.sum()
        if tot:
            out[i] = counts / tot
    return out, cmds


def shade(ax, segs, secs):
    """Mark the steady stretches behind everything else."""
    for i, (a, b) in enumerate(segs):
        ax.axvspan(secs[a], secs[b], color=C_STEADY, zorder=0,
                   label="steady" if i == 0 else None)


def figure(data, meta, path, formats, smooth_s, window_s, max_drift,
           max_dev=None, band_smooth_s=None, max_step=None):
    secs = data["secs"]
    bins, labels = data["bins"], data["labels"]
    win = smooth_window(secs, smooth_s)
    segs = steady_segments(secs, data["largest"], window_s, max_drift,
                           max_dev=max_dev, cv_smooth_s=band_smooth_s,
                           max_step=max_step)

    rows = 4 + int(data.get("gait") is not None) + int(data.get("cov") is not None)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 2.05 * rows + 1.6),
                             sharex=True)
    ax = list(np.atleast_1d(axes))
    i = 0

    # --- group size -------------------------------------------------------
    a = ax[i]; i += 1
    shade(a, segs, secs)
    a.plot(secs, data["largest"], lw=0.7, color=C_LARGEST, alpha=0.30)
    a.plot(secs, roll_mean(data["largest"], win), lw=2.0, color=C_LARGEST,
           label="largest group")
    a.plot(secs, roll_mean(data["mean_size"], win), lw=2.0, color=C_MEANSZ,
           label="mean group size (per robot)")
    a.set_ylabel("group size")
    a.legend(frameon=False, ncol=3, fontsize=8, loc="upper left")

    # --- composition ------------------------------------------------------
    a = ax[i]; i += 1
    fr = np.column_stack([roll_mean(data["frac"][:, b], win)
                          for b in range(data["frac"].shape[1])])
    stacked_bands(a, secs, fr.T * 100, SIZE_COLORS[:len(labels)], labels,
                  edgecolor="none", linewidth=0.0)
    a.set_ylim(0, 100); a.set_ylabel("share of robots (%)")
    a.legend(frameon=False, ncol=len(labels), fontsize=7.5, loc="upper left",
             title="group size", title_fontsize=7.5)

    # --- counts -----------------------------------------------------------
    a = ax[i]; i += 1
    shade(a, segs, secs)
    a.plot(secs, roll_mean(data["n_groups"], win), lw=1.6, color="#666",
           label="all groups")
    a.plot(secs, roll_mean(data["n_multi"], win), lw=1.6, color="#2171b5",
           label="groups with >=2")
    a.set_ylabel("count")
    a.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")

    # --- alignment --------------------------------------------------------
    a = ax[i]; i += 1
    shade(a, segs, secs)
    a.plot(secs, roll_mean(data["alignment"], win), lw=1.8, color=C_ALIGN)
    a.set_ylim(0, 1.05)
    a.set_ylabel("nematic alignment")

    # --- coverage ---------------------------------------------------------
    if data.get("cov") is not None:
        a = ax[i]; i += 1
        shade(a, segs, secs)
        ct, ck = data["cov"]
        a.plot(ct, ck, lw=1.4, color="#8c6bb1")
        a.set_ylabel("coverage k")

    # --- gait pattern -----------------------------------------------------
    if data.get("gait") is not None:
        a = ax[i]; i += 1
        sh, cmds = data["gait"]
        cols = [CAT_COLORS[k % len(CAT_COLORS)] for k in range(len(cmds))]
        stacked_bands(a, secs, (sh * 100).T, cols,
                      [describe_command(c) for c in cmds],
                      edgecolor="white", linewidth=1.0)
        a.set_ylim(0, 100); a.set_ylabel("gait share (%)")
        a.legend(frameon=False, ncol=min(4, len(cmds)), fontsize=7.5,
                 loc="upper left")

    for a in ax:
        a.grid(alpha=0.2)
        a.set_xlim(secs[0], secs[-1])
    ax[-1].set_xlabel("time (s)")

    steady_s = sum(secs[b] - secs[a] for a, b in segs)
    fig.suptitle(meta["title"], fontsize=13, y=0.995)
    fig.text(0.5, 0.967, meta["sub"], ha="center", fontsize=8.5, color=MUTED)
    fig.text(0.5, 0.947,
             f"shaded = steady ({len(segs)} segment(s), "
             f"{steady_s/max(secs[-1]-secs[0], 1e-9):.0%} of the trial; "
             f"sliding {window_s:g}s window, drift <= {max_drift:.0%}"
             + (f", {band_smooth_s:g}s mean within +-{max_dev:.0%}" if max_dev else "")
             + (f", merge step <= {max_step:.0%}" if max_step else "")
             + ")",
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=(0, 0, 1, 0.941))

    base = os.path.splitext(path)[0]
    written = []
    for f in formats:
        p = f"{base}.{f}"
        fig.savefig(p, dpi=150)
        written.append(p)
    plt.close(fig)
    return written


def resolve(target, trial_index):
    """Accept either a trial folder or an experiment folder plus --trial."""
    if find_pos_all(target):
        return target
    subs = sorted(d for d in glob.glob(os.path.join(target, "trial_*"))
                  if os.path.isdir(d) and find_pos_all(d))
    if not subs:
        return None
    if trial_index is None:
        return subs[0]
    want = os.path.join(target, f"trial_{trial_index:04d}")
    return want if want in subs else None


def main():
    ap = argparse.ArgumentParser(
        description="One overview figure for a single trial, on a shared time axis.")
    ap.add_argument("target", help="a trial folder, or an experiment folder (with --trial)")
    ap.add_argument("--trial", type=int, default=None,
                    help="trial index when the target is an experiment folder (default: the first)")
    ap.add_argument("--out", default=None, help="output path (default: inside the trial folder)")
    ap.add_argument("--max-dist", type=float, default=MAXD_DEFAULT,
                    help=f"grouping threshold in px (default {MAXD_DEFAULT:g})")
    ap.add_argument("--stride", type=int, default=10,
                    help="use every Nth recorded frame (default 10, i.e. 6 Hz)")
    ap.add_argument("--smooth-s", type=float, default=2.0,
                    help="moving average for the line panels, in seconds; 0 disables")
    ap.add_argument("--window-s", type=float, default=60.0,
                    help="window for steady-state detection (default 60)")
    ap.add_argument("--max-drift", type=float, default=0.08,
                    help="fractional drift a steady window may show (default 0.08)")
    ap.add_argument("--max-dev", type=float, default=0.15,
                    help="band: the smoothed signal may not stray more than this fraction "
                         "of the window mean; 0 disables (default 0.15)")
    ap.add_argument("--smooth-band-s", type=float, default=10.0,
                    help="moving average the band is judged on, in seconds (default 10)")
    ap.add_argument("--max-step", type=float, default=0.08,
                    help="merge steady windows only while their means stay within this "
                         "fraction; 0 merges any touching windows (default 0.08)")
    ap.add_argument("--formats", nargs="+", default=["png"],
                    choices=["png", "svg", "pdf"],
                    help="output formats (default png)")
    ap.add_argument("--no-coverage", dest="coverage", action="store_false",
                    help="skip the coverage panel even if coverage.csv exists")
    args = ap.parse_args()

    folder = resolve(args.target, args.trial)
    if not folder:
        print(f"No trial with a *_POS_ALL.csv under {args.target}"
              + (f" (trial {args.trial})" if args.trial is not None else ""))
        return 1
    files = trial_files(folder)
    name = os.path.basename(os.path.normpath(folder))
    exp = os.path.basename(os.path.dirname(os.path.normpath(folder)))
    snap = read_snapshot(folder)
    n_rob = snap.get("N_SMARTICLES")
    fps = float(snap.get("RENDER_FPS_HEADLESS", 60.0))

    print(f"{exp}/{name}: {', '.join(sorted(files))}")
    bins, labels = size_bins(int(n_rob) if n_rob else 100)
    g = group_series(files["POS_ALL"], args.max_dist, max(1, args.stride), bins)
    secs = g["steps"] / fps
    data = dict(g, secs=secs, bins=bins, labels=labels)

    if args.coverage and "coverage" in files:
        c = pd.read_csv(files["coverage"])
        data["cov"] = (c["time"].to_numpy(), c["k"].to_numpy())
    if "gait_log" in files:
        sh, cmds = gait_shares(files["gait_log"], secs)
        # a single command for the whole run is a flat 100% band: no information
        data["gait"] = (sh, cmds) if len(cmds) > 1 else None

    cmd0 = snap.get("COMMAND_ARRAY")
    cmd0 = cmd0[0] if isinstance(cmd0, list) and cmd0 else None
    meta = {
        "title": f"{exp} / {name}",
        "sub": (f"N={n_rob}, {describe_command(cmd0) if cmd0 is not None else '?'}"
                f",  max_dist={args.max_dist:g}px,  "
                f"{len(secs)} samples at {fps/max(1,args.stride):g} Hz"),
    }
    out = args.out or os.path.join(folder, f"{name}_overview.png")
    for p in figure(data, meta, out, args.formats, args.smooth_s,
                    args.window_s, args.max_drift,
                    args.max_dev or None, args.smooth_band_s,
                    args.max_step or None):
        print("->", p)
    return 0


if __name__ == "__main__":
    sys.exit(main())

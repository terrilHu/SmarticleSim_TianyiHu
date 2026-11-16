"""
catalyst_radial_build.py
------------------------
Per-trial reduction for catalyst_radial_stats.py. For every trial of the given
experiments, writes one compact <out>/cache/<exp>__<trial>.npz holding where in
the arena the groups are, frame by frame.

Reuses the standard pipeline: batch_group_plots.write_groups_csv (Voronoi
groups, cached next to POS_ALL as *_groups_d65.csv), then group_lifetimes'
hysteresis and tracking (1 s window, min size 3, larger keeps the ID).

Tracks and events are also saved the way group_lifetimes.py saves them, except
that files which already exist are compared to the recomputation and left alone,
so an existing baseline is not overwritten.

Stops before computing a groups cache if the target disk has < --min-free-gb left.

Usage:
    python catalyst_radial_build.py <exp_dir | NAME=exp_dir> [...] --out DIR [--jobs 5]

NAME=exp_dir names the cache files NAME__trial_XXXX.npz instead of after the dir's basename.
"""
import argparse
import glob
import json
import multiprocessing as mp
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

SIZE_EDGES = np.array([1, 3, 6, 10, 20, 10**6])   # [1-2], [3-5], [6-9], [10-19], [20+]
N_BINS = 20                                       # equal-area bins of r/R in [0, 1.1]
RHO_MAX2 = 1.21


def load_xy(pos_csv, T, N):
    df = pd.read_csv(pos_csv, usecols=["Step", "Agent_ID", "X", "Y"],
                     dtype={"Step": np.int32, "Agent_ID": np.int32, "X": np.float32, "Y": np.float32})
    xy = np.full((T, N, 2), np.nan, np.float32)
    xy[df.Step.to_numpy(), df.Agent_ID.to_numpy() - 1] = df[["X", "Y"]].to_numpy()
    return xy


CATALYST_CMDS = (-52, -852)    # strategy runs log the catalyst gait as -52, fixed-catalyst runs as -852


def role_matrix(gait_csv, T, N, fps, catalyst_cmd=CATALYST_CMDS):
    """(T, N) bool: robot is running a catalyst command in that frame (forward-filled log).

    The log's robot_id is 0-based, POS_ALL's Agent_ID 1-based; columns here are 0-based.
    """
    g = pd.read_csv(gait_csv).sort_values("time", kind="mergesort")
    out = np.zeros((T, N), bool)
    for rid, sub in g.groupby("robot_id"):
        frames = np.minimum(np.round(sub.time.to_numpy() * fps).astype(int), T)
        vals = np.isin(sub.command.to_numpy(), catalyst_cmd)
        for a, b, v in zip(frames, np.append(frames[1:], T), vals):
            if v:
                out[a:b, rid] = True
    return out


def group_sizes(S):
    """(T, N) size of each robot's smoothed group."""
    size = np.empty(S.shape, np.int32)
    for t in range(S.shape[0]):
        _, inv, cnt = np.unique(S[t], return_inverse=True, return_counts=True)
        size[t] = cnt[inv]
    return size


def radial_hist(rho, size, is_cat):
    """(T, bins, size classes, 2) robot counts; last axis is [other, catalyst]."""
    T, N = rho.shape
    nc = len(SIZE_EDGES) - 1
    b = np.clip((rho ** 2 / RHO_MAX2 * N_BINS).astype(np.int32), 0, N_BINS - 1)
    c = np.searchsorted(SIZE_EDGES, size, side="right") - 1
    key = ((np.arange(T)[:, None] * N_BINS + b) * nc + c) * 2 + is_cat.astype(np.int32)
    return np.bincount(key.ravel(), minlength=T * N_BINS * nc * 2).reshape(T, N_BINS, nc, 2).astype(np.int16)


def process(job):
    exp, exp_dir, trial_dir, out_dir, min_free_gb = job
    from batch_group_plots import write_groups_csv
    from group_lifetimes import load_labels, smooth_labels, track

    trial = os.path.basename(trial_dir)
    out = os.path.join(out_dir, "cache", f"{exp}__{trial}.npz")
    if os.path.exists(out):
        return f"{exp}/{trial}: cached"
    t0 = time.time()
    cfg = json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))
    fps, N, R = cfg.get("RENDER_FPS_HEADLESS", 60), cfg["N_SMARTICLES"], float(cfg["INNER_R"])
    W, H = float(cfg["W"]), float(cfg["H"])
    prefix = os.path.join(trial_dir, trial)
    groups_csv, pos_csv = prefix + "_groups_d65.csv", prefix + "_POS_ALL.csv"

    if not os.path.exists(groups_csv):
        free = shutil.disk_usage(trial_dir).free / 1e9
        if free < min_free_gb:
            raise SystemExit(f"{free:.1f} GB free on the data disk; refusing to write {groups_csv}")
        write_groups_csv(pos_csv, groups_csv, 65.0)

    L = load_labels(groups_csv)
    T = L.shape[0]
    S = smooth_labels(L, int(round(1.0 * fps / 2)))
    del L
    xy = load_xy(pos_csv, T, N)

    rows = []

    def on_frame(t, groups):
        for tid, _, s, idx in groups:
            c = xy[t, idx].mean(axis=0)
            rows.append((t, tid, s, c[0], c[1]))

    tr, ev = track(S, 3, on_frame=on_frame)
    tr["lifetime_s"] = tr.n_frames / fps
    ev.insert(1, "time_s", ev.step / fps)
    for c in ("t_start", "t_end"):
        tr[c.replace("t_", "time_") + "_s"] = tr[c] / fps

    note = ""
    tag = "d65_m3_w1"
    tr_csv, ev_csv = f"{prefix}_tracks_{tag}.csv", f"{prefix}_events_{tag}.csv"
    if os.path.exists(tr_csv):
        old = pd.read_csv(tr_csv)
        keys = ["t_start", "t_end", "max_size", "n_frames"]
        same = len(old) == len(tr) and all((old[k].to_numpy() == tr[k].to_numpy()).all() for k in keys)
        note = " existing tracks identical" if same else f" EXISTING TRACKS DIFFER ({len(old)} vs {len(tr)})"
    else:
        tr.to_csv(tr_csv, index=False)
        ev.to_csv(ev_csv, index=False)

    frames = np.array(rows, np.float64)
    swarm_c = np.nanmean(xy, axis=1)
    arena_c = np.array([W / 2, H / 2], np.float32)
    ring_c = np.empty((0, 2), np.float32)
    render = prefix + "_render.npz"
    if os.path.exists(render):
        with np.load(render) as z:
            ring_c = z["ring"].mean(axis=1)

    gait = prefix + "_gait_log.csv"
    # fixed-catalyst runs have runtime gait control off but log their -852 robots at t = 0
    is_cat = role_matrix(gait, T, N, fps) if os.path.exists(gait) else np.zeros((T, N), bool)
    size = group_sizes(S)
    rho_s = np.hypot(*(xy - swarm_c[:, None]).transpose(2, 0, 1)) / R
    rho_a = np.hypot(*(xy - arena_c).transpose(2, 0, 1)) / R
    tr_cols = ["track_id", "t_start", "t_end", "birth", "death", "max_size", "size_start", "size_end",
               "n_absorbed", "n_shed", "censored_left", "censored_right"]
    np.savez_compressed(out,
                        frames_t=frames[:, 0].astype(np.int32), frames_id=frames[:, 1].astype(np.int32),
                        frames_size=frames[:, 2].astype(np.int16), frames_xy=frames[:, 3:5].astype(np.float32),
                        swarm_c=swarm_c, arena_c=arena_c, ring_c=ring_c, R=R, fps=fps, N=N, T=T,
                        hist_swarm=radial_hist(rho_s, size, is_cat), hist_arena=radial_hist(rho_a, size, is_cat),
                        n_catalyst=is_cat.sum(axis=1).astype(np.int16),
                        tracks=tr[tr_cols].to_records(index=False), events=ev.to_records(index=False))
    return f"{exp}/{trial}: {time.time() - t0:.0f} s{note}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exps", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=5)
    ap.add_argument("--min-free-gb", type=float, default=8.0)
    a = ap.parse_args()
    os.makedirs(os.path.join(a.out, "cache"), exist_ok=True)
    jobs = []
    for e in a.exps:
        # NAME=DIR names the cache after NAME, for two experiment dirs with the same basename
        name, e = e.split("=", 1) if "=" in e and not os.path.isdir(e) else (os.path.basename(os.path.normpath(e)), e)
        e = os.path.abspath(e)
        jobs += [(name, e, t, a.out, a.min_free_gb) for t in sorted(glob.glob(os.path.join(e, "trial_*")))]
    with mp.Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(process, jobs):
            print(msg, flush=True)


if __name__ == "__main__":
    main()

"""
render_group_video.py
---------------------
A trial's whole video, redrawn in the render_snapshots.py style: each robot body
coloured by the size class of its group in that frame (1-2 grey, 3-5 light blue,
6+ dark blue), arms in neutral grey, the ring as a thin grey outline, catalysts
(gait log command -52 / -852) with an orange border, no text. Nothing is
simulated: poses come from POS_ALL, arms and ring from *_render.npz, groups from
the trial_XXXX_groups_d<d>.csv cache (written first if it is missing).

One frame per render.npz frame, at the recorded video rate. The view is cropped
to everything the robots and ring reach over the whole trial, plus --margin.

Usage:
    python analysis/render_group_video.py <exp_dir | trial_dir> [...] [--out-dir DIR] [--downscale 1]

Output: <out-dir>/<exp>_<trial>_groups.mp4 (default out-dir: the trial folder).
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

from batch_group_plots import read_snapshot, find_pos_all, write_groups_csv
from catalyst_radial_build import role_matrix
from render_snapshots import draw


def trial_dirs(paths):
    out = []
    for p in paths:
        p = os.path.normpath(p)
        if glob.glob(os.path.join(p, "*_render.npz")):
            out.append(p)
        else:
            out += sorted(d for d in glob.glob(os.path.join(p, "trial_*")) if os.path.isdir(d))
    return out


def size_classes(groups_csv, steps, n):
    """(len(steps), n) size-class labels; Agent_ID k is column k - 1."""
    g = pd.read_csv(groups_csv, usecols=["Step", "size", "member_ids"], dtype={"member_ids": str})
    g = g[g.Step.isin(steps) & (g["size"] >= 3)]
    row = {s: i for i, s in enumerate(steps)}
    size = np.ones((len(steps), n), np.int32)
    for s, sz, mem in g.itertuples(index=False):
        size[row[s], np.array(mem.split(), int) - 1] = sz
    return np.where(size >= 6, "6+", np.where(size >= 3, "3-5", "1-2"))


def render(tdir, out_dir, max_dist, ds, margin):
    exp, trial = os.path.basename(os.path.dirname(tdir)), os.path.basename(tdir)
    snap = read_snapshot(tdir)
    W, H = int(snap["W"]), int(snap["H"])
    z = np.load(glob.glob(os.path.join(tdir, "*_render.npz"))[0])
    arms, ring = z["arms"], z["ring"]
    stride, fps = int(z["stride"]), float(z["fps"])

    pos = pd.read_csv(find_pos_all(tdir), usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    pos = pos.sort_values(["Step", "Agent_ID"], kind="mergesort")
    all_steps = pos.Step.unique()
    n = len(pos) // len(all_steps)
    xyt = pos[["X", "Y", "Theta"]].to_numpy(float).reshape(len(all_steps), n, 3)
    nf = min(len(arms), len(ring), (len(all_steps) - 1) // stride + 1)
    steps = all_steps[::stride][:nf]
    xyt = xyt[::stride][:nf]

    gcsv = os.path.join(tdir, f"{trial}_groups_d{max_dist:g}.csv")
    if not os.path.exists(gcsv):
        write_groups_csv(find_pos_all(tdir), gcsv, float(max_dist))
    cls = size_classes(gcsv, steps, n)

    gait = os.path.join(tdir, f"{trial}_gait_log.csv")
    T = int(all_steps[-1]) + 1
    cat = (role_matrix(gait, T, n, snap.get("RENDER_FPS_HEADLESS", 60))[steps]
           if os.path.exists(gait) else np.zeros((nf, n), bool))

    pts = np.vstack([xyt[:, :, :2].reshape(-1, 2), ring[:nf].reshape(-1, 2)])
    x0, y0 = np.maximum(np.floor(pts.min(0)) - margin, 0).astype(int)
    x1, y1 = np.minimum(np.ceil(pts.max(0)) + margin, (W, H)).astype(int)
    # even output size, which most mp4 players want
    x1 -= (x1 - x0) // ds % 2 * ds
    y1 -= (y1 - y0) // ds % 2 * ds
    crop = (x0, y0, x1, y1)

    from analysis import VideoRecorder
    out = os.path.join(out_dir or tdir, f"{exp}_{trial}_groups.mp4")
    rec = VideoRecorder(out, (x1 - x0) // ds, (y1 - y0) // ds, int(round(fps)), downscale=1)
    t0 = time.perf_counter()
    for k in range(nf):
        fr = dict(xyt=xyt[k], cls=cls[k], arms=arms[k], ring=ring[k])
        rec.add_frame_from_surface(draw(snap, fr, crop, ds, bodies_only=False,
                                        mark=cat[k] if cat[k].any() else None))
        if k % 2000 == 0:
            print(f"  {trial}: {k}/{nf}", flush=True)
    rec.close()
    big = (cls == "6+").mean() * 100
    print(f"-> {out}  ({nf} frames, {fps:g} fps, {(x1 - x0) // ds}x{(y1 - y0) // ds}, "
          f"{time.perf_counter() - t0:.0f} s; robots in >=6 groups {big:.1f}% of robot-frames, "
          f"catalyst robot-frames {int(cat.sum())})", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="experiment dirs (all trials) or trial dirs")
    ap.add_argument("--out-dir", default=None, help="default: each trial folder")
    ap.add_argument("--max-dist", type=float, default=65)
    ap.add_argument("--downscale", type=int, default=1)
    ap.add_argument("--margin", type=int, default=30)
    a = ap.parse_args()
    if a.out_dir:
        os.makedirs(a.out_dir, exist_ok=True)
    pygame.init()
    for t in trial_dirs(a.paths):
        render(t, a.out_dir, a.max_dist, a.downscale, a.margin)
    pygame.quit()


if __name__ == "__main__":
    main()

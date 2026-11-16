"""
render_snapshots.py
-------------------
Still frames of a baseline/strategy trial pair at chosen times, drawn the same
way as render_trial.py but only for those frames: every frame is redrawn from
that frame's recorded state alone, so this gives exactly the frames a full
re-render would contain without writing (or holding) a video.

Each robot body is coloured by the size class of the group it is in at that
frame (1-2 / 3-5 / 6+, from the trial_XXXX_groups_d<d>.csv cache, i.e. the
same grouping as the analysis); arms are drawn in neutral grey.

A trial without *_render.npz (it predates SAVE_RENDER_STATE) has no arm angles
or ring outline: it is drawn bodies-only, and the pair's side-by-side figure
then draws both runs bodies-only so the two rows are comparable. The strategy
run's full render is still saved on its own.

Only the frames needed are read from POS_ALL and the groups cache (in chunks),
and free RAM / disk are checked before each run.

Output (in --out-dir):
    <exp>_<trial>_t<sec>s.png               one per run and time
    <base>_vs_<strat>_<trial>_snapshots.png  2 x len(times) grid, with per-frame stats

Usage:
    python render_snapshots.py <base_exp_dir> <strat_exp_dir> --trial trial_0002 --times 0 300 500 --out-dir DIR
"""

import argparse
import glob
import os
import shutil
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

from batch_group_plots import read_snapshot, find_pos_all
from render_trial import box, rot, hex_rgb

INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
CLASS_COLORS = {"1-2": "#c4c4c4", "3-5": "#9ecae1", "6+": "#2171b5"}     # batch_group_plots.SIZE_COLORS family
ARM_COLOR = "#8a8984"
RING_COLOR = "#95a5a6"
MIN_FREE_RAM_GB, MIN_FREE_DISK_GB = 2.0, 2.0


def check_resources(path):
    disk = shutil.disk_usage(path).free / 2**30
    try:
        import psutil
        ram = psutil.virtual_memory().available / 2**30
    except ImportError:
        ram = float("nan")
        if sys.platform == "win32":
            import ctypes

            class MemStatus(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            st = MemStatus()
            st.dwLength = ctypes.sizeof(MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            ram = st.ullAvailPhys / 2**30
    print(f"  free RAM {ram:.1f} GB, free disk {disk:.1f} GB")
    if disk < MIN_FREE_DISK_GB or ram < MIN_FREE_RAM_GB:
        raise SystemExit(f"stopping: below {MIN_FREE_RAM_GB} GB RAM or {MIN_FREE_DISK_GB} GB disk")


def read_steps(csv, steps, usecols):
    """Rows of a large per-frame CSV whose Step is in `steps`, read in chunks."""
    parts = [c[c.Step.isin(steps)] for c in pd.read_csv(csv, usecols=usecols, chunksize=2_000_000)]
    return pd.concat(parts, ignore_index=True)


def load_run(trial_dir, steps, max_dist):
    snap = read_snapshot(trial_dir)
    pose = read_steps(find_pos_all(trial_dir), steps, ["Step", "Agent_ID", "X", "Y", "Theta"])
    groups = read_steps(glob.glob(os.path.join(trial_dir, f"*_groups_d{max_dist}.csv"))[0], steps,
                        ["Step", "size", "member_ids"])
    npz = glob.glob(os.path.join(trial_dir, "*_render.npz"))
    arms = ring = None
    if npz:
        z = np.load(npz[0])
        stride = int(z["stride"]) if "stride" in z else 1
        idx = [s // stride for s in steps]
        arms, ring = z["arms"][idx], z["ring"][idx]
        del z
    frames = []
    for j, s in enumerate(steps):
        p = pose[pose.Step == s].sort_values("Agent_ID")
        ids = p.Agent_ID.to_numpy()
        size_of = pd.Series(1, index=ids)
        for sz, mem in groups[groups.Step == s][["size", "member_ids"]].itertuples(index=False):
            size_of[[int(m) for m in str(mem).split()]] = sz
        cls = np.where(size_of.to_numpy() >= 6, "6+", np.where(size_of.to_numpy() >= 3, "3-5", "1-2"))
        sizes = size_of.to_numpy()
        frames.append(dict(xyt=p[["X", "Y", "Theta"]].to_numpy(float), cls=cls,
                           arms=None if arms is None else arms[j], ring=None if ring is None else ring[j],
                           share6=(sizes >= 6).mean(), share3=(sizes >= 3).mean(),
                           n6=int((groups[(groups.Step == s)]["size"] >= 6).sum())))
    return snap, frames


def draw(snap, fr, crop, ds, bodies_only):
    W, H = int(snap["W"]), int(snap["H"])
    ml, mw, al, aw = (float(snap[k]) for k in ("MAIN_LEN", "MAIN_W", "ARM_LEN", "ARM_W"))
    surf = pygame.Surface((W, H))
    surf.fill((255, 255, 255))
    if fr["ring"] is not None and not bodies_only:
        pygame.draw.polygon(surf, hex_rgb(RING_COLOR), [(float(x), float(y)) for x, y in fr["ring"]], 6)
    colors = {k: hex_rgb(v) for k, v in CLASS_COLORS.items()}
    arm = hex_rgb(ARM_COLOR)
    order = np.argsort([{"1-2": 0, "3-5": 1, "6+": 2}[c] for c in fr["cls"]], kind="stable")   # large groups on top
    for i in order:
        x, y, th = fr["xyt"][i]
        if fr["arms"] is not None and not bodies_only:
            for sign, k in ((-1, 0), (1, 1)):
                hx, hy = rot((sign * ml / 2.0, 0.0), th)
                a = th + float(fr["arms"][i][k])
                ox, oy = rot((sign * al / 2.0, 0.0), a)
                box(surf, x + hx + ox, y + hy + oy, a, al, aw, arm)
        box(surf, x, y, th, ml, mw, colors[fr["cls"][i]])
    x0, y0, x1, y1 = crop
    sub = surf.subsurface(pygame.Rect(x0, y0, x1 - x0, y1 - y0)).copy()
    del surf
    return pygame.transform.smoothscale(sub, ((x1 - x0) // ds, (y1 - y0) // ds))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("strategy")
    ap.add_argument("--trial", required=True)
    ap.add_argument("--times", nargs="+", type=float, default=[0, 300, 500])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--downscale", type=int, default=None, help="default: VIDEO_DOWNSCALE")
    ap.add_argument("--margin", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    pygame.init()

    runs = []
    for exp in (args.baseline, args.strategy):
        tdir = os.path.join(exp, args.trial)
        snap = read_snapshot(tdir)
        fps = snap.get("RENDER_FPS_HEADLESS", 60)
        npz = glob.glob(os.path.join(tdir, "*_render.npz"))
        stride = int(np.load(npz[0])["stride"]) if npz else 2
        # last recorded step that also has render state
        n_steps = int(pd.read_csv(os.path.join(exp, f"{os.path.basename(exp)}_trial_stats.csv"))
                      .set_index("trial").loc[args.trial, "frames"])
        max_step = (n_steps - 1) // stride * stride
        steps = [min(int(round(t * fps)) // stride * stride, max_step) for t in args.times]
        print(f"{os.path.basename(exp)}/{args.trial}: steps {steps}" + ("" if npz else "  (no render.npz: bodies only)"))
        check_resources(args.out_dir)
        snap, frames = load_run(tdir, steps, args.max_dist)
        runs.append(dict(name=os.path.basename(exp), snap=snap, frames=frames, full=bool(npz),
                         secs=[s / fps for s in steps]))

    # one crop for the whole pair: robots and ring of every frame, plus a margin
    pts = np.vstack([np.vstack([f["xyt"][:, :2]] + ([f["ring"]] if f["ring"] is not None else []))
                     for r in runs for f in r["frames"]])
    W, H = int(runs[0]["snap"]["W"]), int(runs[0]["snap"]["H"])
    x0, y0 = (np.floor(pts.min(0)) - args.margin).astype(int)
    x1, y1 = (np.ceil(pts.max(0)) + args.margin).astype(int)
    crop = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    ds = args.downscale or int(runs[0]["snap"].get("VIDEO_DOWNSCALE", 1))
    pair_bodies_only = not all(r["full"] for r in runs)

    grid = []
    for r in runs:
        row = []
        for f, sec in zip(r["frames"], r["secs"]):
            img = draw(r["snap"], f, crop, ds, bodies_only=False)
            path = os.path.join(args.out_dir, f"{r['name']}_{args.trial}_t{sec:03.0f}s.png")
            pygame.image.save(img, path)
            if pair_bodies_only and r["full"]:
                img = draw(r["snap"], f, crop, ds, bodies_only=True)
            row.append(pygame.surfarray.array3d(img).swapaxes(0, 1))
        grid.append(row)
        check_resources(args.out_dir)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "text.color": INK})
    nt = len(args.times)
    h, w = grid[0][0].shape[:2]
    fig, axes = plt.subplots(2, nt, figsize=(4.2 * nt, 4.2 * 2 * h / w + 1.2), facecolor=SURFACE,
                             layout="constrained")
    for i, (r, row) in enumerate(zip(runs, grid)):
        for j, (img, f, sec) in enumerate(zip(row, r["frames"], r["secs"])):
            ax = axes[i, j]
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d0cfca")
            ax.set_title(f"{'baseline' if i == 0 else 'strategy'} ({r['name']}), t = {sec:.0f} s\n"
                         f"≥6: {f['share6']*100:.0f}% of robots in {f['n6']} groups · ≥3: {f['share3']*100:.0f}%",
                         fontsize=9.5)
    handles = [Patch(color=c, label=f"robot in a group of size {k}") for k, c in CLASS_COLORS.items()]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9.5)
    note = ("  ·  bodies only: the baseline has no saved render state (arms, ring)" if pair_bodies_only else "")
    fig.suptitle(f"{runs[0]['name']} vs {runs[1]['name']}, {args.trial}{note}", fontsize=11)
    fig.get_layout_engine().set(rect=(0, 0.04, 1, 0.95))
    out = os.path.join(args.out_dir, f"{runs[0]['name']}_vs_{runs[1]['name']}_{args.trial}_snapshots.png")
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    pygame.quit()
    print(f"wrote {out} and {2 * nt} single frames")


if __name__ == "__main__":
    sys.exit(main())

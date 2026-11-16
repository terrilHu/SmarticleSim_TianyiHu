"""
render_ic.py
------------
One initial condition from an init_conditions_*.json file, drawn the same way
as render_snapshots.py: bodies coloured by the size class of their group
(1-2 / 3-5 / 6+, same Voronoi grouping as the analysis), arms from the stored
thL/thR, ring as the nominal regular n-gon add_ring builds.

This is the state before warmup: the "t = 0 s" frames of a run are recorded
after WARMUP_STEPS, by which point baseline and strategy runs of the same IC
have already drifted apart, so this is the one frame they truly share.

Geometry (W, H, INNER_R, RING_N_SIDES, body sizes) comes from a run that used
this IC (--like <exp_dir>/<trial>), so the crop and scale match its snapshots.

Output (in --out-dir):
    IC<k>_initial.png             the frame alone
    IC<k>_initial_annotated.png   with title, group stats and legend

Usage:
    python render_ic.py <init_conditions.json> 13 --like <exp_dir>/trial_0002 --out-dir DIR
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

from batch_group_plots import read_snapshot, group_table
from render_snapshots import draw, check_resources, CLASS_COLORS, INK, SURFACE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ic_file")
    ap.add_argument("index", type=int)
    ap.add_argument("--like", required=True, help="a trial dir run from this IC (for geometry)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--work-dir", default=None, help="where the one-frame grouping cache goes (default: out-dir)")
    ap.add_argument("--max-dist", type=float, default=65.0)
    ap.add_argument("--downscale", type=int, default=None, help="default: VIDEO_DOWNSCALE")
    ap.add_argument("--margin", type=int, default=80)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    work = args.work_dir or args.out_dir
    os.makedirs(work, exist_ok=True)
    check_resources(args.out_dir)

    ic = json.load(open(args.ic_file, encoding="utf-8"))[args.index]
    sm = ic["smarticles"]
    snap = read_snapshot(args.like)
    W, H = int(snap["W"]), int(snap["H"])
    xyt = np.array([[s["pos"][0], s["pos"][1], s["angle"]] for s in sm], float)
    arms = np.array([[s["thL"], s["thR"]] for s in sm], float)
    n_sides, r = int(snap["RING_N_SIDES"]), float(snap["INNER_R"])
    k = np.arange(n_sides)
    ring = np.c_[W / 2 + r * np.cos(2 * np.pi * k / n_sides), H / 2 + r * np.sin(2 * np.pi * k / n_sides)]

    # same grouping as the runs: a one-frame POS_ALL through group_table (Agent_ID is 1-based, IC order)
    pos = os.path.join(work, f"IC{args.index}_POS_ALL.csv")
    pd.DataFrame({"Step": 0, "Agent_ID": np.arange(1, len(sm) + 1),
                  "X": xyt[:, 0], "Y": xyt[:, 1], "Theta": xyt[:, 2]}).to_csv(pos, index=False)
    groups, _, _ = group_table(pos, args.max_dist, force=True)
    size_of = np.ones(len(sm), int)
    for sz, mem in groups[["size", "member_ids"]].itertuples(index=False):
        size_of[[int(m) - 1 for m in str(mem).split()]] = sz
    cls = np.where(size_of >= 6, "6+", np.where(size_of >= 3, "3-5", "1-2"))
    fr = dict(xyt=xyt, cls=cls, arms=arms, ring=ring)

    pts = np.vstack([xyt[:, :2], ring])
    x0, y0 = (np.floor(pts.min(0)) - args.margin).astype(int)
    x1, y1 = (np.ceil(pts.max(0)) + args.margin).astype(int)
    crop = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    ds = args.downscale or int(snap.get("VIDEO_DOWNSCALE", 1))

    pygame.init()
    img = draw(snap, fr, crop, ds, bodies_only=False)
    out = os.path.join(args.out_dir, f"IC{args.index}_initial.png")
    pygame.image.save(img, out)
    arr = pygame.surfarray.array3d(img).swapaxes(0, 1)
    pygame.quit()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({"font.size": 10, "text.color": INK})
    h, w = arr.shape[:2]
    fig, ax = plt.subplots(figsize=(5.2, 5.2 * h / w + 0.9), facecolor=SURFACE, layout="constrained")
    ax.imshow(arr)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color("#d0cfca")
    n6 = int((groups["size"] >= 6).sum())
    ax.set_title(f"initial condition #{args.index} (before warmup; shared by baseline and strategy)\n"
                 f"≥6: {(size_of >= 6).mean()*100:.0f}% of robots in {n6} group{'' if n6 == 1 else 's'} · "
                 f"≥3: {(size_of >= 3).mean()*100:.0f}%", fontsize=9.5)
    fig.legend(handles=[Patch(color=c, label=f"group of size {k}") for k, c in CLASS_COLORS.items()],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    out2 = os.path.join(args.out_dir, f"IC{args.index}_initial_annotated.png")
    fig.savefig(out2, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"IC #{args.index}: {len(sm)} robots, >=3 {(size_of >= 3).mean():.1%}, >=6 {(size_of >= 6).mean():.1%}")
    print(f"wrote {out}\n      {out2}")


if __name__ == "__main__":
    sys.exit(main())

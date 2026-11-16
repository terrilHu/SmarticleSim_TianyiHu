"""
render_trial.py  -  redraw a trial's video from recorded state, no physics.

Reads what the trial saved and draws every frame again:

    trial_XXXX_POS_ALL.csv    body pose (x, y, theta) per robot
    trial_XXXX_render.npz     arm joint angles + ring outline, per frame
    config_snapshot.json      arena and robot geometry

The npz is the part that makes this possible. POS_ALL carries only the body
pose, so the arms -- the thing that makes a smarticle look like one -- are not
in it, and the ring is movable: the swarm pushes it out of shape (measured: a
vertex moving 73 px and the outline spanning 144-305 px around a nominal 265),
so it cannot be reconstructed from the config either.

Because nothing is simulated, restyling costs a re-render instead of a re-run,
and the drawing is free to do things the live renderer cannot: shade robots by
the group they are in, cut to a time range, change every colour.

    python analysis/render_trial.py ".../a826/trial_0000"
    python analysis/render_trial.py ".../a826/trial_0000" --color-by group
    python analysis/render_trial.py ".../trial_0000" --from 100 --to 140 --downscale 1
"""

import argparse
import glob
import json
import math
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import pygame

from batch_group_plots import CAT_COLORS, read_snapshot, find_pos_all


def hex_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)


def box(surface, cx, cy, ang, length, width, color):
    """A rectangle of length x width centred at (cx, cy), rotated by ang."""
    # float() throughout: these coordinates arrive as numpy scalars from the
    # recorded arrays, and pygame rejects those with "points must be number
    # pairs".
    hl, hw = float(length) / 2.0, float(width) / 2.0
    cx, cy, ang = float(cx), float(cy), float(ang)
    pts = []
    for dx, dy in ((-hl, -hw), (hl, -hw), (hl, hw), (-hl, hw)):
        rx, ry = rot((dx, dy), ang)
        pts.append((cx + rx, cy + ry))
    pygame.draw.polygon(surface, color, pts)


def draw_frame(surf, geo, pose, arms, ring, style, group_of=None):
    """One frame: ring outline first, then every robot's body and two arms."""
    surf.fill(style["bg"])

    if ring is not None and style["ring"]:
        pts = [(float(x), float(y)) for x, y in ring]
        if len(pts) >= 3:
            pygame.draw.polygon(surf, style["ring_color"], pts,
                                max(1, int(style["ring_w"])))

    ml, mw = geo["main_len"], geo["main_w"]
    al, aw = geo["arm_len"], geo["arm_w"]
    for i in range(len(pose)):
        x, y, th = float(pose[i][0]), float(pose[i][1]), float(pose[i][2])
        body_col = style["body"]
        if group_of is not None:
            g = group_of[i]
            body_col = (style["singleton"] if g < 0
                        else CAT_RGB[g % len(CAT_RGB)])
        # hinges, in world coordinates
        hlx, hly = rot((-ml / 2.0, 0.0), th)
        hrx, hry = rot((ml / 2.0, 0.0), th)
        for sign, (hx, hy), col in ((-1, (x + hlx, y + hly), style["armL"]),
                                    (+1, (x + hrx, y + hry), style["armR"])):
            a = th + arms[i][0 if sign < 0 else 1]
            # the hinge sits at the arm's inner end, so the arm centre is half
            # an arm-length away from it along the arm's own axis
            ox, oy = rot((sign * al / 2.0, 0.0), a)
            box(surf, hx + ox, hy + oy, a, al, aw, col)
        box(surf, x, y, th, ml, mw, body_col)


CAT_RGB = [hex_rgb(c) for c in CAT_COLORS]


def main():
    ap = argparse.ArgumentParser(
        description="Redraw a trial's video from its recorded state.")
    ap.add_argument("trial", help="a trial folder containing *_render.npz")
    ap.add_argument("--out", default=None, help="output .mp4 (default: in the trial folder)")
    ap.add_argument("--from", dest="t0", type=float, default=None, help="start time (s)")
    ap.add_argument("--to", dest="t1", type=float, default=None, help="end time (s)")
    ap.add_argument("--downscale", type=int, default=None,
                    help="shrink by this factor (default: the trial's own VIDEO_DOWNSCALE)")
    ap.add_argument("--fps", type=float, default=None,
                    help="output frame rate (default: the recorded one)")
    ap.add_argument("--color-by", choices=["none", "group"], default="none",
                    help="'group' shades each robot by the group it is in")
    ap.add_argument("--max-dist", type=float, default=65.0,
                    help="grouping threshold for --color-by group")
    ap.add_argument("--bg", default="#ffffff")
    ap.add_argument("--body", default="#3498db")
    ap.add_argument("--arm-left", default="#ff6464")
    ap.add_argument("--arm-right", default="#6464ff")
    ap.add_argument("--ring-color", default="#95a5a6")
    ap.add_argument("--ring-width", type=int, default=6)
    ap.add_argument("--no-ring", dest="ring", action="store_false")
    args = ap.parse_args()

    folder = args.trial
    npz = glob.glob(os.path.join(folder, "*_render.npz"))
    pos = find_pos_all(folder)
    if not npz or not pos:
        print("Need both *_render.npz and *_POS_ALL.csv/.npz in " + folder)
        if not npz:
            print("  No render state: this trial predates SAVE_RENDER_STATE. "
                  "The ring outline and arm angles were never recorded, so the "
                  "video cannot be rebuilt -- re-run from config_snapshot.json "
                  "instead (deterministic, same result).")
        return 1

    z = np.load(npz[0])
    arms, ring = z["arms"], z["ring"]
    stride = int(z["stride"]) if "stride" in z else 1
    fps = args.fps or float(z["fps"]) if "fps" in z else 30.0

    snap = read_snapshot(folder)
    geo = {k.lower(): snap[k] for k in ("MAIN_LEN", "MAIN_W", "ARM_LEN", "ARM_W")}
    W, H = int(snap["W"]), int(snap["H"])
    ds = args.downscale if args.downscale is not None else int(snap.get("VIDEO_DOWNSCALE", 1))

    from pos_all_io import read_pos_all
    df = read_pos_all(pos, usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    steps = df["Step"].unique()
    n = len(df) // len(steps)
    xyt = df[["X", "Y", "Theta"]].to_numpy(np.float32).reshape(len(steps), n, 3)
    # render state was sampled every `stride`-th recorded frame
    xyt = xyt[::stride][:len(arms)]
    nf = min(len(xyt), len(arms), len(ring))
    t = np.arange(nf) / fps

    lo = 0 if args.t0 is None else int(np.searchsorted(t, args.t0))
    hi = nf if args.t1 is None else int(np.searchsorted(t, args.t1))
    lo, hi = max(0, lo), min(nf, max(lo + 1, hi))

    style = dict(bg=hex_rgb(args.bg), body=hex_rgb(args.body),
                 armL=hex_rgb(args.arm_left), armR=hex_rgb(args.arm_right),
                 ring_color=hex_rgb(args.ring_color), ring_w=args.ring_width,
                 ring=args.ring, singleton=(170, 170, 170))

    grouper = None
    if args.color_by == "group":
        from pos_all_grouping import compute_frame_groups
        grouper = compute_frame_groups

    pygame.init()
    surf = pygame.Surface((W, H))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from analysis import VideoRecorder
    out = args.out or os.path.join(
        folder, os.path.basename(os.path.normpath(folder)) + "_rerender.mp4")
    rec = VideoRecorder(out, W, H, int(round(fps)), downscale=ds)

    ids = np.arange(n)
    for k in range(lo, hi):
        gof = None
        if grouper is not None:
            gs = grouper(xyt[k][:, :2], xyt[k][:, 2], ids, args.max_dist, True)
            gof = np.full(n, -1, dtype=int)
            for gi, g in enumerate(gs):
                if g["size"] >= 2:
                    gof[list(g["local_idx"])] = gi
        draw_frame(surf, geo, xyt[k], arms[k], ring[k], style, gof)
        rec.add_frame_from_surface(surf)
        if (k - lo) % 200 == 0:
            print(f"  {k-lo+1}/{hi-lo} frames", flush=True)
    rec.close()
    print(f"-> {out}  ({hi-lo} frames, {fps:g} fps, {W//ds}x{H//ds})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
real_group_overlay.py
---------------------
Re-render the hardware overlay videos from the recorded ArUco detections, in
the simulation snapshot style: no text, one dot per robot centre, a line for
every within-group link, coloured by the size of the robot's group:

    size 1-2   grey        #c4c4c4
    size 3-5   light blue  #9ecae1
    size >= 6  dark blue   #2171b5

(the CLASS_COLORS of analysis/render_snapshots.py).

Usage:
    python analysis/real_group_overlay.py D:/UMich-Smarticles/videos/1105_sim2real_exp/11_12
    python analysis/real_group_overlay.py <dir> --stamps 1112_185734 --jobs 1
    python analysis/real_group_overlay.py <dir> --preview 3000   # one PNG, no video

Output: <dir>/group_overlay/<stamp>_group_overlay.mp4 (same size/fps as the source).

Where everything comes from
===========================
* Positions: record<stamp>.csv, one row per recognised frame (row r <-> video
  frame r * DETECT_EVERY_N, exactly as the hardware render_overlay_video maps
  them). A robot that was not detected is written as 0,0,0 and is not drawn.
* Groups and links are recomputed from those positions with the hardware rule
  (grouping.py): Delaunay edges no longer than GROUP_MAX_DIST pixels, groups
  are connected components. The links drawn are therefore exactly the edges
  that put robots in the same group, not a cosmetic spanning tree. The
  recomputed partition is checked against group<stamp>.csv and the agreement
  is printed -- it should be 100%.
* Frames between two recognised frames are filled the way the original overlay
  does it (OVERLAY_INTERP): positions are interpolated linearly, while
  membership and links stay those of the earlier recognised frame, so no split
  or merge is drawn that was not measured. Gaps longer than
  OVERLAY_INTERP_MAX_GAP frames hold the last recognised frame.

DETECT_EVERY_N, GROUP_MAX_DIST and OVERLAY_INTERP_MAX_GAP are read from
config<stamp>.txt. The source video's timestamp watermark (top left) is part of
the recorded pixels and stays.
"""

import argparse
import glob
import multiprocessing as mp
import os
import re
import time

import cv2
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay

# Size classes: (lowest size in class, BGR colour)
def _bgr(hexstr):
    h = hexstr.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))

CLASSES = [(1, _bgr("#c4c4c4")), (3, _bgr("#9ecae1")), (6, _bgr("#2171b5"))]
OUTLINE = _bgr("#3a3a3a")
SHIFT = 4                       # cv2 sub-pixel drawing: coords * 2**SHIFT
SCALE = 1 << SHIFT


def class_of(size):
    c = 0
    for k, (lo, _col) in enumerate(CLASSES):
        if size >= lo:
            c = k
    return c


# -----------------------------------------------------------------------------
# Inputs
# -----------------------------------------------------------------------------

def read_config(path):
    """The few run parameters this needs, from config<stamp>.txt."""
    cfg = {"DETECT_EVERY_N": 2, "GROUP_MAX_DIST": 100.0,
           "OVERLAY_INTERP_MAX_GAP": 6, "marker_to_index": None}
    if not os.path.isfile(path):
        return cfg
    txt = open(path, encoding="utf-8").read()
    for key, typ in (("DETECT_EVERY_N", int), ("GROUP_MAX_DIST", float),
                     ("OVERLAY_INTERP_MAX_GAP", int)):
        m = re.search(rf"^\s*{key}\s+([-\d.]+)\s*$", txt, re.M)
        if m:
            cfg[key] = typ(float(m.group(1)))
    m = re.search(r"marker id -> .*?\]\s*\n\s*(.+)", txt)
    if m:
        cfg["marker_to_index"] = {int(a): int(b) for a, b in
                                  re.findall(r"(\d+)->(\d+)", m.group(1))}
    return cfg


def load_record(path):
    """-> (row index array, xy (rows, robots, 2) with NaN where undetected)."""
    df = pd.read_csv(path)
    idx = df.iloc[:, 0].to_numpy(int)
    xcols = [c for c in df.columns if re.fullmatch(r"\d+_\d+x", str(c))]
    ycols = [c.replace("x", "y") for c in xcols]
    xy = np.stack([df[xcols].to_numpy(float), df[ycols].to_numpy(float)], axis=2)
    missing = ~np.isfinite(xy).all(axis=2) | ((xy[..., 0] == 0) & (xy[..., 1] == 0))
    xy[missing] = np.nan
    return idx, xy


def group_frame(xy, max_dist):
    """
    Hardware grouping on one frame.
    -> (robot indices present, their xy, edges as index pairs into that list,
        group size per present robot)
    """
    present = np.where(np.isfinite(xy).all(axis=1))[0]
    pts = xy[present]
    n = len(pts)
    edges = np.zeros((0, 2), int)
    if n >= 2:
        try:
            s = Delaunay(pts).simplices
            e = np.sort(np.concatenate((s[:, [0, 1]], s[:, [1, 2]], s[:, [0, 2]])), axis=1)
            e = np.unique(e, axis=0)
            d = np.hypot(*(pts[e[:, 0]] - pts[e[:, 1]]).T)
            edges = e[d <= max_dist]
        except Exception:          # too few / collinear points: no edges, like grouping.py
            pass
    parent = np.arange(n)

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = np.array([find(a) for a in range(n)], int)
    _, inv, counts = np.unique(roots, return_inverse=True, return_counts=True)
    return present, pts, edges, counts[inv], inv


def check_against_groups(group_csv, keys, marker_to_index):
    """Fraction of recognised frames whose recomputed partition equals the csv's."""
    if not os.path.isfile(group_csv):
        return None
    g = pd.read_csv(group_csv, usecols=["Step", "member_ids"], dtype={"member_ids": str})
    m2i = marker_to_index or {}
    rec = {}
    for step, mem in zip(g["Step"].to_numpy(), g["member_ids"].to_numpy()):
        rec.setdefault(int(step), set()).add(
            frozenset(m2i.get(int(x), int(x)) for x in mem.split()))
    same = total = 0
    for step, (present, _pts, _e, _sz, lab) in keys.items():
        mine = {}
        for r, l in zip(present, lab):
            mine.setdefault(int(l), set()).add(int(r))
        mine = {frozenset(v) for v in mine.values()}
        theirs = rec.get(step, set())
        if not mine and not theirs:
            continue
        total += 1
        same += mine == theirs
    return same, total


# -----------------------------------------------------------------------------
# Drawing
# -----------------------------------------------------------------------------

def draw(frame, pts, edges, sizes, dot_r, line_w, outline_w):
    """Lines under dots; within each layer small classes first so big groups sit on top."""
    cls = np.array([class_of(int(s)) for s in sizes], int)
    P = np.round(pts * SCALE).astype(np.int32)
    for c in range(len(CLASSES)):
        col = CLASSES[c][1]
        for a, b in edges:
            if cls[a] == c:
                cv2.line(frame, tuple(P[a]), tuple(P[b]), col, line_w,
                         cv2.LINE_AA, SHIFT)
    r_out = int(round((dot_r + outline_w) * SCALE))
    r_in = int(round(dot_r * SCALE))
    for c in range(len(CLASSES)):
        col = CLASSES[c][1]
        for i in np.where(cls == c)[0]:
            if outline_w > 0:
                cv2.circle(frame, tuple(P[i]), r_out, OUTLINE, -1, cv2.LINE_AA, SHIFT)
            cv2.circle(frame, tuple(P[i]), r_in, col, -1, cv2.LINE_AA, SHIFT)
    return frame


def keyframe_table(record_csv, cfg):
    """-> (sorted video frame numbers, {step: grouped frame}, {video frame: step})"""
    idx, xy = load_record(record_csv)
    every = cfg["DETECT_EVERY_N"]
    keys, vidx, step_of = {}, [], {}
    for r, row in zip(idx, xy):
        if not np.isfinite(row).all(axis=1).any():
            continue                       # nothing recognised: a gap, like the original
        keys[int(r)] = group_frame(row, cfg["GROUP_MAX_DIST"])
        v = int(r) * every
        vidx.append(v)
        step_of[v] = int(r)
    return vidx, keys, step_of


def frame_geometry(k, idx, vidx, keys, step_of, max_gap):
    """Positions/edges/sizes to draw at video frame idx, given current keyframe k."""
    present, pts, edges, sizes, _lab = keys[step_of[vidx[k]]]
    gap = vidx[k + 1] - vidx[k] if k + 1 < len(vidx) else 0
    if gap > 0 and idx > vidx[k] and (max_gap <= 0 or gap <= max_gap):
        t = (idx - vidx[k]) / gap
        p1, x1 = keys[step_of[vidx[k + 1]]][:2]
        where1 = {int(r): j for j, r in enumerate(p1)}
        pts = pts.copy()
        for i, r in enumerate(present):
            j = where1.get(int(r))
            if j is not None:
                pts[i] = (1 - t) * pts[i] + t * x1[j]
    return pts, edges, sizes


def render(job):
    (stamp, exp_dir, out_dir, opts) = job
    src = os.path.join(exp_dir, f"{stamp}.mp4")
    cfg = read_config(os.path.join(exp_dir, f"config{stamp}.txt"))
    vidx, keys, step_of = keyframe_table(os.path.join(exp_dir, f"record{stamp}.csv"), cfg)
    chk = check_against_groups(os.path.join(exp_dir, f"group{stamp}.csv"), keys,
                               cfg["marker_to_index"])
    chk_s = (f"partition matches group csv in {chk[0]}/{chk[1]} frames"
             if chk else "no group csv to check against")
    if not vidx:
        return f"{stamp}: no recognised frames, skipped"

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        return f"{stamp}: cannot open {src}"
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 12.0
    s = w / 1920.0
    dot_r = opts["dot_r"] * s
    line_w = max(1, int(round(opts["line_w"] * s)))
    outline_w = opts["outline_w"] * s
    max_gap = cfg["OVERLAY_INTERP_MAX_GAP"]

    if opts["preview"] is not None:
        f_no = int(opts["preview"])
        cap.set(cv2.CAP_PROP_POS_FRAMES, f_no)
        ok, frame = cap.read()
        cap.release()
        k = max(0, int(np.searchsorted(vidx, f_no, side="right")) - 1)
        draw(frame, *frame_geometry(k, f_no, vidx, keys, step_of, max_gap),
             dot_r, line_w, outline_w)
        out = os.path.join(out_dir, f"{stamp}_preview_{f_no}.png")
        cv2.imwrite(out, frame)
        return f"{stamp}: {chk_s}; preview -> {out}"

    out = os.path.join(out_dir, f"{stamp}_group_overlay.mp4")
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not wr.isOpened():
        cap.release()
        return f"{stamp}: cannot create {out}"
    t0 = time.perf_counter()
    idx = k = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            while k + 1 < len(vidx) and vidx[k + 1] <= idx:
                k += 1
            draw(frame, *frame_geometry(k, idx, vidx, keys, step_of, max_gap),
                 dot_r, line_w, outline_w)
            wr.write(frame)
            idx += 1
    finally:
        wr.release()
        cap.release()
    return (f"{stamp}: {chk_s}; {idx} frames in {time.perf_counter() - t0:.0f}s"
            f" -> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir")
    ap.add_argument("--stamps", nargs="+", default=None,
                    help="only these trials (default: every record<stamp>.csv with a video)")
    ap.add_argument("--out", default=None, help="default <exp_dir>/group_overlay")
    ap.add_argument("--jobs", type=int, default=3, help="videos rendered at once")
    ap.add_argument("--dot-r", type=float, default=9.0, help="dot radius, px at 1920 wide")
    ap.add_argument("--line-w", type=float, default=4.0, help="link width, px at 1920 wide")
    ap.add_argument("--outline-w", type=float, default=1.5,
                    help="dark rim around dots so grey stays visible on white markers; 0 = none")
    ap.add_argument("--preview", type=int, default=None, metavar="FRAME",
                    help="write one annotated PNG of this video frame instead of the video")
    args = ap.parse_args()

    stamps = args.stamps or sorted(
        os.path.basename(p)[len("record"):-len(".csv")]
        for p in glob.glob(os.path.join(args.exp_dir, "record*.csv"))
        if os.path.isfile(os.path.join(args.exp_dir,
                                       os.path.basename(p)[len("record"):-len(".csv")] + ".mp4")))
    out_dir = args.out or os.path.join(args.exp_dir, "group_overlay")
    os.makedirs(out_dir, exist_ok=True)
    opts = {"dot_r": args.dot_r, "line_w": args.line_w, "outline_w": args.outline_w,
            "preview": args.preview}
    jobs = [(s, args.exp_dir, out_dir, opts) for s in stamps]
    print(f"{len(jobs)} trials: {' '.join(stamps)}")
    n = min(max(1, args.jobs), len(jobs))
    if n > 1:
        with mp.Pool(n) as pool:
            for msg in pool.imap_unordered(render, jobs):
                print(" ", msg, flush=True)
    else:
        for j in jobs:
            print(" ", render(j), flush=True)


if __name__ == "__main__":
    main()

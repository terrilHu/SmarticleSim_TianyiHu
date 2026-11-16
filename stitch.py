"""
stitch.py  ─  Join a chain of branches into one ordinary trial folder.

    python stitch.py <branch folder> [--out DIR] [--csv] [--render [render_trial args]]

Walks from the given branch up to its root trial and keeps, from each folder,
only the part before the next branch point:

    root     Steps 0 .. s1
    b1       Steps s1+1 .. s2          (b1 = root/branches/bNNN_..., and so on)
    leaf     Steps sk+1 .. end

and writes them as one trial on the root's time axes:

    trial_XXXX_POS_ALL.npz   (or .csv with --csv)
    trial_XXXX_render.npz    arm angles + ring outline, for render_trial.py
    trial_XXXX_gait_log.csv  each segment's rows from its own start onwards
    trial_XXXX_coverage.csv
    trial_XXXX_alignment.csv
    config_snapshot.json     the leaf's config (the root's arena and robots)
    segments.json            each segment's folder, Step / time range and
                             intervention -- the record of who changed what when

The result reads like any trial: batch_group_plots, group_persistence and
render_trial take the folder directly. Legacy pair-metric files (_impacts,
_GRs*) are not stitched.

Default output folder: <leaf>/stitched/.
"""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import pos_all_io
from branch import read_branch_json, parent_of


def chain(leaf):
    """[(folder, branch_json or None)] from root to leaf."""
    out, f = [], os.path.abspath(leaf)
    while True:
        bj = read_branch_json(f)
        out.append((f, bj))
        if bj is None:
            break
        f = parent_of(f)
    return out[::-1]


def one(folder, suffix):
    hits = [os.path.join(folder, n) for n in sorted(os.listdir(folder))
            if n.startswith("trial_") and n.endswith(suffix)]
    return hits[0] if hits else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("leaf", help="branch folder (the last segment)")
    ap.add_argument("--out", default=None, help="output folder (default <leaf>/stitched)")
    ap.add_argument("--csv", action="store_true", help="write POS_ALL as CSV instead of npz")
    ap.add_argument("--render", nargs=argparse.REMAINDER, default=None,
                    help="then run analysis/render_trial.py on the result; "
                         "anything after --render is passed through to it")
    args = ap.parse_args(argv)

    segs = chain(args.leaf)
    if len(segs) < 2:
        sys.exit(f"{args.leaf} is not a branch folder (no branch.json)")
    out = os.path.abspath(args.out or os.path.join(args.leaf, "stitched"))
    os.makedirs(out, exist_ok=True)

    root_dir = segs[0][0]
    leaf_bj  = segs[-1][1]
    tid      = int(leaf_bj["trial_id"])
    prefix   = os.path.join(out, f"trial_{tid:04d}")

    # Segment bounds on the Step axis: folder i keeps [first_i, next_branch_step]
    bounds = []
    for i, (f, bj) in enumerate(segs):
        lo = 0 if bj is None else int(bj["first_step"])
        hi = int(segs[i + 1][1]["branch_step"]) if i + 1 < len(segs) else None
        bounds.append((lo, hi))

    # ── POS_ALL ──────────────────────────────────────────────────────────────
    parts = []
    for (f, bj), (lo, hi) in zip(segs, bounds):
        path = pos_all_io.find_pos_all(f)
        if path is None:
            sys.exit(f"no POS_ALL in {f}")
        print(f"[stitch] reading {path}")
        pos, s0 = pos_all_io.load_array(path)
        a = lo - s0
        b = (hi - s0 + 1) if hi is not None else len(pos)
        if a != 0 and bj is not None:
            sys.exit(f"{path} starts at Step {s0}, expected {lo}")
        if b > len(pos) or b <= a:
            sys.exit(f"{path} has Steps {s0}..{s0 + len(pos) - 1}, need {lo}..{hi}")
        parts.append(pos[a:b])
    pos = np.concatenate(parts)
    n_frames, n = pos.shape[:2]
    if args.csv:
        with open(prefix + "_POS_ALL.csv", "w", newline="") as fh:
            fh.write("Step,Agent_ID,X,Y,Theta\r\n")
            for s in range(0, n_frames, 2000):
                blk = pos[s:s + 2000]
                fh.write("".join(
                    f"{s + i},{j + 1},{r[0]:.4f},{r[1]:.4f},{r[2]:.4f}\r\n"
                    for i, fr in enumerate(blk) for j, r in enumerate(fr)))
        pos_path = prefix + "_POS_ALL.csv"
    else:
        pos_path = prefix + "_POS_ALL.npz"
        pos_all_io.write_npz(pos_path, pos, step0=0)
    print(f"[stitch] POS_ALL: {n_frames} frames x {n} robots -> {pos_path}")

    # ── Render state ─────────────────────────────────────────────────────────
    # Render frame r is Step r*stride. Each file starts at render_first
    # (0 for the root).
    arms, ring, fps, stride = [], [], None, None
    for (f, bj), (lo, hi) in zip(segs, bounds):
        p = one(f, "_render.npz")
        if p is None:
            arms = None
            print(f"[stitch] no render state in {f}; skipping render.npz")
            break
        with np.load(p) as z:
            fps    = float(z["fps"]) if fps is None else fps
            stride = int(z["stride"]) if stride is None else stride
            r0 = 0 if bj is None else int(bj["render_first"])
            r_lo = math.ceil(lo / stride)
            r_hi = (hi // stride) if hi is not None else r0 + len(z["arms"]) - 1
            arms.append(z["arms"][r_lo - r0:r_hi - r0 + 1])
            ring.append(z["ring"][r_lo - r0:r_hi - r0 + 1])
    if arms:
        np.savez_compressed(prefix + "_render.npz",
                            arms=np.concatenate(arms), ring=np.concatenate(ring),
                            fps=np.float32(fps), stride=np.int32(stride))

    # ── Time-keyed tables ────────────────────────────────────────────────────
    # gait_log time is on the recorded axis (t_rel); coverage time is raw sim
    # time (t_sim). A segment owns (its branch time, next branch time].
    def cut(col_time, key):
        rows = []
        for i, (f, bj) in enumerate(segs):
            p = one(f, key)
            if p is None:
                return None
            df = pd.read_csv(p)
            lo_t = None if bj is None else bj[col_time]
            hi_t = segs[i + 1][1][col_time] if i + 1 < len(segs) else None
            m = np.ones(len(df), bool)
            if lo_t is not None:
                # A branch's own log repeats everything it inherited; from
                # its branch time on, its rows are what happened. Rows at
                # exactly t* (decisions taken in frame s, and any --cmd)
                # are kept from both sides and de-duplicated below.
                m &= df["time"].to_numpy() >= lo_t - 1e-6 if key.endswith("gait_log.csv") \
                    else df["time"].to_numpy() > lo_t + 1e-9
            if hi_t is not None:
                m &= df["time"].to_numpy() <= hi_t + (1e-6 if key.endswith("gait_log.csv") else 1e-9)
            rows.append(df[m])
        return pd.concat(rows, ignore_index=True)

    g = cut("t_rel", "_gait_log.csv")
    if g is not None:
        g = g.sort_values("time", kind="stable")
        keep, last = [], {}
        for idx, rid, cmd in zip(g.index, g["robot_id"], g["command"]):
            if last.get(rid) != cmd:
                last[rid] = cmd
                keep.append(idx)
        g = g.loc[keep]
        with open(prefix + "_gait_log.csv", "w", newline="", encoding="utf-8") as fh:
            fh.write("time,robot_id,command,source\r\n")
            fh.write("".join(f"{t:.6f},{r},{c},{s}\r\n" for t, r, c, s in
                             zip(g["time"], g["robot_id"], g["command"], g["source"])))

    c = cut("t_sim", "_coverage.csv")
    if c is not None:
        with open(prefix + "_coverage.csv", "w", newline="", encoding="utf-8") as fh:
            fh.write("time,k,A_union,A0_sum,overlap,k_naive\r\n")
            fh.write("".join(f"{r[0]:.6f},{r[1]:.6f},{r[2]:.2f},{r[3]:.2f},{r[4]:.6f},{r[5]:.6f}\r\n"
                             for r in c.itertuples(index=False)))

    # Alignment is per Step: concatenate by Step range.
    al = []
    for (f, bj), (lo, hi) in zip(segs, bounds):
        p = one(f, "_alignment.csv")
        if p is None:
            al = None
            break
        df = pd.read_csv(p)
        m = df["Step"] >= lo
        if hi is not None:
            m &= df["Step"] <= hi
        al.append(df[m])
    if al:
        pd.concat(al, ignore_index=True).to_csv(prefix + "_alignment.csv", index=False)

    # ── Config and segment record ────────────────────────────────────────────
    shutil.copy2(os.path.join(segs[-1][0], "config_snapshot.json"),
                 os.path.join(out, "config_snapshot.json"))
    t_rec0 = leaf_bj["t_rec0"]
    seg_rec = []
    for (f, bj), (lo, hi) in zip(segs, bounds):
        hi_step = hi if hi is not None else n_frames - 1
        seg_rec.append({
            "folder": f, "steps": [lo, hi_step],
            "t_rel": [lo / leaf_bj["fps"] if bj is None else bj["t_rel"],
                      hi_step / leaf_bj["fps"]],
            "intervention": None if bj is None else bj["intervention"],
        })
    with open(os.path.join(out, "segments.json"), "w", encoding="utf-8") as fh:
        json.dump({"trial_id": tid, "root": root_dir, "t_rec0": t_rec0,
                   "segments": seg_rec}, fh, indent=2, default=str)
    print(f"[stitch] {len(segs)} segments -> {out}")
    for s in seg_rec:
        iv = s["intervention"]
        what = "recorded" if iv is None else ", ".join(
            f"{k}={'<spec>' if k == 'spec' else v}" for k, v in iv.items()
            if k in ("spec", "cmd", "controller") and v)
        fr = (iv or {}).get("cmd_a_fraction")
        if fr:
            what = ", ".join(x for x in (
                f"CMD_A_FRACTION={fr['fraction']} ({fr['cmd_a']:+d}: "
                f"{fr['a_before']}->{fr['a_after']})", what) if x)
        print(f"         Steps {s['steps'][0]:>6}..{s['steps'][1]:<6} "
              f"t {s['t_rel'][0]:8.3f}..{s['t_rel'][1]:8.3f}  {what}")

    if args.render is not None:
        cmd = [sys.executable, os.path.join(HERE, "analysis", "render_trial.py"), out] + args.render
        print("[stitch] " + " ".join(cmd))
        subprocess.call(cmd)
    return out


if __name__ == "__main__":
    main()

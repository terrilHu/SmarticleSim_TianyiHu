"""
sweep_example_snapshots.py
--------------------------
First and last frame of chosen gait-sweep cells, one trial per cell, drawn the way
render_snapshots.py draws them (body coloured by group size class, grey arms, ring).
Meant for the example cells marked on the amplitude x frequency heat maps
(plot_sweep_heatmaps.py), so each title gives the gait and the largest group.

Each cell may use its own trial (the one nearest that cell's mean); one crop is shared
by every image so the scale matches.

Output (in --out-dir):
    <exp>_<trial>_t<sec>s.png              one per cell and frame; the drawing only
    sweep_examples.png / _bare.png         cells as columns, start (top) and end (bottom)
    sweep_examples.csv                     per frame: largest group, share >= 6 / >= 3

Usage:
    python sweep_example_snapshots.py --out-dir DIR LABEL=EXP_DIR:TRIAL [LABEL=EXP_DIR:TRIAL ...]
e.g. "Example (1)=E:\\1107_baseline\\sync\\a816:trial_0001"
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import catalyst_radial_figs as crf
from sweep_stats import decode
from render_snapshots import CLASS_COLORS, SURFACE, check_resources, draw, load_run

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pygame


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cells", nargs="+", metavar="LABEL=EXP_DIR:TRIAL")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--margin", type=int, default=80)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    pygame.init()
    runs = []
    for x in a.cells:
        label, rest = x.split("=", 1)
        d, trial = rest.rsplit(":", 1)
        name = os.path.basename(os.path.normpath(d))
        cfg = json.load(open(os.path.join(d, "config_snapshot.json"), encoding="utf-8"))
        fps = cfg.get("RENDER_FPS_HEADLESS", 60)
        T = int(pd.read_csv(os.path.join(d, f"{name}_trial_stats.csv")).set_index("trial").loc[trial, "frames"])
        with np.load(os.path.join(d, trial, f"{trial}_render.npz")) as r:
            stride = int(r["stride"])
        steps = [0, (T - 1) // stride * stride]     # first frame, last step that also has render state
        check_resources(a.out_dir)
        snap, frames = load_run(os.path.join(d, trial), steps, a.max_dist)
        _, _, adeg, fhz = decode(name)
        gait = f"{adeg:g}°, {fhz:g} Hz"
        print(f"{label}: {name}/{trial} steps {steps}", flush=True)
        runs.append(dict(label=label, name=name, trial=trial, snap=snap, frames=frames, secs=[s / fps for s in steps], gait=gait))

    pts = np.vstack([np.vstack([f["xyt"][:, :2], f["ring"]]) for r in runs for f in r["frames"]])
    W, H = int(runs[0]["snap"]["W"]), int(runs[0]["snap"]["H"])
    x0, y0 = (np.floor(pts.min(0)) - a.margin).astype(int)
    x1, y1 = (np.ceil(pts.max(0)) + a.margin).astype(int)
    crop = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    ds = int(runs[0]["snap"].get("VIDEO_DOWNSCALE", 1))

    fig, axes = plt.subplots(2, len(runs), figsize=(4.2 * len(runs), 10.4), facecolor=SURFACE,
                             layout="constrained", squeeze=False)
    rows = []
    for j, r in enumerate(runs):
        for i, (f, sec) in enumerate(zip(r["frames"], r["secs"])):
            img = draw(r["snap"], f, crop, ds, bodies_only=False)
            pygame.image.save(img, os.path.join(a.out_dir, f"{r['name']}_{r['trial']}_t{sec:03.0f}s.png"))
            ax = axes[i, j]
            ax.imshow(pygame.surfarray.array3d(img).swapaxes(0, 1))
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)
                s.set_color("#d0cfca")
            ax.set_title(f"{r['label']}: {r['name']} ({r['gait']}), {r['trial']}, t = {sec:.0f} s\n"
                         f"largest group {f['largest']} · ≥6: {f['share6'] * 100:.0f}% · ≥3: {f['share3'] * 100:.0f}%",
                         fontsize=9.5, fontweight="normal")
            rows.append(dict(label=r["label"], exp=r["name"], trial=r["trial"], t_s=sec, largest=f["largest"],
                             share_ge6=f["share6"], n_ge6=f["n6"], share_ge3=f["share3"]))
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "sweep_examples.csv"), index=False)
    fig.legend(handles=[Patch(color=c, label=f"robot in a group of size {k}") for k, c in CLASS_COLORS.items()],
               loc="outside lower center", ncol=3, frameon=False, fontsize=9.5)
    fig.suptitle("gait-sweep examples: start (top) and end (bottom)", fontsize=11, fontweight="normal")
    fig.savefig(os.path.join(a.out_dir, "sweep_examples.png"), dpi=170, facecolor=SURFACE)
    crf.strip_text(fig)
    for ax in axes.ravel():
        for s in ax.spines.values():
            s.set_visible(False)
    fig.savefig(os.path.join(a.out_dir, "sweep_examples_bare.png"), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    pygame.quit()
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())

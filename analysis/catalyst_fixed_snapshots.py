"""
catalyst_fixed_snapshots.py
---------------------------
Start and end snapshots of the most representative trial of fixed-catalyst runs
(catalyst_fixed_dose_curves.py), drawn the way render_snapshots.py draws them:
body coloured by group size class (1-2 / 3-5 / 6+), grey arms, no text.
--mark-cmd C outlines the robots running command C (the catalysts, from each
trial's gait log) in orange; the images then go to snapshots_<trial>_mark<C>/.

Trial k uses the same initial condition in every condition (INIT_INDICES), so one
trial index is chosen for all conditions: the start frames then share a layout and
only the outcome differs.

--trial auto scores each trial index on the two quantities of the level chart,
share of robots in groups >= 6 and >= 3, per condition:
    window_z  the trial's last-<last>-s mean minus the mean of the trial means,
              in SDs of the trial means (the per-trial values behind the chart's points)
    end_z     the trial's final frame minus the pooled per-frame mean of the window,
              in pooled per-frame SDs
score = mean |window_z| + mean |end_z| over conditions and quantities; lowest wins
(the same rule as catalyst_end_snapshots.py).

Output (in --out-dir):
    trial_selection.csv                    every z, every trial
    snapshots_<trial>/<exp>_<trial>_t<sec>s.png   start and end frame per condition
    snapshots_<trial>/snapshots_<trial>.csv       per-frame share >= 6 / >= 3 of those frames
    snapshots_<trial>/overview_<trial>.png        all frames on one sheet, for checking
    (snapshots_<trial>_mark<C>/ with --mark-cmd; its csv adds catalyst counts)

Usage:
    python catalyst_fixed_snapshots.py <datafile root> --out-dir DIR
           --exps a816_12-852 a816_24-852 ... [--trial auto] [--last 100]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

from render_snapshots import CLASS_COLORS, MARK_COLOR, SURFACE, check_resources, draw, load_run

import pygame


def per_frame(groups_csv, n, fps, last):
    """Per-step share of robots in groups >= 6 and >= 3 over the last `last` s (last entry = final frame)."""
    df = pd.read_csv(groups_csv, usecols=["Step", "size"])
    T = int(df.Step.max()) + 1
    a = int(np.ceil(T - last * fps))
    df = df[df.Step >= a]
    steps = np.arange(a, T)
    out = {}
    for s in (6, 3):
        w = df[df["size"] >= s].groupby("Step")["size"].sum().reindex(steps, fill_value=0)
        out[f"ge{s}"] = w.to_numpy() / n
    return out, T


def command_mask(trial_dir, step, fps, n, cmd):
    """(n,) bool: robot (0-based, = sorted Agent_ID - 1) runs `cmd` (one code or several) at `step` (last logged command)."""
    g = glob.glob(os.path.join(trial_dir, "*_gait_log.csv"))[0]
    g = pd.read_csv(g).sort_values("time", kind="mergesort")
    g = g[g.time <= step / fps + 1e-9].groupby("robot_id").command.last()
    m = np.zeros(n, bool)
    m[g.index[np.isin(g.to_numpy(), np.atleast_1d(cmd))]] = True
    return m


def pick_trial(root, exps, trials, last, out_dir):
    rows = []
    for e in exps:
        cfg = json.load(open(os.path.join(root, e, "config_snapshot.json"), encoding="utf-8"))
        n, fps = cfg["N_SMARTICLES"], cfg.get("RENDER_FPS_HEADLESS", 60)
        fr = {}
        for tn in trials:
            fr[tn] = per_frame(glob.glob(os.path.join(root, e, tn, f"{tn}_groups_d65.csv"))[0], n, fps, last)[0]
            print(f"  {e}/{tn}", flush=True)
        for q in ("ge6", "ge3"):
            means = np.array([fr[tn][q].mean() for tn in trials])
            pool = np.concatenate([fr[tn][q] for tn in trials])
            for i, tn in enumerate(trials):
                rows.append(dict(exp=e, trial=tn, quantity=q, window_mean=means[i], final_frame=fr[tn][q][-1],
                                 cond_mean=means.mean(),
                                 window_z=(means[i] - means.mean()) / means.std(ddof=1),
                                 end_z=(fr[tn][q][-1] - pool.mean()) / pool.std(ddof=1)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(out_dir, "trial_selection.csv"), index=False)
    g = D.groupby("trial")
    S = pd.DataFrame({"window": g.window_z.apply(lambda v: v.abs().mean()),
                      "end": g.end_z.apply(lambda v: v.abs().mean()),
                      "worst_window": g.window_z.apply(lambda v: v.abs().max()),
                      "worst_end": g.end_z.apply(lambda v: v.abs().max())})
    S["score"] = S.window + S.end
    print(S.sort_values("score").round(3).to_string())
    best = S.score.idxmin()
    print(D[D.trial == best].round(3).to_string(index=False))
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--exps", nargs="+", required=True,
                    help="experiment dirs under root, or absolute paths (e.g. a baseline elsewhere)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--trial", default="auto")
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--last", type=float, default=100.0)
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--margin", type=int, default=80)
    ap.add_argument("--mark-cmd", type=int, default=None, help="outline robots running this command")
    ap.add_argument("--suptitle", default="fixed catalysts (-852)", help="overview title, before ', <trial>: ...'")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    trials = [f"trial_{i:04d}" for i in range(a.n_trials)]
    trial = pick_trial(a.root, a.exps, trials, a.last, a.out_dir) if a.trial == "auto" else a.trial
    print("trial:", trial)
    out = os.path.join(a.out_dir, f"snapshots_{trial}" + ("" if a.mark_cmd is None else f"_mark{a.mark_cmd}"))
    os.makedirs(out, exist_ok=True)

    pygame.init()
    runs = []
    for e in a.exps:
        ed = os.path.join(a.root, e)                   # an absolute --exps entry ignores root
        e = os.path.basename(os.path.normpath(ed))
        d = os.path.join(ed, trial)
        cfg = json.load(open(os.path.join(ed, "config_snapshot.json"), encoding="utf-8"))
        fps = cfg.get("RENDER_FPS_HEADLESS", 60)
        T = int(pd.read_csv(os.path.join(ed, f"{e}_trial_stats.csv")).set_index("trial").loc[trial, "frames"])
        npz = os.path.join(d, f"{trial}_render.npz")
        if os.path.exists(npz):
            with np.load(npz) as r:
                stride = int(r["stride"])
        else:
            stride = 2      # as render_snapshots.py: an older run with no render state is drawn bodies only
        steps = [0, (T - 1) // stride * stride]       # first frame, last step that also has render state
        check_resources(out)
        snap, frames = load_run(d, steps, a.max_dist)
        for f, s in zip(frames, steps):
            n = len(f["xyt"])
            f["mark"] = None if a.mark_cmd is None else command_mask(d, s, fps, n, a.mark_cmd)
        print(f"{e}: steps {steps}" + ("" if os.path.exists(npz) else "  (no render.npz: bodies only, no ring)"), flush=True)
        runs.append(dict(name=e, snap=snap, frames=frames, secs=[s / fps for s in steps]))

    # one crop for every image: robots and ring of every frame, plus a margin
    pts = np.vstack([np.vstack([f["xyt"][:, :2]] + ([f["ring"]] if f["ring"] is not None else []))
                     for r in runs for f in r["frames"]])
    W, H = int(runs[0]["snap"]["W"]), int(runs[0]["snap"]["H"])
    x0, y0 = (np.floor(pts.min(0)) - a.margin).astype(int)
    x1, y1 = (np.ceil(pts.max(0)) + a.margin).astype(int)
    crop = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    ds = int(runs[0]["snap"].get("VIDEO_DOWNSCALE", 1))

    summary, imgs = [], {}
    for r in runs:
        for f, sec in zip(r["frames"], r["secs"]):
            img = draw(r["snap"], f, crop, ds, bodies_only=False, mark=f["mark"])
            path = os.path.join(out, f"{r['name']}_{trial}_t{sec:03.0f}s.png")
            pygame.image.save(img, path)
            imgs[(r["name"], sec)] = pygame.surfarray.array3d(img).swapaxes(0, 1)
            summary.append(dict(exp=r["name"], trial=trial, t_s=round(sec, 2), share_ge6=f["share6"],
                                n_ge6_groups=f["n6"], share_ge3=f["share3"], file=os.path.basename(path)))
            if f["mark"] is not None:
                m, c = f["mark"], f["cls"]
                summary[-1].update(n_catalysts=int(m.sum()), cat_in_ge6=int((m & (c == "6+")).sum()),
                                   cat_in_3to5=int((m & (c == "3-5")).sum()),
                                   cat_in_1to2=int((m & (c == "1-2")).sum()))
    S = pd.DataFrame(summary)
    S.to_csv(os.path.join(out, f"snapshots_{trial}.csv"), index=False)
    print(S.round(3).to_string(index=False))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    ncol = len(runs)
    fig, axes = plt.subplots(2, ncol, figsize=(3.2 * ncol, 7.2), facecolor=SURFACE, layout="constrained")
    for j, r in enumerate(runs):
        for i, (f, sec) in enumerate(zip(r["frames"], r["secs"])):
            ax = axes[i, j]
            ax.imshow(imgs[(r["name"], sec)])
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#d0cfca")
            cat = "" if f["mark"] is None or not f["mark"].any() else \
                f"\ncatalysts in ≥6: {(f['mark'] & (f['cls'] == '6+')).sum()} / {f['mark'].sum()}"
            ax.set_title(f"{r['name']}, t = {sec:.0f} s\n≥6: {f['share6'] * 100:.0f}% · ≥3: {f['share3'] * 100:.0f}%" + cat,
                         fontsize=9)
    handles = [Patch(color=c, label=f"robot in a group of size {k}") for k, c in CLASS_COLORS.items()]
    if a.mark_cmd is not None:
        handles.append(Patch(facecolor="none", edgecolor=MARK_COLOR, lw=2, label=f"catalyst ({a.mark_cmd})"))
    fig.legend(handles=handles,
               loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9)
    fig.suptitle(f"{a.suptitle}, {trial}: start (top) and end (bottom)", fontsize=11)
    fig.savefig(os.path.join(out, f"overview_{trial}.png"), dpi=130, facecolor=SURFACE)
    plt.close(fig)
    pygame.quit()


if __name__ == "__main__":
    sys.exit(main())

"""
catalyst_end_snapshots.py
-------------------------
The last recorded frame of one trial, under every condition, drawn the way
render_snapshots.py draws it (body coloured by group size class, grey arms).

--trial auto picks the trial closest to typical across all conditions, using the
smoothed groups in catalyst_radial_build.py's cache:

    share_ge6    share of robots in groups >= 6
    outer_share  share of those robots whose group centroid is outside --edge
    largest      largest group

For each condition and quantity, with the condition's five trials pooled:
    window_z  distance of the trial's last-100-s mean from the mean of the trial means,
              in SDs of the trial means
    end_z     distance of the trial's final frame from the pooled mean of the last 100 s,
              in pooled per-frame SDs
score = mean |window_z| + mean |end_z|, over conditions and quantities; lowest wins.
The first term asks for a typical run, the second for a typical picture of it.
--window-start sets where the window starts (default 400 s, i.e. the last 100 s of a
500-s run); --quantities picks from the three above and
    lr_share     share of the >= 6 robots whose group centroid lies left or right of the
                 swarm centroid (|cos theta| >= cos 45 deg, world axes), where x_end puts
                 its catalysts (catalyst_end_azimuth.py's caps)

The titles use the raw grouping, as render_snapshots.py does, so a title can differ a
little from the smoothed numbers in trial_selection.csv.

--mark-cmd C [C ...] outlines, in orange, the robots whose last logged command (gait log)
at the drawn step is one of the codes, e.g. --mark-cmd -852 -52 for the catalysts
(strategy runs log the catalyst gait as -52, fixed-catalyst runs as -852).
A condition missing from cache/ (e.g. a fixed-catalyst run) takes its frame count from
<exp>_trial_stats.csv and its fps from config_snapshot.json; --trial auto needs the cache.

--with-start also draws each condition's first frame, as a top row (start) above the end row.
A condition whose cache name is not its directory name is given as NAME=EXP_DIR@CACHE.
--select-only writes trial_selection.csv and stops.

Output (in <results>/snapshots_<trial>/, except trial_selection.csv in <results>):
(<results>/<--subdir> instead, if given)
    <exp>_<trial>_t<sec>s.png         one per condition; the drawing only
    end_snapshots_<trial>.png         all conditions, with titles and legend
    end_snapshots_<trial>_bare.png    the same panels, with no text at all
    (start_end_snapshots_<trial>[_bare].png with --with-start)

Usage:
    python catalyst_end_snapshots.py <out> <results> [--trial auto] [--edge 0.75] [--mark-cmd -852 -52]
           [--subdir NAME] [--window-start 400] [--quantities ...] [--with-start] [--select-only]
           NAME=EXP_DIR[@CACHE] [NAME=EXP_DIR[@CACHE] ...]

<out> holds cache/. The first NAME=EXP_DIR is the baseline.
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

import catalyst_radial_figs as crf
from render_snapshots import CLASS_COLORS, MARK_COLOR, SURFACE, check_resources, draw, load_run
from catalyst_fixed_snapshots import command_mask

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pygame

WINDOW_S = 400.0
QUANTITIES = ("share_ge6", "outer_share", "largest")
C45 = np.cos(np.pi / 4)


def per_frame(path, edge, window_s=WINDOW_S):
    """Per-frame share_ge6, outer_share, largest, lr_share from window_s on (the last entry is the final frame)."""
    z = np.load(path, allow_pickle=True)
    T, N, R, fps = int(z["T"]), int(z["N"]), float(z["R"]), float(z["fps"])
    a = int(np.ceil(window_s * fps))
    t, s, xy = z["frames_t"], z["frames_size"].astype(float), z["frames_xy"]
    keep = t >= a
    t, s, xy = t[keep] - a, s[keep], xy[keep]
    n = T - a
    big = s >= 6
    dxy = xy - z["swarm_c"][t + a]
    rho = np.hypot(*dxy.T) / R
    cap = np.abs(dxy[:, 0]) >= C45 * np.maximum(np.hypot(*dxy.T), 1e-9)
    in_big = np.bincount(t[big], weights=s[big], minlength=n)
    in_out = np.bincount(t[big & (rho >= edge)], weights=s[big & (rho >= edge)], minlength=n)
    in_cap = np.bincount(t[big & cap], weights=s[big & cap], minlength=n)
    largest = np.zeros(n)
    np.maximum.at(largest, t, s)
    return dict(share_ge6=in_big / N, outer_share=in_out / np.maximum(in_big, 1), largest=largest,
                lr_share=in_cap / np.maximum(in_big, 1))


def run_length(out, exp, exp_dir, trial):
    """(T, fps) of a trial: from the radial cache if it has the run, else trial_stats.csv + config_snapshot.json."""
    c = os.path.join(out, "cache", f"{exp}__{trial}.npz")
    if os.path.exists(c):
        with np.load(c, allow_pickle=True) as z:
            return int(z["T"]), float(z["fps"])
    T = int(pd.read_csv(os.path.join(exp_dir, f"{exp}_trial_stats.csv")).set_index("trial").loc[trial, "frames"])
    cfg = json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))
    return T, float(cfg.get("RENDER_FPS_HEADLESS", 60))


def pick_trial(out, exps, edge, trials, results, window_s=WINDOW_S, quantities=QUANTITIES):
    rows = []
    for c, e in exps.items():
        fr = {tn: per_frame(os.path.join(out, "cache", f"{e}__{tn}.npz"), edge, window_s) for tn in trials}
        for q in quantities:
            means = np.array([fr[tn][q].mean() for tn in trials])
            pool = np.concatenate([fr[tn][q] for tn in trials])
            for i, tn in enumerate(trials):
                rows.append(dict(cond=c, trial=tn, quantity=q, window_mean=means[i], final_frame=fr[tn][q][-1],
                                 window_z=(means[i] - means.mean()) / means.std(ddof=1),
                                 end_z=(fr[tn][q][-1] - pool.mean()) / pool.std(ddof=1)))
    D = pd.DataFrame(rows)
    D.to_csv(os.path.join(results, "trial_selection.csv"), index=False)
    g = D.groupby("trial")
    S = pd.DataFrame({"window": g.window_z.apply(lambda v: v.abs().mean()), "end": g.end_z.apply(lambda v: v.abs().mean()),
                      "worst_window": g.window_z.apply(lambda v: v.abs().max()), "worst_end": g.end_z.apply(lambda v: v.abs().max())})
    S["score"] = S.window + S.end
    print(S.sort_values("score").round(3).to_string())
    return S.score.idxmin()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("results")
    ap.add_argument("conds", nargs="+", metavar="NAME=EXP_DIR[@CACHE]")
    ap.add_argument("--trial", default="auto")
    ap.add_argument("--edge", type=float, default=0.75)
    ap.add_argument("--n-trials", type=int, default=5)
    ap.add_argument("--window-start", type=float, default=WINDOW_S, help="selection window start in s (default 400)")
    ap.add_argument("--quantities", nargs="+", default=list(QUANTITIES),
                    choices=["share_ge6", "outer_share", "largest", "lr_share"])
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--downscale", type=int, default=None, help="default: VIDEO_DOWNSCALE")
    ap.add_argument("--margin", type=int, default=80)
    ap.add_argument("--mark-cmd", type=int, nargs="+", default=None, help="outline robots running any of these commands")
    ap.add_argument("--subdir", default=None, help="output folder in <results>; default snapshots_<trial>")
    ap.add_argument("--with-start", action="store_true", help="also draw the first frame, as a top row")
    ap.add_argument("--select-only", action="store_true", help="write trial_selection.csv and stop")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    dirs, exps = {}, {}
    for x in a.conds:
        c, d = x.split("=", 1)
        d, _, cache = d.partition("@")
        dirs[c] = d
        exps[c] = cache or os.path.basename(os.path.normpath(d))
    trials = [f"trial_{i:04d}" for i in range(a.n_trials)]
    if a.trial == "auto":
        trial = pick_trial(a.out, exps, a.edge, trials, a.results, a.window_start, a.quantities)
    else:
        trial = a.trial
    print("trial:", trial)
    if a.select_only:
        return
    out_dir = os.path.join(a.results, a.subdir or f"snapshots_{trial}")
    os.makedirs(out_dir, exist_ok=True)

    pygame.init()
    runs = []
    for c, d in dirs.items():
        name = os.path.basename(os.path.normpath(d))
        T, fps = run_length(a.out, exps[c], d, trial)
        with np.load(os.path.join(d, trial, f"{trial}_render.npz")) as r:
            stride = int(r["stride"])
        steps = ([0] if a.with_start else []) + [(T - 1) // stride * stride]   # last step that also has render state
        check_resources(out_dir)
        snap, frames = load_run(os.path.join(d, trial), steps, a.max_dist)
        for f, s in zip(frames, steps):
            f["mark"] = None if a.mark_cmd is None else command_mask(os.path.join(d, trial), s, fps, len(f["xyt"]), a.mark_cmd)
        print(f"{c} ({name}): steps {steps}" +
              ("" if a.mark_cmd is None else ", marked " + " / ".join(str(int(f["mark"].sum())) for f in frames)), flush=True)
        runs.append(dict(cond=c, name=name, snap=snap, frames=frames, secs=[s / fps for s in steps]))

    pts = np.vstack([np.vstack([f["xyt"][:, :2], f["ring"]]) for r in runs for f in r["frames"]])
    W, H = int(runs[0]["snap"]["W"]), int(runs[0]["snap"]["H"])
    x0, y0 = (np.floor(pts.min(0)) - a.margin).astype(int)
    x1, y1 = (np.ceil(pts.max(0)) + a.margin).astype(int)
    crop = (max(0, x0), max(0, y0), min(W, x1), min(H, y1))
    ds = a.downscale or int(runs[0]["snap"].get("VIDEO_DOWNSCALE", 1))

    nrow = len(runs[0]["frames"])
    h = 5.6 if a.mark_cmd is None else 6.3       # room for the catalyst line in the titles
    fig, axes = plt.subplots(nrow, len(runs), figsize=(4.2 * len(runs), h if nrow == 1 else 0.8 + 5.0 * nrow),
                             facecolor=SURFACE, layout="constrained", squeeze=False)
    summary = []
    for j, r in enumerate(runs):
        for i, (f, sec) in enumerate(zip(r["frames"], r["secs"])):
            ax = axes[i, j]
            img = draw(r["snap"], f, crop, ds, bodies_only=False, mark=f["mark"])
            pygame.image.save(img, os.path.join(out_dir, f"{r['name']}_{trial}_t{sec:03.0f}s.png"))
            ax.imshow(pygame.surfarray.array3d(img).swapaxes(0, 1))
            big = f["cls"] == "6+"
            c = f["xyt"][:, :2].mean(axis=0)     # the swarm centroid, as in the radial analysis
            rho = np.hypot(*(f["xyt"][:, :2] - c).T) / float(r["snap"]["INNER_R"])
            outer = (rho[big] >= a.edge).mean() if big.any() else float("nan")
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(True)         # catalyst_radial_figs turns the top and right ones off
                s.set_color("#d0cfca")
            role = "baseline" if r["cond"] == "baseline" else r["cond"]
            m = f["mark"]
            cat = "" if m is None or not m.any() else f"\ncatalysts in ≥6: {(m & big).sum()} / {m.sum()}"
            where = f"\nof those ≥6, {outer * 100:.0f}% outside {a.edge:g} R" if big.any() else ""
            ax.set_title(f"{role} ({r['name']}), t = {sec:.0f} s\n"
                         f"≥6: {f['share6'] * 100:.0f}% of robots in {f['n6']} groups · ≥3: {f['share3'] * 100:.0f}%"
                         + where + cat, fontsize=9.5, fontweight="normal")
            summary.append(dict(cond=r["cond"], exp=r["name"], trial=trial, t_s=sec, share_ge6=f["share6"], n_ge6=f["n6"],
                                share_ge3=f["share3"], outer_share_of_ge6_robots=outer))
            if m is not None:
                summary[-1].update(n_catalysts=int(m.sum()), cat_in_ge6=int((m & big).sum()),
                                   cat_in_3to5=int((m & (f["cls"] == "3-5")).sum()),
                                   cat_in_1to2=int((m & (f["cls"] == "1-2")).sum()))
    stem = ("start_end_snapshots_" if nrow > 1 else "end_snapshots_") + trial
    pd.DataFrame(summary).to_csv(os.path.join(out_dir, f"{stem}.csv"), index=False)
    handles = [Patch(color=col, label=f"robot in a group of size {k}") for k, col in CLASS_COLORS.items()]
    if a.mark_cmd is not None:
        handles.append(Patch(facecolor="none", edgecolor=MARK_COLOR, lw=2,
                             label="catalyst (" + " / ".join(str(v) for v in a.mark_cmd) + ")"))
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles), frameon=False, fontsize=9.5)
    fig.suptitle(f"{runs[0]['name']} vs " + " / ".join(r["cond"] for r in runs[1:]) + f", {trial}" +
                 (": start (top) and end (bottom)" if nrow > 1 else ""), fontsize=11, fontweight="normal")
    fig.savefig(os.path.join(out_dir, f"{stem}.png"), dpi=170, facecolor=SURFACE)
    crf.strip_text(fig)
    for ax in axes.ravel():
        for s in ax.spines.values():
            s.set_visible(False)
    fig.savefig(os.path.join(out_dir, f"{stem}_bare.png"), dpi=170, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    pygame.quit()
    print(pd.DataFrame(summary).round(3).to_string())


if __name__ == "__main__":
    sys.exit(main())

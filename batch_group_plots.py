"""
batch_group_plots.py
--------------------
Batch-plot "group composition over time" figures, plus cross-trial averaged
statistics for each group of experiments.

The input is **a list of experiment directories**, each containing a series of
trial folders:

    datafile/exp_A/                 <- this is the level passed on the command line
        config_snapshot.json
        trial_0000/trial_0000_POS_ALL.csv
        trial_0001/trial_0001_POS_ALL.csv
        ...
    datafile/exp_B/
        ...

Usage:
    python batch_group_plots.py datafile/exp_A datafile/exp_B
    python batch_group_plots.py "datafile/*"                  # wildcard
    python batch_group_plots.py --dirs-from list.txt          # one experiment dir per line
    python batch_group_plots.py datafile/exp_A --figs composition mean_groupsize

Output has two tiers. **Per trial** (written back into that trial's folder by
default):

    <trial>_composition.png   how many robots are in each size bin? (stacked area)
    <trial>_groupsize.png     largest group, and "grab a random robot, how big is its group" over time
    <trial>_kymograph.png     who is grouped with whom, and when?
    <trial>_counts.png        number of groups, largest-cluster fraction
    <trial>_snapshots.png     what it looks like spatially

**Per experiment** (written into the experiment directory), averaging all
trials of that experiment:

    <exp>_mean_composition.png   averaged stacked area
    <exp>_mean_groupsize.png     thin lines = individual trials, bold = mean, band = ±1 std
    <exp>_mean_counts.png        mean and spread of group count / largest-cluster fraction
    <exp>_trial_stats.csv        one row of scalar stats per trial
    <exp>_mean_timeseries.csv    mean/std of each quantity after aligning to a common time axis

The time axis's t=0 is **the first recorded frame**, not the simulation's
t=0: RECORD_AFTER_WARMUP skips the WARMUP_STEPS warm-up period, so a 12s
trial has only about 11s of data on disk. This way t=0 lines up to the same
physical moment (right after warm-up) across trials, so cross-trial averaging
is aligned correctly.

Before averaging across trials, series are interpolated onto a common time
axis (the length of the shortest trial; extrapolating past that is meaningless
since the shortest trial simply has no data there). A duration mismatch is
reported in the log along with the truncated length.

Grouping results are cached as trial_XXXX_groups_d<max_dist>.csv (columns
match pos_all_grouping's process_pos_all_groups exactly); a re-plot reads the
cache directly next time, --force forces a recompute. Caches for different
max_dist values don't overwrite each other -- the threshold is in the filename.

The stacked-area plot shows the **smoothed** share (2s moving average by
default, changeable with --smooth-s, 0 disables it). So a band can read
thinner than its own bin's lower bound: a frame might have one 16-robot group
giving an instantaneous share of 16%, but if that group only exists for 30% of
a 2s window, the plotted value is 0.3*16% ~= 5%. In other words the band's
thickness is "how much robot-time fell into this bin over the window", not the
share at any single instant. Use --smooth-s 0 to see instantaneous values.

Plotting reuses the colour scheme and smoothing approach from the hardware-side
plot_group_evolution.py; the data source differs: hardware reads
group<stamp>.csv with a Time column, here time is computed from Step / fps,
with fps preferentially read from the experiment directory's
config_snapshot.json (RENDER_FPS_HEADLESS).

Grouping uses pos_all_grouping.compute_frame_groups -- the same function
strategy.py's realtime decision-making calls, verified frame-by-frame to
produce identical groupings (provided max_dist matches).
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pos_all_grouping import compute_frame_groups

# Figure labels are always English: usable for both papers and reports, and it
# avoids a cross-machine Chinese font dependency
# (Windows has SimHei, Linux often has nothing, so the same script would render
# differently on the two).
plt.rcParams["axes.unicode_minus"] = False

FIG_TRIAL = ("composition", "groupsize", "kymograph", "counts", "snapshots")
FIG_EXP = ("mean_composition", "mean_groupsize", "mean_counts")
FIG_ALIASES = {
    "all": FIG_TRIAL + FIG_EXP,
    "trial": FIG_TRIAL,
    "exp": FIG_EXP,
    "aggregation": ("composition", "groupsize"),   # the two halves of the original combined figure
}

SIZE_COLORS = ["#d9d9d9", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
C_LARGEST, C_MEANSZ, C_ALIGN = "#08306b", "#e6550d", "#31a354"

EPILOG = """\
The positional arguments are experiment directories, each containing
trial_XXXX/ subfolders.
--figs options:
    per trial      : composition groupsize kymograph counts snapshots
    per experiment : mean_composition mean_groupsize mean_counts
    aliases        : all / trial / exp / aggregation(=composition+groupsize)
"""


def size_bins(n):
    """
    Bin edges scale with population size. The hardware-side version hard-codes
    a 9+ cap, which is fine for 17 robots; at 100 robots "9+" would dump the
    vast majority of robots into one bin and the plot would carry no information.
    """
    if n <= 20:
        edges = [1, 3, 5, 8]
    elif n <= 60:
        edges = [1, 3, 6, 12]
    else:
        edges = [1, 5, 10, 15]
    bins, labels, lo = [], [], 1
    for e in edges:
        bins.append((lo, e))
        labels.append(str(lo) if lo == e else f"{lo}-{e}")
        lo = e + 1
    bins.append((lo, max(lo, n)))
    labels.append(f"{lo}+")
    return bins, labels


# =============================================================================
# Loading: POS_ALL -> per-frame grouping
# =============================================================================

def find_pos_all(folder):
    hits = sorted(glob.glob(os.path.join(folder, "*_POS_ALL.csv")))
    return hits[0] if hits else None


def find_trials(exp_dir):
    """
    All direct subdirectories of the experiment directory that contain
    *_POS_ALL.csv, sorted by name.

    If the experiment directory itself directly holds a POS_ALL (someone
    passed a single trial directory in for convenience), treat it as "an
    experiment with just one trial" instead of silently doing nothing.
    """
    out = [sub for sub in sorted(glob.glob(os.path.join(exp_dir, "*")))
           if os.path.isdir(sub) and find_pos_all(sub)]
    if not out and find_pos_all(exp_dir):
        out = [exp_dir]
    return out


def detect_fps(folder, fallback=60.0):
    """
    How many POS_ALL frames are recorded per second. Prefers this experiment's
    own config_snapshot.json (run_trial writes it into the experiment
    directory), falling back to the current config.py.
    """
    here = os.path.abspath(folder)
    for d in (here, os.path.dirname(here)):
        path = os.path.join(d, "config_snapshot.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    v = json.load(f).get("RENDER_FPS_HEADLESS")
                if v:
                    return float(v), "config_snapshot.json"
            except Exception:
                pass
    try:
        from config import RENDER_FPS_HEADLESS
        return float(RENDER_FPS_HEADLESS), "config.py"
    except Exception:
        return float(fallback), "fallback"


def group_table(pos_all_csv, max_dist, include_singletons=True, force=False):
    """
    One row per group per frame, columns matching
    pos_all_grouping.process_pos_all_groups.

    Results are cached alongside the source as <prefix>_groups_d<max_dist>.csv.
    The computation itself calls compute_frame_groups; the only change from
    process_pos_all_groups is replacing its "rescan the whole table every
    frame" (O(frames^2)) with a sort-then-slice -- the grouping logic itself is
    untouched.
    """
    tag = f"{max_dist:g}".replace(".", "p")
    cache = pos_all_csv.replace("_POS_ALL.csv", f"_groups_d{tag}.csv")
    if not include_singletons:
        cache = cache.replace(".csv", "_nosing.csv")
    if os.path.isfile(cache) and not force:
        return pd.read_csv(cache), cache, True

    df = pd.read_csv(pos_all_csv)
    df.columns = df.columns.str.strip()
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")

    step_col = df["Step"].to_numpy()
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    th = df["Theta"].to_numpy(dtype=float)
    aid = df["Agent_ID"].to_numpy()

    steps, starts = np.unique(step_col, return_index=True)
    bounds = np.append(starts, len(step_col))

    rows = []
    for k, step in enumerate(steps):
        sl = slice(bounds[k], bounds[k + 1])
        for gi, g in enumerate(compute_frame_groups(
                xy[sl], th[sl], aid[sl], max_dist, include_singletons)):
            rows.append({
                "Step": int(step),
                "group_local_id": gi,
                "size": g["size"],
                "alignment": g["alignment"],
                "centroid_x": g["centroid"][0],
                "centroid_y": g["centroid"][1],
                "member_ids": " ".join(str(int(a)) for a in g["agent_ids"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out, cache, False


def load_frames(groups_df, fps):
    """-> [(step, seconds, [ {members, size, alignment, centroid}, ... ]), ...]"""
    frames = []
    for step, g in groups_df.groupby("Step", sort=True):
        frames.append((int(step), float(step) / fps, [{
            "members": frozenset(int(x) for x in str(r["member_ids"]).split()),
            "size": int(r["size"]),
            "alignment": float(r["alignment"]),
            "centroid": (float(r["centroid_x"]), float(r["centroid_y"])),
        } for _, r in g.iterrows()]))
    return frames


def track_groups(frames, min_size=2, min_jaccard=0.3):
    """
    Assign each frame's groups a cross-frame-stable id: greedy one-to-one
    matching by Jaccard overlap of member sets. Same idea as
    pos_all_grouping.track_groups; reimplemented independently here because
    plotting wants to track every group with size>=2, while that version is
    organized around a DataFrame.

    -> [ {gid: members}, ... ] one dict per frame
    """
    prev = {}
    next_gid = 1
    out = []
    for _step, _sec, groups in frames:
        cur = [g["members"] for g in groups if g["size"] >= min_size]
        pairs = []
        for gid, pm in prev.items():
            for ci, m in enumerate(cur):
                inter = len(pm & m)
                if inter:
                    j = inter / len(pm | m)
                    if j >= min_jaccard:
                        pairs.append((-j, -inter, gid, ci))
        pairs.sort()
        used_g, used_c, now = set(), set(), {}
        for _j, _i, gid, ci in pairs:
            if gid in used_g or ci in used_c:
                continue
            used_g.add(gid); used_c.add(ci)
            now[gid] = cur[ci]
        for ci, m in enumerate(cur):
            if ci not in used_c:
                now[next_gid] = m
                next_gid += 1
        out.append(now)
        prev = now
    return out


# =============================================================================
# Per-trial time series (consumed by both plotting and cross-trial averaging)
# =============================================================================

def trial_series(frames, bins):
    """Collapse the per-frame grouping results into a handful of equal-length time series."""
    m = len(frames)
    secs = np.array([f[1] for f in frames], dtype=float)
    frac = np.zeros((m, len(bins)))
    largest = np.empty(m); mean_sz = np.empty(m); align = np.empty(m)
    n_all = np.empty(m); n_multi = np.empty(m); n_seen = np.empty(m)

    for i, (_s, _t, groups) in enumerate(frames):
        sizes = [g["size"] for g in groups]
        total = sum(sizes)
        for g in groups:
            for b, (lo, hi) in enumerate(bins):
                if lo <= g["size"] <= hi:
                    frac[i, b] += g["size"] / max(total, 1)
                    break
        largest[i] = max(sizes)
        # Robot's-eye-view mean group size: grab a random robot, how big is its group
        mean_sz[i] = sum(s * s for s in sizes) / max(total, 1)
        align[i] = sum(g["size"] * g["alignment"] for g in groups) / max(total, 1)
        n_all[i] = len(groups)
        n_multi[i] = sum(1 for s in sizes if s >= 2)
        n_seen[i] = total

    return {"secs": secs, "frac": frac, "largest": largest, "mean_size": mean_sz,
            "alignment": align, "n_groups": n_all, "n_multi": n_multi,
            "n_seen": n_seen, "largest_frac": largest / np.maximum(n_seen, 1)}


def smooth_window(secs, seconds):
    """Convert "how many seconds" into a moving-window frame count; <=0 means no smoothing."""
    if seconds is None or seconds <= 0:
        return 1
    span = max(float(secs[-1]) - float(secs[0]), 1e-9)
    return max(3, int(round(seconds * len(secs) / span)))


def roll_mean(v, win):
    if win <= 1:
        return np.asarray(v, dtype=float)
    return pd.Series(v).rolling(win, center=True, min_periods=1).mean().to_numpy()


def roll_median(v, win):
    if win <= 1:
        return np.asarray(v, dtype=float)
    return pd.Series(v).rolling(win, center=True, min_periods=1).median().to_numpy()


# =============================================================================
# Figure 1a: composition (how many robots in each size bin)
# =============================================================================

def fig_composition(sr, labels, path, title="", smooth_s=2.0, note=None):
    secs = sr["secs"]
    # Per-frame shares are very jittery (a single Voronoi edge in one frame can
    # merge and split two groups), so plotting them raw would be drowned in
    # noise. A ~2s moving average is used so the trend is actually visible.
    win = smooth_window(secs, smooth_s)
    frac_s = np.column_stack([roll_mean(sr["frac"][:, b], win)
                              for b in range(sr["frac"].shape[1])])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.stackplot(secs, frac_s.T * 100, labels=labels,
                 colors=SIZE_COLORS[:len(labels)], edgecolor="none")
    ax.set_ylim(0, 100)
    ax.set_xlim(secs[0], secs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("share of robots (%)")
    ax.set_title(f"Group size composition{title}")
    ax.legend(title="group size", loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(labels), frameon=False, fontsize=9)
    if note is None:
        note = (f"({smooth_s:g}s moving average — a band can read below its own "
                f"size when it is occupied only part of the window)"
                if win > 1 else "(per-frame share, no smoothing)")
    ax.text(0.995, 0.03, note, transform=ax.transAxes, ha="right",
            fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Figure 1b: group size
# =============================================================================

def fig_groupsize(sr, path, title="", smooth_s=2.0):
    secs = sr["secs"]
    win = smooth_window(secs, smooth_s)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(secs, sr["largest"], lw=0.8, color=C_LARGEST, alpha=0.30)
    ax.plot(secs, roll_mean(sr["largest"], win), lw=2.2, color=C_LARGEST,
            label="largest group")
    ax.plot(secs, roll_mean(sr["mean_size"], win), lw=2.2, color=C_MEANSZ,
            label="mean group size (per robot)")
    ax.set_xlim(secs[0], secs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("group size")
    ax.set_title(f"Group size over time{title}")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(secs, roll_mean(sr["alignment"], win), lw=1.2, color=C_ALIGN, ls="--")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("size-weighted alignment", color=C_ALIGN)
    ax2.tick_params(axis="y", colors=C_ALIGN)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Figure 2: kymograph (time x robot)
# =============================================================================

def fig_kymograph(frames, tracked, path, min_life=6, title=""):
    ids = sorted({m for _s, _t, gs in frames for g in gs for m in g["members"]})
    row = {m: i for i, m in enumerate(ids)}
    secs = [f[1] for f in frames]

    # The group id each robot belongs to on each frame; -1 for a lone robot
    M = np.full((len(ids), len(frames)), -1, dtype=int)
    for j, now in enumerate(tracked):
        for gid, members in now.items():
            for m in members:
                if m in row:
                    M[row[m], j] = gid

    # Only assign colours to groups that "live long enough". Groups that
    # exist for just a frame or two before dispersing all get a single
    # neutral light purple, otherwise hundreds of transient groups would each
    # claim their own colour and the plot would turn into coloured noise.
    life = {g: int((M == g).any(axis=0).sum()) for g in np.unique(M) if g >= 0}
    stable = sorted([g for g, n in life.items() if n >= min_life])
    cmap = plt.get_cmap("tab20")
    color = {g: cmap(i % 20) for i, g in enumerate(stable)}
    rgb = np.ones(M.shape + (3,))
    rgb[M >= 0] = (0.72, 0.72, 0.78)         # briefly grouped
    for g, c in color.items():
        rgb[M == g] = c[:3]
    rgb[M < 0] = (0.95, 0.95, 0.95)          # alone

    # With many robots, per-row labels would blur together, so use sparse ticks
    step = max(1, len(ids) // 40)
    fig, ax = plt.subplots(figsize=(12, min(24, 0.32 * len(ids) + 2)))
    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower",
              extent=[secs[0], secs[-1], -0.5, len(ids) - 0.5])
    ax.set_yticks(range(0, len(ids), step))
    ax.set_yticklabels([ids[i] for i in range(0, len(ids), step)], fontsize=8)
    ax.set_ylabel("robot ID")
    ax.set_xlabel("time (s)")
    # Legend text goes on a second line: with many robots a one-line title
    # would get its right half clipped by the canvas
    ax.set_title(f"Group membership per robot{title}\n"
                 f"colour = groups lasting >{min_life} frames   |   "
                 f"light purple = transient   |   grey = alone",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Figure 3: counts
# =============================================================================

def fig_counts(sr, path, title="", smooth_s=2.0):
    secs = sr["secs"]
    w = smooth_window(secs, smooth_s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(secs, sr["n_groups"], lw=0.7, alpha=0.3, color="#666")
    a1.plot(secs, roll_median(sr["n_groups"], w), lw=2, color="#666",
            label="all groups")
    a1.plot(secs, roll_median(sr["n_multi"], w), lw=2, color="#2171b5",
            label="groups with >=2")
    a1.plot(secs, roll_median(sr["n_seen"], w), lw=1.2, ls=":", color="#999",
            label="robots detected")
    a1.set_ylabel("count")
    a1.set_title(f"Group counts{title}", fontsize=11)
    a1.legend(frameon=False, ncol=3, fontsize=9)
    a1.grid(alpha=0.25)

    a2.plot(secs, sr["largest_frac"], lw=0.7, alpha=0.3, color=C_LARGEST)
    a2.plot(secs, roll_median(sr["largest_frac"], w), lw=2.2, color=C_LARGEST)
    a2.set_ylim(0, 1)
    a2.set_xlim(secs[0], secs[-1])
    a2.set_ylabel("largest group / all")
    a2.set_xlabel("time (s)")
    a2.grid(alpha=0.25)
    a2.set_title("Largest cluster fraction", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Figure 4: spatial snapshots
# =============================================================================

def fig_snapshots(frames, tracked, path, n=5, title=""):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    cmap = plt.get_cmap("tab20")
    xs = [g["centroid"][0] for _s, _t, gs in frames for g in gs]
    ys = [g["centroid"][1] for _s, _t, gs in frames for g in gs]
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        _s, sec, groups = frames[i]
        for g in groups:
            gid = next((k for k, v in tracked[i].items() if v == g["members"]), None)
            c = cmap(gid % 20) if gid is not None else (0.8, 0.8, 0.8)
            ax.scatter(*g["centroid"], s=25 + 45 * g["size"], color=c,
                       edgecolor="k", linewidth=0.4, alpha=0.85)
            if g["size"] >= 2:
                ax.annotate(str(g["size"]), g["centroid"], fontsize=7,
                            ha="center", va="center")
        ax.set_title(f"t = {sec:.0f}s", fontsize=10)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)   # simulation world y points down, matching the video
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"Spatial snapshots{title}\n"
                 "marker size ~ group size, same colour = same group",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Cross-trial averaging
# =============================================================================

def align_series(all_sr, n_points=None):
    """
    Interpolate a number of trials' series onto a common time axis.

    The axis spans 0 .. min(trial durations): extrapolating is meaningless
    since the shortest trial simply has no data past that point, and padding
    it in would fabricate the illusion that "every trial is still running".
    """
    t_end = min(float(sr["secs"][-1]) for sr in all_sr)
    n_points = n_points or max(len(sr["secs"]) for sr in all_sr)
    grid = np.linspace(0.0, t_end, int(n_points))

    keys = ("largest", "mean_size", "alignment", "n_groups", "n_multi",
            "n_seen", "largest_frac")
    stacked = {k: np.stack([np.interp(grid, sr["secs"], sr[k]) for sr in all_sr])
               for k in keys}
    nb = all_sr[0]["frac"].shape[1]
    stacked["frac"] = np.stack([                              # (trials, T, bins)
        np.stack([np.interp(grid, sr["secs"], sr["frac"][:, b])
                  for b in range(nb)], axis=1)
        for sr in all_sr])
    return grid, stacked, t_end


def fig_mean_composition(grid, stacked, labels, path, title="", smooth_s=2.0):
    n = stacked["frac"].shape[0]
    note = (f"(mean of {n} trials, {smooth_s:g}s moving average — a band can "
            f"read below its own size)" if smooth_s and smooth_s > 0
            else f"(mean of {n} trials, no smoothing)")
    fig_composition({"secs": grid, "frac": stacked["frac"].mean(axis=0)},
                    labels, path, title, smooth_s, note=note)


def _band(ax, grid, arr, color, label, win, show_trials=True):
    """Thin lines = individual trials, bold line = mean, band = ±1 std."""
    if show_trials:
        for r in arr:
            ax.plot(grid, roll_mean(r, win), lw=0.7, color=color, alpha=0.25)
    m = roll_mean(arr.mean(axis=0), win)
    s = roll_mean(arr.std(axis=0), win)
    ax.fill_between(grid, m - s, m + s, color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, m, lw=2.4, color=color, label=label)


def fig_mean_groupsize(grid, stacked, path, title="", smooth_s=2.0):
    win = smooth_window(grid, smooth_s)
    n = stacked["largest"].shape[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    _band(a1, grid, stacked["largest"], C_LARGEST, "largest group", win)
    a1.set_ylabel("largest group size")
    a1.set_title(f"Group size across trials{title}\n"
                 f"thin = individual trials (n={n}), bold = mean, band = ±1 std",
                 fontsize=11)
    a1.legend(frameon=False, loc="upper left")
    a1.grid(alpha=0.25)

    _band(a2, grid, stacked["mean_size"], C_MEANSZ,
          "mean group size (per robot)", win)
    a2.set_ylabel("group size")
    a2.set_xlabel("time (s)")
    a2.set_xlim(grid[0], grid[-1])
    a2.legend(frameon=False, loc="upper left")
    a2.grid(alpha=0.25)

    ax = a2.twinx()
    ax.plot(grid, roll_mean(stacked["alignment"].mean(axis=0), win),
            lw=1.2, color=C_ALIGN, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("size-weighted alignment", color=C_ALIGN)
    ax.tick_params(axis="y", colors=C_ALIGN)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_mean_counts(grid, stacked, path, title="", smooth_s=2.0):
    win = smooth_window(grid, smooth_s)
    n = stacked["n_groups"].shape[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)

    _band(a1, grid, stacked["n_groups"], "#666", "all groups", win)
    _band(a1, grid, stacked["n_multi"], "#2171b5", "groups with >=2", win,
          show_trials=False)
    a1.set_ylabel("count")
    a1.set_title(f"Group counts across trials{title}\n"
                 f"thin = individual trials (n={n}), bold = mean, band = ±1 std",
                 fontsize=11)
    a1.legend(frameon=False, ncol=2, fontsize=9)
    a1.grid(alpha=0.25)

    _band(a2, grid, stacked["largest_frac"], C_LARGEST, "largest / all", win)
    a2.set_ylim(0, 1)
    a2.set_xlim(grid[0], grid[-1])
    a2.set_ylabel("largest group / all")
    a2.set_xlabel("time (s)")
    a2.legend(frameon=False, loc="upper left")
    a2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# Process one trial / one group of experiments
# =============================================================================

def process_trial(folder, exp_name, bins, labels, args, fps):
    pos_all = find_pos_all(folder)
    name = os.path.basename(os.path.normpath(folder))

    gdf, cache, cached = group_table(pos_all, args.max_dist,
                                     not args.no_singletons, args.force)
    if gdf.empty:
        print(f"    [skip] {name}: group table is empty")
        return None

    frames = load_frames(gdf, fps)
    sr = trial_series(frames, bins)

    out_dir = args.out or folder
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{exp_name}_{name}" if args.out else name
    title = f"  [{exp_name}/{name}, max_dist={args.max_dist:g}]" if args.title else ""

    made = []

    def p(fig_name):
        fn = f"{prefix}_{fig_name}.png"
        made.append(fn)
        return os.path.join(out_dir, fn)

    if "composition" in args.figs:
        fig_composition(sr, labels, p("composition"), title, args.smooth_s)
    if "groupsize" in args.figs:
        fig_groupsize(sr, p("groupsize"), title, args.smooth_s)
    if "counts" in args.figs:
        fig_counts(sr, p("counts"), title, args.smooth_s)
    # Cross-frame tracking is only needed for these two figures, so other
    # combinations don't have to pay that cost
    if {"kymograph", "snapshots"} & args.figs:
        tracked = track_groups(frames, min_size=args.min_size)
        if "kymograph" in args.figs:
            fig_kymograph(frames, tracked, p("kymograph"),
                          min_life=max(3, int(round(args.min_life_s * fps))),
                          title=title)
        if "snapshots" in args.figs:
            fig_snapshots(frames, tracked, p("snapshots"), args.snapshots, title)

    tail = sr["secs"] >= sr["secs"][-1] - args.steady_s
    stats = {
        "exp": exp_name, "trial": name,
        "n_robots": int(sr["n_seen"].max()),
        "frames": len(frames), "duration_s": float(sr["secs"][-1]),
        "largest_mean": float(sr["largest"].mean()),
        "largest_steady": float(sr["largest"][tail].mean()),
        "largest_peak": int(sr["largest"].max()),
        "mean_size_steady": float(sr["mean_size"][tail].mean()),
        "n_groups_steady": float(sr["n_groups"][tail].mean()),
        "largest_frac_steady": float(sr["largest_frac"][tail].mean()),
        "alignment_steady": float(sr["alignment"][tail].mean()),
    }
    print(f"    {name}: {len(frames)} frames / {stats['duration_s']:.1f}s, "
          f"largest group mean {stats['largest_mean']:.1f} / last {args.steady_s:g}s "
          f"{stats['largest_steady']:.1f} / peak {stats['largest_peak']}"
          f"   [{'cached' if cached else 'computed'}]")
    for fn in made:
        print(f"      -> {fn}")
    return sr, stats


STEADY_COLS = ("largest_mean", "largest_steady", "largest_peak",
               "mean_size_steady", "n_groups_steady",
               "largest_frac_steady", "alignment_steady")


def process_experiment(exp_dir, args):
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    trials = find_trials(exp_dir)
    if not trials:
        print(f"  [skip] no trial subdirectories containing *_POS_ALL.csv")
        return None

    fps, fps_src = (args.fps, "--fps") if args.fps else detect_fps(exp_dir)
    print(f"  {len(trials)} trials, fps={fps:g} ({fps_src})")

    # Bin edges must be shared across the whole experiment, otherwise the
    # trials' stacked-area plots aren't comparable and can't be averaged
    # together. Scan each trial's robot count first and use the max to set the bins.
    n_max = 0
    for t in trials:
        n_max = max(n_max, int(pd.read_csv(find_pos_all(t),
                                           usecols=["Agent_ID"])
                               ["Agent_ID"].nunique()))
    bins, labels = size_bins(n_max)

    all_sr, all_stats = [], []
    for t in trials:
        r = process_trial(t, exp_name, bins, labels, args, fps)
        if r:
            all_sr.append(r[0])
            all_stats.append(r[1])
    if not all_sr:
        return None

    out_dir = args.out or exp_dir
    os.makedirs(out_dir, exist_ok=True)
    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(out_dir, f"{exp_name}_trial_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"    -> {os.path.basename(stats_path)}  ({len(stats_df)} rows)")

    grid, stacked, t_end = align_series(all_sr)
    durations = [float(sr["secs"][-1]) for sr in all_sr]
    if max(durations) - min(durations) > 1e-6:
        print(f"    [note] trial durations differ "
              f"({min(durations):.1f}~{max(durations):.1f}s), "
              f"cross-trial average truncated to {t_end:.1f}s")

    ts = pd.DataFrame({"time": grid})
    for k in ("largest", "mean_size", "alignment", "n_groups", "n_multi",
              "largest_frac"):
        ts[f"{k}_mean"] = stacked[k].mean(axis=0)
        ts[f"{k}_std"] = stacked[k].std(axis=0)
    for b, lab in enumerate(labels):
        ts[f"share_{lab}_mean"] = stacked["frac"][:, :, b].mean(axis=0)
        ts[f"share_{lab}_std"] = stacked["frac"][:, :, b].std(axis=0)
    ts_path = os.path.join(out_dir, f"{exp_name}_mean_timeseries.csv")
    ts.to_csv(ts_path, index=False)
    print(f"    -> {os.path.basename(ts_path)}")

    title = f"  [{exp_name}, max_dist={args.max_dist:g}]" if args.title else ""
    sm = args.smooth_s
    for fig_name, fn in (("mean_composition",
                          lambda p: fig_mean_composition(grid, stacked, labels,
                                                         p, title, sm)),
                         ("mean_groupsize",
                          lambda p: fig_mean_groupsize(grid, stacked, p, title, sm)),
                         ("mean_counts",
                          lambda p: fig_mean_counts(grid, stacked, p, title, sm))):
        if fig_name in args.figs:
            fp = os.path.join(out_dir, f"{exp_name}_{fig_name}.png")
            fn(fp)
            print(f"    -> {os.path.basename(fp)}")

    row = {"exp": exp_name, "n_trials": len(stats_df),
           "n_robots": int(stats_df["n_robots"].max()),
           "duration_s": t_end}
    for c in STEADY_COLS:
        row[f"{c}_mean"] = float(stats_df[c].mean())
        row[f"{c}_std"] = float(stats_df[c].std(ddof=0))
    print(f"    experiment average (n={len(stats_df)}): largest group, last {args.steady_s:g}s "
          f"{row['largest_steady_mean']:.2f} ± {row['largest_steady_std']:.2f}, "
          f"group count {row['n_groups_steady_mean']:.2f} ± "
          f"{row['n_groups_steady_std']:.2f}, "
          f"largest-cluster fraction {row['largest_frac_steady_mean']:.3f} ± "
          f"{row['largest_frac_steady_std']:.3f}")
    return row


# =============================================================================
# Entry point
# =============================================================================

def expand_experiments(patterns):
    out, seen = [], set()
    for pat in patterns:
        hits = glob.glob(pat) or ([pat] if os.path.isdir(pat) else [])
        for h in sorted(hits):
            d = os.path.normpath(h)
            if os.path.isdir(h) and d not in seen:
                seen.add(d)
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Batch-plot group evolution figures, and average across trials for each experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    ap.add_argument("experiments", nargs="*",
                    help="Experiment directories (each containing trial_XXXX/ subfolders); wildcards allowed")
    ap.add_argument("--dirs-from", default=None,
                    help="Read the list of experiment directories from a file, one per line (lines starting with # are comments)")
    ap.add_argument("--max-dist", type=float, default=65.0,
                    help="Voronoi adjacency distance threshold, in pixels (default 65)")
    ap.add_argument("--no-singletons", action="store_true",
                    help="Don't count lone robots as size=1 groups")
    ap.add_argument("--figs", nargs="+",
                    choices=list(FIG_TRIAL) + list(FIG_EXP) + list(FIG_ALIASES),
                    default=["all"], help="Only plot these figures (default: all)")
    ap.add_argument("--out", default=None,
                    help="Write all figures and csv output to this directory (default: back into each experiment's own directory)")
    ap.add_argument("--fps", type=float, default=None,
                    help="Recording frame rate, frames per second (default: read from config_snapshot.json)")
    ap.add_argument("--smooth-s", type=float, default=2.0,
                    help="Moving-average window for the line/stacked-area plots, in seconds; 0 = no smoothing (default 2)")
    ap.add_argument("--steady-s", type=float, default=10.0,
                    help="How many trailing seconds count as steady state, used for scalar statistics (default 10)")
    ap.add_argument("--min-size", type=int, default=2,
                    help="Minimum size counted as a 'group' in the kymograph")
    ap.add_argument("--min-life-s", type=float, default=1.5,
                    help="Minimum lifetime, in seconds, for a group to get its own color in the kymograph")
    ap.add_argument("--snapshots", type=int, default=5,
                    help="Number of time points to plot in the spatial snapshots")
    ap.add_argument("--title", action="store_true",
                    help="Show the experiment/trial name and max_dist in the figure title")
    ap.add_argument("--force", action="store_true",
                    help="Ignore any existing group cache and recompute")
    ap.add_argument("--summary", default=None,
                    help="Write the summary across experiments to a single csv file")
    args = ap.parse_args()

    figs = set()
    for f in args.figs:
        figs |= set(FIG_ALIASES.get(f, (f,)))
    args.figs = figs

    patterns = list(args.experiments)
    if args.dirs_from:
        with open(args.dirs_from, encoding="utf-8") as f:
            patterns += [ln.strip() for ln in f
                         if ln.strip() and not ln.lstrip().startswith("#")]
    if not patterns:
        ap.error("You must give at least one experiment directory (or use --dirs-from)")

    exps = expand_experiments(patterns)
    if not exps:
        print("No directories found. The positional arguments should be "
              "experiment directories containing trial_XXXX/ subfolders.")
        return 1

    print(f"{len(exps)} experiments total, max_dist={args.max_dist:g}, "
          f"plotting {', '.join(sorted(args.figs))}")
    rows = []
    for i, exp in enumerate(exps, 1):
        print(f"[{i}/{len(exps)}] {exp}")
        try:
            r = process_experiment(exp, args)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  [FAIL] {exp}: {type(e).__name__}: {e}")

    if args.summary and rows:
        pd.DataFrame(rows).to_csv(args.summary, index=False)
        print(f"\nSummary written to {args.summary}")
    print(f"\nDone: {len(rows)}/{len(exps)} experiments")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

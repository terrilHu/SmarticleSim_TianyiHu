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
    <trial>_patterns.png      share of robots running each gait pattern (opt-in)

**Per experiment** (written into the experiment directory), averaging all
trials of that experiment:

    <exp>_mean_composition.png   averaged stacked area
    <exp>_mean_groupsize.png     thin lines = individual trials, bold = mean, band = ±1 std
    <exp>_mean_counts.png        mean and spread of group count / largest-cluster fraction
    <exp>_mean_patterns.png      averaged gait-pattern share (opt-in)
    <exp>_radial.png             where in the arena groups form (opt-in)
    <exp>_chains.png             whether those groups run ALONG the ring (opt-in)
    <exp>_trial_stats.csv        one row of scalar stats per trial
    <exp>_mean_timeseries.csv    mean/std of each quantity after aligning to a common time axis

The last two figures are **opt-in** -- `--figs` defaults to everything else,
because each has a cost the rest don't:

`patterns` / `mean_patterns` plot which gait each robot was assigned over time.
The quantity is the **strategy's decision** -- the moment a command was chosen,
not the phase zero-crossing where the arm picked it up (up to one gait period
later). That definition is used everywhere, including by the recorder in
simulation.py, for one reason: the decision is a pure function of the recorded
positions, so a trial that predates the log can be reconstructed on the same
footing as one that carries it, and the two are comparable. Recording the
execution instead would make old and new trials incommensurable.

Source, in order of preference:
  <trial>_gait_log.csv          written by runs from 2026-09 onwards
  <trial>_gait_log_rebuilt.csv  replayed from POS_ALL + config_snapshot.json,
                                cached on first use -- see reconstruct_gait_log
                                for how faithful that is (98% of robot-ticks;
                                the first ~second of a rebuild is a cold start)
A rebuilt trial is marked "rebuilt from POS_ALL" on the figure. Colours are
assigned to commands once across every experiment in the invocation, so a given
gait reads the same in every figure.

`radial` and `chains` together answer "do groups form preferentially in the
outer ring". They are two different questions and you usually want both --
`--figs where` asks for the pair:

  radial  *where* the groups are. Radius is normalised against **how much arena
          is at that radius**: an annulus at r has area proportional to r, so a
          raw radius histogram makes any distribution look wall-heavy.
  chains  *how they lie there*. Groups in these runs are chains (median aspect
          ratio ~10:1), and a ring of radial spokes has exactly the same radial
          density as a ring of tangential chains -- density alone cannot tell
          them apart. This measures the chain's local direction against the
          local tangent, and compares it against the same chains spun to random
          orientations that still fit inside the arena. That null is what makes
          runs at different N comparable at all: INNER_R scales with population
          while body length does not, so at N=17 a 3-robot chain cannot be
          tangential near the wall and reads as "radial" for purely geometric
          reasons. See the section comment above chain_profile.

Both re-read POS_ALL (the group cache keeps member ids but not positions), so
they are the slowest figures here; per-trial results are cached as
<trial>_radial_b<bins>_m<minsize>.csv and <trial>_chains_m<minsize>_b<bins>.csv,
and re-plots are instant. `chains` subsamples frames (--chain-frames, default
400 per trial) because it re-measures every chain --chain-rotations times to
build the null.

--radial-compare PATH and --chain-compare PATH each write one figure overlaying
every experiment, which is the form to use for "groups form along the outer
ring in these conditions but not those".

The time axis's t=0 is **the first recorded frame**, not the simulation's
t=0: RECORD_AFTER_WARMUP skips the WARMUP_STEPS warm-up period, so a 12s
trial has only about 11s of data on disk. This way t=0 lines up to the same
physical moment (right after warm-up) across trials, so cross-trial averaging
is aligned correctly.

Before averaging across trials, series are interpolated onto a common time
axis (the length of the shortest trial; extrapolating past that is meaningless
since the shortest trial simply has no data there). A duration mismatch is
reported in the log along with the truncated length.

Caching, in two layers
======================
    trial_XXXX_groups_d<max_dist>.csv      the Voronoi grouping, one row per
        group per frame (columns match pos_all_grouping's
        process_pos_all_groups exactly)
    trial_XXXX_series_d<max_dist>_b<bins>.npz   the per-trial time series that
        every figure is actually drawn from

The second layer is what makes changing your mind cheap. Turning the group
table into the time series means walking ~1.1M rows into Python objects, which
cost 58s per trial and ran on **every** invocation whatever --figs said -- so
asking for `groupsize` and then realising you wanted `mean_groupsize` paid it
twice, and `radial`, which never looks at those objects at all, paid it too.
With the series cached, the second run is ~3s. Only the kymograph and snapshots
need the intermediate itself, so only they rebuild it.

--force recomputes both layers. The keys carry everything the contents depend
on -- max_dist, --no-singletons, and the size-bin edges -- so caches for
different settings never overwrite or get mistaken for each other.

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
import hashlib
import json
import math
import os
import pickle
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

# Keep text as text in the vector formats instead of outlining it into curves.
# This is the setting that makes "open it and change the font" actually work:
# with the default svg.fonttype="path" every label becomes a set of bezier
# outlines, which an editor can recolour but can never re-typeset or spell-check.
# fonttype 42 is TrueType, which Illustrator and Inkscape both treat as live text.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42

# Fonts an SVG is likely to find again wherever it is opened, most portable
# first. DejaVu Sans is matplotlib's own default and is last here on purpose:
# it renders fine but almost nothing outside matplotlib has it installed.
FONT_PREFERENCE = ("Arial", "Helvetica", "Liberation Sans", "Segoe UI",
                   "Verdana", "Tahoma", "DejaVu Sans")


def choose_font(name=None):
    """
    Pick one installed sans-serif and pin matplotlib to it alone. -> its name.

    This exists because of how SVG font fallback is written versus how a vector
    editor reads it. matplotlib's default font.sans-serif is a ten-name CSS
    fallback chain, and with svg.fonttype="none" the whole chain is written into
    every text element:

        font-family: 'DejaVu Sans', 'Bitstream Vera Sans',
                     'Computer Modern Sans Serif', ... , 'Avant Garde'

    A browser walks that list and takes the first it has. Illustrator does not:
    it tries to resolve **every** name and raises "An unknown problem occurred"
    for each one missing, so opening the file means dismissing six dialogs.
    Pinning the list to a single installed name emits `font-family: 'Arial',
    sans-serif` and the file opens clean.

    The font must be one that is really installed, not merely named: matplotlib
    lays the text out using the metrics of the font it actually loads, and if it
    silently fell back to DejaVu while the file declared Arial, every label
    would be positioned for the wrong font and drift once the editor re-rendered
    it. So this checks the installed set rather than trusting findfont, which
    substitutes a near match instead of failing.
    """
    from matplotlib import font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for cand in ([name] if name else list(FONT_PREFERENCE)):
        if cand in have:
            plt.rcParams["font.family"] = "sans-serif"
            plt.rcParams["font.sans-serif"] = [cand]
            return cand
    if name:
        print(f"[note] font {name!r} is not installed; leaving matplotlib's "
              f"default. Installed sans-serif options include: "
              f"{', '.join(sorted(n for n in FONT_PREFERENCE if n in have)) or '(none of the usual)'}")
    return None

# Set once from --formats in main(). A module-level default rather than a
# parameter threaded through all ten figure functions: every one of them ends in
# the same save_fig(fig, path) call and none of them has any other reason to
# know about output formats.
OUTPUT_FORMATS = ["png"]
FIG_FORMATS = ("png", "svg", "pdf", "eps", "pickle")


def written_name(path):
    """
    How a figure's filename should be reported, given OUTPUT_FORMATS.

    The figure functions all build a .png path and save_fig swaps the extension
    per format, so printing the path verbatim would announce a .png that
    --formats svg never wrote.
    """
    base = os.path.splitext(os.path.basename(path))[0]
    exts = [".fig.pickle" if f == "pickle" else "." + f for f in OUTPUT_FORMATS]
    return base + (exts[0] if len(exts) == 1 else "{" + ",".join(exts) + "}")


def save_fig(fig, path, dpi=150):
    """
    Write one figure in every format in OUTPUT_FORMATS.

    `path` is the .png name the callers build; the extension is replaced per
    format, so a figure requested as png+svg lands as <name>.png and <name>.svg
    side by side.

    "pickle" writes <name>.fig.pickle, matplotlib's equivalent of a MATLAB .fig:
    the live Figure object, reopenable with --open and then editable through the
    plot window or from code. It is the only format that preserves the figure as
    something you can keep *plotting into* rather than just annotating, and the
    only one that is version-fragile -- a pickle written here loads reliably only
    under a similar matplotlib, so it is a working format, not an archival one.
    Vector output (svg/pdf) is the one to keep.
    """
    base = os.path.splitext(path)[0]
    for fmt in OUTPUT_FORMATS:
        if fmt == "pickle":
            try:
                with open(base + ".fig.pickle", "wb") as fh:
                    pickle.dump(fig, fh)
            except Exception as e:
                # Some artists don't pickle; that must not cost the real figure
                print(f"      [note] could not pickle {os.path.basename(base)}: "
                      f"{type(e).__name__}: {e}")
        else:
            fig.savefig(f"{base}.{fmt}", dpi=dpi)

FIG_TRIAL = ("composition", "groupsize", "kymograph", "counts", "snapshots",
             "patterns")
FIG_EXP = ("mean_composition", "mean_groupsize", "mean_counts",
           "mean_patterns", "radial", "chains")
# Neither of the two opt-in figures is in "all": patterns needs a gait log that
# only runs from 2026-09 onwards have, and radial re-reads POS_ALL, which is the
# slowest thing this script can do. Ask for them by name, or use the aliases.
FIG_DEFAULT = tuple(f for f in FIG_TRIAL + FIG_EXP
                    if f not in ("patterns", "mean_patterns", "radial",
                                 "chains"))
FIG_ALIASES = {
    "all": FIG_DEFAULT,
    "trial": tuple(f for f in FIG_TRIAL if f != "patterns"),
    "exp": tuple(f for f in FIG_EXP
                 if f not in ("mean_patterns", "radial", "chains")),
    "aggregation": ("composition", "groupsize"),   # the two halves of the original combined figure
    "everything": FIG_TRIAL + FIG_EXP,
    "pattern": ("patterns", "mean_patterns"),      # both tiers of the gait-share figure
    "where": ("radial", "chains"),                 # where groups sit, and how they lie
}

SIZE_COLORS = ["#d9d9d9", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
C_LARGEST, C_MEANSZ, C_ALIGN = "#08306b", "#e6550d", "#31a354"

# Categorical slots, assigned in this fixed order and never cycled: one per gait
# pattern in the patterns figure, one per experiment in the radial figure. A 9th
# entity folds into "other" (C_OTHER) rather than getting a generated hue.
# This is the dataviz reference palette's documented order, which clears the
# adjacent-pair CVD and normal-vision floors in both light and dark -- the
# pairlist that applies to stacked areas and lines. Three of its light-mode
# slots sit below 3:1 contrast on white, so both figures carry direct labels
# and write their numbers to CSV, not colour alone.
CAT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
              "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
C_OTHER = "#9e9e9e"
C_REF = "#8c8c8c"          # "uniform" / baseline reference lines

EPILOG = """\
The positional arguments are experiment directories, each containing
trial_XXXX/ subfolders.
--figs options:
    per trial      : composition groupsize kymograph counts snapshots
                     patterns          (opt-in, needs *_gait_log.csv)
    per experiment : mean_composition mean_groupsize mean_counts
                     mean_patterns     (opt-in, needs *_gait_log.csv)
                     radial chains     (opt-in, re-read POS_ALL)
    aliases        : all        = everything except the four opt-in figures
                     everything = literally all of them
                     trial / exp / pattern(=patterns+mean_patterns)
                     where(=radial+chains) / aggregation(=composition+groupsize)

--formats decides what each figure is written as (default png only):
    svg / pdf   vector, and text stays **text** (svg.fonttype="none",
                pdf.fonttype=42), so fonts, sizes and colours can be changed in
                Inkscape or Illustrator without the labels having been outlined
                into un-editable curves. This is the format to keep.
    pickle      <name>.fig.pickle -- matplotlib's answer to a MATLAB .fig: the
                live Figure object. Reopen it with --open for an interactive
                window whose toolbar edits line colours, styles, labels and
                limits in place, or load it in code to keep plotting into it:

                    import pickle, matplotlib.pyplot as plt
                    fig = pickle.load(open("x.fig.pickle", "rb"))
                    ax = fig.get_axes()[0]
                    ax.get_lines()[0].set_color("#c0392b")
                    fig.savefig("x_edited.png", dpi=300)

                It is the only format that survives as something you can still
                *replot*, and the only one tied to the matplotlib version that
                wrote it -- a working format, not an archival one.
Several may be given at once; every figure is then written in each.

Examples:
    # gait-pattern share over time, both tiers
    python batch_group_plots.py datafile/exp_A --figs pattern

    # keep an editable copy of everything alongside the png
    python batch_group_plots.py datafile/exp_A --formats png svg pickle
    python batch_group_plots.py --open datafile/exp_A/exp_A_radial.fig.pickle

    # "do the chains form along the outer ring?" across several conditions
    python batch_group_plots.py "datafile/compare/*" --figs where \\
        --radial-compare datafile/compare/radial.png \\
        --chain-compare  datafile/compare/chains.png
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
        # member_ids must stay text: in a trial where every group is a
        # singleton every cell holds a single number, pandas infers float64,
        # and the ids come back as "1.0" -- which int() then refuses.
        return pd.read_csv(cache, dtype={"member_ids": str}), cache, True

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


# =============================================================================
# Which gait pattern each robot was executing
# =============================================================================

def find_gait_log(folder):
    """The recorded log if the run wrote one, else a reconstruction if one is cached."""
    for pat in ("*_gait_log.csv", "*_gait_log_rebuilt.csv"):
        hits = sorted(glob.glob(os.path.join(folder, pat)))
        if hits:
            return hits[0]
    return None


def read_snapshot(folder):
    """This trial's (or its experiment's) config_snapshot.json, or {}."""
    here = os.path.abspath(folder)
    for d in (here, os.path.dirname(here)):
        path = os.path.join(d, "config_snapshot.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
    return {}


def reconstruct_gait_log(folder, fps, force=False):
    """
    Rebuild the gait timeline for a trial that predates *_gait_log.csv, writing
    it as <prefix>_gait_log_rebuilt.csv in the same schema.

    Close to exact, but not exact. decide() is a pure function of the positions
    in POS_ALL, which is why the recorder logs the **decision** rather than the
    zero-crossing where the arm picked it up -- the decision is the part that
    can be recomputed. Measured against a trial that carries both (20 robots,
    13 s, hysteresis of 2 ticks): **98.0% of robot-ticks agree, and the plotted
    shares differ by 0.7 percentage points.** Two sources of residual:

      cold start -- the replay begins at the first *recorded* frame, but the
          live strategy had already been running through the warm-up, so its
          hysteresis counters were warm. Everything confirmed during warm-up
          reads as unconfirmed for the first max(group_n_ticks, n_frames_join)
          ticks of a rebuild. Those frames are not on disk, so this cannot be
          recovered -- it was 8 of 20 robots on the very first tick here.
      tick phase -- live polls land on frame boundaries and self-gate on
          `t >= _next_t`, so the real tick times drift off an exact `period`
          grid, while the replay takes every round(period*fps)-th frame. Where
          a robot sits right on a hysteresis threshold the two can disagree for
          a tick or two. Scattered and unbiased, not systematic.

    So a rebuilt trial is sound for shares and trends, and its first second or
    so should not be read closely.

    Two cases, both driven by config_snapshot.json:

      runtime gait control ON  -> replay STRATEGY_SPEC over POS_ALL. The
          hysteresis in both layers is path-dependent, so the replay must start
          at the first frame and run the whole trial; there is no seeking into
          the middle.
      OFF (or no STRATEGY_SPEC) -> every robot held its COMMAND_ARRAY entry for
          the whole trial, so the timeline is one row per robot at t=0. Nothing
          is inferred; this is just reading the snapshot.

    Returns the path, or None when the snapshot does not say enough.
    """
    pos_all = find_pos_all(folder)
    if not pos_all:
        return None
    out = pos_all.replace("_POS_ALL.csv", "_gait_log_rebuilt.csv")
    if os.path.isfile(out) and not force:
        return out

    snap = read_snapshot(folder)
    cmds = snap.get("COMMAND_ARRAY")
    spec = snap.get("STRATEGY_SPEC")
    runtime_on = bool(snap.get("ENABLE_RUNTIME_GAIT_CONTROL"))
    ctrl = str(snap.get("RUNTIME_GAIT_CONTROLLER") or "")

    if not cmds:
        return None

    rows = [(0.0, i, int(c)) for i, c in enumerate(cmds)]

    if runtime_on and spec:
        if not ctrl.endswith(":strategy"):
            # A hand-written callback can depend on anything -- velocities, its
            # own ctx, a random draw -- so replaying positions would not
            # reproduce it. Better to emit the initial assignment alone than a
            # confident-looking guess.
            print(f"      [note] {os.path.basename(folder)}: controller is "
                  f"{ctrl!r}, not the declarative strategy; only the initial "
                  f"COMMAND_ARRAY can be rebuilt")
        else:
            from strategy import LayeredStrategy, frame_record
            spec = dict(spec)
            if "skip_claimed" not in spec:
                # The snapshot was written before skip_claimed existed, so the
                # run it describes used the old merge, where a role selected
                # robots the group layer then overwrote. LayeredStrategy now
                # defaults the flag to True, so replaying without pinning it
                # would faithfully rebuild a run that never happened.
                spec["skip_claimed"] = False
                print(f"      [note] {os.path.basename(folder)}: snapshot "
                      f"predates skip_claimed; replaying with it off, as the "
                      f"run did")
            df = pd.read_csv(pos_all)
            df.columns = df.columns.str.strip()
            df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
            step = df["Step"].to_numpy()
            xy = df[["X", "Y"]].to_numpy(dtype=float)
            th = df["Theta"].to_numpy(dtype=float)
            aid = df["Agent_ID"].to_numpy()
            steps, starts = np.unique(step, return_index=True)
            bounds = np.append(starts, len(step))

            strat = LayeredStrategy(**spec)
            every = max(1, int(round(float(spec.get("period", 0.25)) * fps)))
            for k in range(0, len(steps), every):
                sl = slice(bounds[k], bounds[k + 1])
                # POS_ALL's Agent_ID is 1-based; robot ids are 0-based
                rec = frame_record(aid[sl] - 1, xy[sl], th[sl], strat.max_dist,
                                   include_singletons=strat.include_singletons)
                strat.decide(float(steps[k]) / fps, rec, rec["ids"])
            rows.extend(strat.decisions)

    # Same collapse the recorder applies: the COMMAND_ARRAY entry and the
    # strategy's first decision both sit at t=0, and only the later one is
    # visible to a forward-fill, so keep just that. Then drop no-ops.
    rows.sort(key=lambda r: (r[0], r[1]))
    pre, post = {}, []
    for t, rid, cmd in rows:
        if t > 0.0:
            post.append((t, rid, cmd))
        else:
            pre[rid] = cmd
    rows = sorted((0.0, rid, cmd) for rid, cmd in pre.items()) + post

    deduped, last = [], {}
    for t, rid, cmd in rows:
        if last.get(rid) != cmd:
            last[rid] = cmd
            deduped.append((t, rid, cmd))

    with open(out, "w", newline="", encoding="utf-8") as f:
        f.write("time,robot_id,command,source\r\n")
        f.write("".join(f"{t:.6f},{rid},{cmd},rebuilt\r\n"
                        for t, rid, cmd in deduped))
    return out


def describe_command(cmd):
    """
    "-52" -> "-52  75deg 1Hz ph-rnd anti" -- the signed XYZ integer decoded into
    what it actually does, so the legend is readable without the encoding table.
    Returns the bare number if the digits fall outside the lookup tables.

    The X digit (initial phase) is part of the label even though it rarely
    matters, because without it 832 and 32 both read "45deg 1Hz" and the legend
    grows two rows that look identical. X=0 means a random initial phase.
    """
    try:
        from gait import AMPLI_TABLE, FREQ_TABLE
        c = int(cmd)
        a = abs(c)
        z, y, x = a % 10, (a % 100) // 10, a // 100
        if not (1 <= z <= len(FREQ_TABLE) and 1 <= y <= len(AMPLI_TABLE)):
            return str(c)
        deg = round(np.degrees(AMPLI_TABLE[y - 1]))
        return (f"{c:+d}  {deg:g}° {FREQ_TABLE[z - 1]:g}Hz "
                + (f"ph{x}" if x else "ph-rnd")
                + (" anti" if c < 0 else ""))
    except Exception:
        return str(cmd)


def gait_commands(folder):
    """The distinct commands appearing in one trial's gait log, with total robot-time each."""
    path = find_gait_log(folder)
    if not path:
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "command" not in df.columns:
        return {}
    return {int(c): int(n) for c, n in df["command"].value_counts().items()}


def pattern_shares(gait_csv, secs, order):
    """
    Share of robots executing each command, sampled at every time in secs.
    -> (T, len(order)+1) array; the last column is "other" (commands not in
    order), which is all-zero and dropped by the caller when nothing lands there.

    The log is an **event list** -- one row per robot at t=0, then one row per
    switch -- so the state at any moment is a forward-fill per robot. Rather
    than materialising a robots x frames table, this walks the events once in
    time order and keeps a running count per command: each event moves one
    robot from one bucket to another, and each output time just snapshots the
    counts. O(events + T*k) instead of O(frames*robots).

    The recorded time is when a command **took effect** (its zero-crossing),
    which is up to one gait period after the strategy asked for it -- the share
    plotted here is what the robots were doing, not what they were told.
    """
    df = pd.read_csv(gait_csv)
    df = df.sort_values("time", kind="mergesort")
    ev_t = df["time"].to_numpy(dtype=float)
    ev_r = df["robot_id"].to_numpy()
    ev_c = df["command"].to_numpy(dtype=int)

    slot = {int(c): i for i, c in enumerate(order)}
    other = len(order)
    counts = np.zeros(other + 1, dtype=int)
    cur = {}                                     # robot_id -> slot index

    out = np.zeros((len(secs), other + 1), dtype=float)
    j = 0
    for i, t in enumerate(secs):
        while j < len(ev_t) and ev_t[j] <= t:
            s = slot.get(int(ev_c[j]), other)
            prev = cur.get(ev_r[j])
            if prev is not None:
                counts[prev] -= 1
            counts[s] += 1
            cur[ev_r[j]] = s
            j += 1
        total = counts.sum()
        if total:
            out[i] = counts / total
    return out


def fig_patterns(secs, share, order, has_other, path, title="",
                 smooth_s=2.0, note=None, rebuilt=False):
    """Stacked area: share of robots assigned each gait pattern over time."""
    win = smooth_window(secs, smooth_s)
    cols = [roll_mean(share[:, b], win) for b in range(share.shape[1])]
    labels = [describe_command(c) for c in order]
    colors = [CAT_COLORS[i % len(CAT_COLORS)] for i in range(len(order))]
    if has_other:
        labels.append("other")
        colors.append(C_OTHER)
    else:
        cols = cols[:len(order)]

    fig, ax = plt.subplots(figsize=(11, 4.8))
    # 2px surface gap between stacked bands so adjacent fills stay separable
    # even where two of them are nearly the same lightness
    stacked_bands(ax, secs, np.array(cols) * 100, colors, labels,
                  edgecolor="white", linewidth=1.0)
    ax.set_ylim(0, 100)
    ax.set_xlim(secs[0], secs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("share of robots (%)")
    ax.set_title(f"Gait pattern share{title}")
    ax.legend(title="command", loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=min(4, len(labels)), frameon=False, fontsize=8)

    # Direct labels: three of the categorical slots fall below 3:1 on white, so
    # identity must not rest on colour alone. Only bands thick enough to hold
    # text get one -- a number on every band would be noise. The ink is a text
    # token picked by the fill's luminance (dark fills take the light one), not
    # the series colour.
    mids = np.cumsum(np.array(cols) * 100, axis=0) - np.array(cols) * 50
    for b, lab in enumerate(labels):
        m = np.asarray(cols[b]).mean() * 100
        if m < 7:
            continue
        rgb = matplotlib.colors.to_rgb(colors[b])
        lum = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        k = len(secs) // 2
        ax.text(secs[k], mids[b][k], f"{lab.split()[0]}\n{m:.0f}%",
                ha="center", va="center", fontsize=8,
                color="#1a1a1a" if lum > 0.55 else "#ffffff",
                linespacing=1.15)

    if note is None:
        note = (f"({smooth_s:g}s moving average)" if win > 1
                else "(per-frame share, no smoothing)")
    if rebuilt:
        note = "rebuilt from POS_ALL  " + note
    # Above the axes, not inside: the stack always fills 0-100%, so anything
    # placed in the plot lands on top of a coloured band
    ax.text(1.0, 1.015, note, transform=ax.transAxes, ha="right",
            fontsize=8, color="#555")
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


def load_frames(groups_df, fps, with_members=True):
    """
    -> [(step, seconds, [ {members, size, alignment, centroid}, ... ]), ...]

    Columns are pulled out as numpy arrays once and indexed positionally, rather
    than walking the table with groupby().iterrows(). The group table of a 500s
    trial is ~1.1M rows, and iterrows() builds a full pandas Series per row:
    measured at 31.6s of the 58s this function used to take, with
    Series.__init__ called 1.1M times. Same values, same order.

    Order within a frame is preserved -- the sort is stable and the cache is
    already written in step order. It matters: track_groups breaks ties in its
    greedy matching by iteration order, so a reshuffle would repaint kymograph
    colours.

    with_members=False skips parsing the member id strings into frozensets,
    which is the other 24s. Only track_groups (and so the kymograph and
    snapshots figures) looks at them; every other figure reads size / alignment
    / centroid, so the common case should not pay for 1.1M frozensets.
    """
    df = groups_df.sort_values("Step", kind="mergesort")
    steps = df["Step"].to_numpy()
    size = df["size"].to_numpy()
    align = df["alignment"].to_numpy(dtype=float)
    cx = df["centroid_x"].to_numpy(dtype=float)
    cy = df["centroid_y"].to_numpy(dtype=float)
    mem = df["member_ids"].astype(str).to_numpy() if with_members else None

    uniq, starts = np.unique(steps, return_index=True)
    bounds = np.append(starts, len(steps))

    frames = []
    for k, step in enumerate(uniq):
        lo, hi = int(bounds[k]), int(bounds[k + 1])
        frames.append((int(step), float(step) / fps, [{
            "members": (frozenset(int(x) for x in mem[i].split())
                        if with_members else None),
            "size": int(size[i]),
            "alignment": float(align[i]),
            "centroid": (float(cx[i]), float(cy[i])),
        } for i in range(lo, hi)]))
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


def series_cache_path(pos_all_csv, max_dist, include_singletons, bins):
    """
    Where this trial's trial_series() output is cached.

    The bin edges go into the key because they decide the `frac` columns, and
    they are chosen from the largest population across the whole experiment --
    so the same trial plotted as part of a different set of trials legitimately
    gets different bins and must not read back the other set's cache.
    """
    tag = f"{max_dist:g}".replace(".", "p")
    sig = hashlib.md5(repr(list(bins)).encode()).hexdigest()[:8]
    name = f"_series_d{tag}_b{sig}.npz"
    if not include_singletons:
        name = name.replace(".npz", "_nosing.npz")
    return pos_all_csv.replace("_POS_ALL.csv", name)


_SERIES_KEYS = ("secs", "largest", "mean_size", "alignment",
                "n_groups", "n_multi", "n_seen", "largest_frac")


def read_series_cache(path):
    """The cached series dict, or None. npz rather than csv: these feed every
    figure and the averaging, so the round-trip has to be bit-exact, and a text
    format would make that a question about float formatting."""
    if not os.path.isfile(path):
        return None
    try:
        with np.load(path) as z:
            sr = {k: z[k] for k in _SERIES_KEYS}
            sr["frac"] = z["frac"]
        return sr
    except Exception:
        return None            # truncated or written by an older layout


def write_series_cache(path, sr):
    try:
        np.savez(path, frac=sr["frac"], **{k: sr[k] for k in _SERIES_KEYS})
    except Exception as e:
        print(f"      [note] could not cache the series: {type(e).__name__}: {e}")


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

def stacked_bands(ax, x, rows, colors, labels, edgecolor="white",
                  linewidth=1.0, max_points=3000):
    """
    Stacked area chart, one closed polygon per band. -> the patch handles.

    Deliberately **not** ax.stackplot, which builds a PolyCollection. The SVG
    backend writes a collection as

        <defs><path id="m2b46..." d="M 47.47 -225.51 ..."/></defs>
        <g clip-path="..."><use xlink:href="#m2b46..." x="0" y="345.6"
                                style="fill: #2a78d6"/></g>

    -- geometry parked in <defs> with negative coordinates, instantiated by a
    <use> that supplies both the offset that brings it on-canvas and the fill.
    Illustrator does not resolve that indirection, so the bands import as
    nothing at all while the axes, labels, legend swatches and tick marks (all
    written inline) come through fine: a figure that looks empty but is not.

    Filling each band separately emits an ordinary inline <path> carrying its
    own fill and real coordinates, which every editor reads. Raster output is
    unchanged.

    max_points caps how many samples a band is drawn from. The cross-trial
    figures interpolate onto a grid as dense as the recording itself -- ~30000
    points for a 500s trial -- which at 11in x 150dpi is about 18 points per
    pixel. Nothing on screen changes, but each band becomes a 60000-vertex
    polygon and the file reaches 4.4MB, which is slow to open and miserable to
    edit. Subsampling to ~2 points per pixel is lossless at this figure size,
    and the series are a moving average already, so there are no single-sample
    spikes to lose. All bands are subsampled on the same indices, or the stack
    would not line up.
    """
    x = np.asarray(x, dtype=float)
    rows = [np.asarray(r, dtype=float) for r in rows]
    if max_points and len(x) > max_points:
        keep = np.unique(np.linspace(0, len(x) - 1, int(max_points)).astype(int))
        x = x[keep]
        rows = [r[keep] for r in rows]
    xs = np.concatenate([x, x[::-1]])
    lo = np.zeros_like(x)
    handles = []
    for row, colour, lab in zip(rows, colors, labels):
        hi = lo + np.asarray(row, dtype=float)
        poly, = ax.fill(xs, np.concatenate([hi, lo[::-1]]), facecolor=colour,
                        edgecolor=edgecolor, linewidth=linewidth, label=lab,
                        closed=True)
        handles.append(poly)
        lo = hi
    return handles


def fig_composition(sr, labels, path, title="", smooth_s=2.0, note=None):
    secs = sr["secs"]
    # Per-frame shares are very jittery (a single Voronoi edge in one frame can
    # merge and split two groups), so plotting them raw would be drowned in
    # noise. A ~2s moving average is used so the trend is actually visible.
    win = smooth_window(secs, smooth_s)
    frac_s = np.column_stack([roll_mean(sr["frac"][:, b], win)
                              for b in range(sr["frac"].shape[1])])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    stacked_bands(ax, secs, frac_s.T * 100, SIZE_COLORS[:len(labels)],
                  labels, edgecolor="none", linewidth=0.0)
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
    save_fig(fig, path)
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
    save_fig(fig, path)
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
    save_fig(fig, path)
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
    save_fig(fig, path)
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
    save_fig(fig, path)
    plt.close(fig)


# =============================================================================
# Figure 5: where in the arena do groups form?
#
# The question "do groups prefer the outer ring" cannot be answered by a
# histogram of radius: an annulus at r has area proportional to r, so the outer
# ring holds more robots than the middle even when robots are spread perfectly
# uniformly. Every quantity here is therefore normalised against how much arena
# is actually at that radius, or expressed as a conditional probability that
# divides that bias out on both sides.
#
# Two panels:
#   (a) P(robot is in a group | its radius) -- the claim itself. Rising toward
#       r/R = 1 means a robot near the wall is more likely to be grouped than
#       one in the middle. Each experiment's overall grouped fraction is drawn
#       as a faint dotted line in the same colour: the curve sitting above its
#       own dotted line at large r *is* "groups form preferentially outside".
#   (b) radial density of **all** robots, area-normalised (1.0 = uniform) --
#       the confound check. If robots simply pile up against the wall, (b)
#       shows it, and (a) is what tells you grouping is more than that.
# =============================================================================

def ring_geometry(folder, fallback_n_sides=35):
    """
    (cx, cy, R, shape, n_sides, source) from this experiment's
    config_snapshot.json, falling back to the current config.py. Returns None if
    neither is readable -- the radial figure is skipped rather than drawn
    against a guessed arena.

    The config.py fallback is reported in `source` and flagged by the caller,
    because it is only right by luck: config.py holds whatever the *last* edit
    left behind, and an arena of the wrong radius silently rescales r/R and
    drops every robot outside it. radial_profile cross-checks the geometry
    against the positions for exactly this reason.
    """
    src, source = {}, None
    here = os.path.abspath(folder)
    for d in (here, os.path.dirname(here)):
        path = os.path.join(d, "config_snapshot.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    src = json.load(f)
                source = os.path.relpath(path, here)
                break
            except Exception:
                pass
    if not src:
        try:
            import config as _c
            src = {k: getattr(_c, k) for k in
                   ("W", "H", "INNER_R", "RING_SHAPE", "RING_N_SIDES")
                   if hasattr(_c, k)}
            source = "config.py"
        except Exception:
            return None
    try:
        w, h, r = float(src["W"]), float(src["H"]), float(src["INNER_R"])
    except (KeyError, TypeError, ValueError):
        return None
    shape = str(src.get("RING_SHAPE", "circle")).lower()
    n_sides = int(src.get("RING_N_SIDES") or fallback_n_sides)
    return w / 2.0, h / 2.0, r, shape, n_sides, source


def ring_area_fraction(edges, R, shape="circle", n_sides=35, n_quad=4000):
    """
    Fraction of the arena's area falling in each annulus [edges[i], edges[i+1]].

    For a circle that is just (r2^2 - r1^2) / R^2. For the regular polygon ring
    the arena stops short of R between vertices, so the outermost annuli are
    only partly inside. At radius r the fraction of the full circle that lies
    inside a regular n-gon of circumradius R is

        f(r) = 1                                  r <= a
             = max(0, 1 - n*arccos(a/r)/pi)       r >  a,     a = R*cos(pi/n)

    (inside one of the n sectors, the point at angular offset psi is inside iff
    r*cos(psi) <= a). Area out to r is then the integral of 2*pi*s*f(s), done
    here on a fine grid -- exact enough that the normalisation is never the
    thing limiting this plot.
    """
    edges = np.asarray(edges, dtype=float)
    s = np.linspace(0.0, R, int(n_quad))
    if shape.startswith("poly") and n_sides >= 3:
        a = R * math.cos(math.pi / n_sides)
        with np.errstate(invalid="ignore"):
            f = np.where(s <= a, 1.0,
                         1.0 - n_sides * np.arccos(np.clip(a / np.maximum(s, 1e-12),
                                                           -1.0, 1.0)) / math.pi)
        f = np.clip(f, 0.0, 1.0)
    else:
        f = np.ones_like(s)
    cum = np.concatenate([[0.0], np.cumsum(np.diff(s) * 2 * math.pi
                                           * 0.5 * (s[1:] * f[1:] + s[:-1] * f[:-1]))])
    A = np.interp(edges, s, cum)
    total = cum[-1]
    return np.diff(A) / total if total > 0 else np.full(len(edges) - 1, np.nan)


def radial_profile(pos_all_csv, groups_df, geom, nbins=20, min_size=3,
                   force=False):
    """
    Per-trial radial histograms, cached as <prefix>_radial_d..._b..._m....csv.

    -> {"edges", "r_mid", "n_all", "n_grouped", "area_frac"} where the counts
    are **robot-frame observations** (one per robot per frame), not robots.

    Re-reads POS_ALL because the group cache stores member ids but not
    positions; the result is small, so the cache means only the first plot of a
    trial pays for it.
    """
    cx, cy, R, shape, n_sides = geom[:5]
    tag = f"b{nbins}_m{min_size}"
    cache = pos_all_csv.replace("_POS_ALL.csv", f"_radial_{tag}.csv")
    if os.path.isfile(cache) and not force:
        c = pd.read_csv(cache)
        # size_sum arrived later than the rest; a cache without it predates the
        # mean-group-size panel and has to be rebuilt rather than half-read
        if "size_sum" in c.columns:
            return {"edges": np.append(c["r_lo"].to_numpy(), c["r_hi"].iloc[-1]),
                    "r_mid": c["r_mid"].to_numpy(),
                    "n_all": c["n_all"].to_numpy(dtype=float),
                    "n_grouped": c["n_grouped"].to_numpy(dtype=float),
                    "size_sum": c["size_sum"].to_numpy(dtype=float),
                    "area_frac": c["area_frac"].to_numpy(dtype=float)}

    df = pd.read_csv(pos_all_csv, usecols=["Step", "Agent_ID", "X", "Y"])
    df.columns = df.columns.str.strip()
    r = np.hypot(df["X"].to_numpy(dtype=float) - cx,
                 df["Y"].to_numpy(dtype=float) - cy)

    # Size of the group each (frame, robot) observation belongs to; 1 for a
    # robot in none (which is every robot when the table was built with
    # --no-singletons). Encoding the pair as one integer key turns the lookup
    # into a sort + searchsorted instead of a few hundred thousand Python
    # tuples.
    size_of = np.ones(len(df), dtype=np.int64)
    if len(groups_df):
        memb = groups_df["member_ids"].astype(str).str.split()
        lens = memb.str.len().to_numpy()
        flat = np.fromiter((int(x) for lst in memb for x in lst),
                           dtype=np.int64, count=int(lens.sum()))
        fsteps = np.repeat(groups_df["Step"].to_numpy(dtype=np.int64), lens)
        fsizes = np.repeat(groups_df["size"].to_numpy(dtype=np.int64), lens)
        stride = int(max(df["Agent_ID"].max(), flat.max() if len(flat) else 0)) + 1
        gkey = fsteps * stride + flat
        mine = (df["Step"].to_numpy(dtype=np.int64) * stride
                + df["Agent_ID"].to_numpy(dtype=np.int64))
        order = np.argsort(gkey, kind="mergesort")
        gkey, fsizes = gkey[order], fsizes[order]
        at = np.clip(np.searchsorted(gkey, mine), 0, max(len(gkey) - 1, 0))
        hit = gkey[at] == mine
        size_of[hit] = fsizes[at[hit]]
    grouped = size_of >= min_size

    # Positions outside the arena mean the geometry doesn't belong to this data
    # (almost always: no config_snapshot.json, so it fell back to config.py).
    # np.histogram would just drop them and the figure would look fine, so say
    # so instead.
    outside = float(np.mean(r > R))
    if outside > 0.01:
        print(f"      [WARN] {outside*100:.0f}% of positions lie outside "
              f"R={R:g} around ({cx:.0f}, {cy:.0f}) -- that arena is probably "
              f"not this trial's. Observed max radius {r.max():.0f}. "
              f"The radial figure ignores everything past R.")

    edges = np.linspace(0.0, R, int(nbins) + 1)
    n_all, _ = np.histogram(r, bins=edges)
    n_grp, _ = np.histogram(r[grouped], bins=edges)
    size_sum, _ = np.histogram(r, bins=edges, weights=size_of.astype(float))
    area = ring_area_fraction(edges, R, shape, n_sides)
    mid = 0.5 * (edges[:-1] + edges[1:])

    pd.DataFrame({"r_lo": edges[:-1], "r_hi": edges[1:], "r_mid": mid,
                  "n_all": n_all, "n_grouped": n_grp, "size_sum": size_sum,
                  "area_frac": area}).to_csv(cache, index=False)
    return {"edges": edges, "r_mid": mid, "n_all": n_all.astype(float),
            "n_grouped": n_grp.astype(float), "size_sum": size_sum,
            "area_frac": area}


def merge_radial(profiles):
    """Pool several trials' histograms -- counts add, the area weights are shared."""
    return {"edges": profiles[0]["edges"], "r_mid": profiles[0]["r_mid"],
            "area_frac": profiles[0]["area_frac"],
            **{k: sum(p[k] for p in profiles)
               for k in ("n_all", "n_grouped", "size_sum")}}


def fig_radial(profiles, labels, R_list, path, title="", min_size=3):
    """
    profiles  one merged profile per experiment (or per trial), same bin count
    labels    one legend label per profile

    Three panels, most informative first. The top two are **conditional on a
    robot being at radius r**, which is what makes them the right answer to
    "does chaining prefer the outer ring": conditioning divides out both the
    annulus area and however the robots happen to be distributed, so no null
    model or geometric correction is needed -- the comparison is already
    like-for-like at every radius.

        (a) mean size of the group a robot at r belongs to. Threshold-free, so
            it cannot be blunted by a badly chosen cutoff, and it carries
            magnitude: on the 0916 N=100 runs it goes 3.6 in the middle to 9.9
            at r/R ~ 0.8.
        (b) P(the robot at r is in a group of >= min_size). Reads as a
            probability, but note that a low min_size **saturates**: at
            min_size=3 nearly every robot qualifies everywhere and the curve
            flattens (0.54 -> 0.83 on the same data, versus 0.01 -> 0.35 at
            min_size=12). If (b) looks flat while (a) does not, raise
            --radial-min-size.
        (c) where the robots are at all, area-normalised. Not part of the
            claim -- it is the context for reading (a) and (b), e.g. whether an
            outer peak coincides with where most robots actually are.
    """
    fig, (a1, a2, a3) = plt.subplots(3, 1, figsize=(9.5, 9.6), sharex=True)
    n = len(profiles)

    for i, (p, lab, R) in enumerate(zip(profiles, labels, R_list)):
        c = CAT_COLORS[i % len(CAT_COLORS)] if n > 1 else C_LARGEST
        x = p["r_mid"] / R
        nall = p["n_all"]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean_sz = np.where(nall > 0, p["size_sum"] / np.maximum(nall, 1), np.nan)
            p_grp = np.where(nall > 0, p["n_grouped"] / np.maximum(nall, 1), np.nan)
            dens = np.where(p["area_frac"] > 0,
                            (nall / max(nall.sum(), 1)) / p["area_frac"], np.nan)
        ov_sz = p["size_sum"].sum() / max(nall.sum(), 1)
        ov_gp = p["n_grouped"].sum() / max(nall.sum(), 1)

        # Each experiment's own overall level as a dotted line in its colour:
        # the claim is "rises above its own baseline toward the wall", which is
        # then readable directly off the panel instead of by comparing numbers.
        a1.plot(x, mean_sz, lw=2.0, color=c, label=f"{lab}  (mean {ov_sz:.1f})")
        a1.axhline(ov_sz, color=c, lw=1.0, ls=":", alpha=0.55)
        a2.plot(x, p_grp, lw=2.0, color=c, label=f"{lab}  (overall {ov_gp:.2f})")
        a2.axhline(ov_gp, color=c, lw=1.0, ls=":", alpha=0.55)
        a3.plot(x, dens, lw=2.0, color=c, label=lab)

    a1.set_ylabel("mean group size at r")
    a1.set_ylim(bottom=0)
    a1.set_title(f"Where chaining happens{title}\n"
                 f"dotted = each experiment's own overall level; rising above "
                 f"it toward the wall = chaining prefers the outer ring",
                 fontsize=9.5)
    a1.legend(frameon=False, fontsize=8, loc="upper left", ncol=min(2, n))

    a2.set_ylim(0, 1)
    a2.set_ylabel(f"P(in a group of >={min_size})")
    a2.legend(frameon=False, fontsize=8, loc="upper left", ncol=min(2, n))

    a3.axhline(1.0, color=C_REF, lw=1.2, ls="--")
    a3.text(0.012, 1.0, " uniform", color=C_REF, fontsize=8, va="bottom")
    a3.set_ylim(bottom=0)
    a3.set_xlim(0, 1)
    a3.set_ylabel("robot density / uniform")
    a3.set_xlabel("distance from arena centre  r / R")
    a3.set_title("Context: where the robots are at all (area-normalised)",
                 fontsize=9.5)

    for ax in (a1, a2, a3):
        ax.grid(alpha=0.25)
    fig.tight_layout()
    save_fig(fig, path)
    plt.close(fig)


# =============================================================================
# Figure 6: are the chains oriented ALONG the ring?
#
# Groups here are chains, not blobs -- median PCA aspect ratio 10:1 on the
# 0916 runs -- so "groups form in the outer ring" is really two claims, and the
# radial figure only tests the first:
#     (i)  the chains sit near the wall          -> fig_radial
#     (ii) the chains run *along* the wall       -> here
# Density alone misses (ii) entirely: a ring of radial spokes and a ring of
# tangential chains have identical radial densities.
#
# The measure is a nematic order parameter, because a chain has an axis, not a
# direction: with alpha the angle between the local chain direction and the
# local tangent,
#     cos(2*alpha) = +1  chain runs along the ring
#                  =  0  no preference
#                  = -1  chain points straight at the wall
# taken per robot rather than per chain, against the tangent at that robot's
# own position. Per chain would be wrong twice over: a long chain hugging the
# wall has its centroid well inside (it is a chord), and a curved chain has no
# single axis. The local direction is the line through each robot's two nearest
# chain-mates, which follows curvature.
#
# THE NULL IS NOT OPTIONAL. A chain of length l can only lie tangentially at
# radius r if its ends still fit inside: with INNER_R scaling with population
# (245px at N=17 vs 594px at N=100) and body length fixed at ~105px, an N=17
# arena is barely 2.3 body lengths in radius and a 3-robot chain simply cannot
# be tangential out there. Measured on the 0916 runs, N=17 reads <cos2a> =
# -0.25, which looks like a strong radial preference and is almost entirely
# geometry: the null is -0.21, so the excess is -0.04, i.e. nothing.
#
# So every observed chain is also re-measured after being spun about its own
# centroid by random angles, keeping only orientations that still fit inside
# the ring. That holds the chain's shape, length and position fixed and varies
# only what is being tested. observed - null is the part that is preference.
# =============================================================================

def _local_chain_dirs(p):
    """
    Unit direction of the chain at each of its members: the line through that
    member's two nearest chain-mates. Undirected -- only the axis matters, and
    cos(2a) is invariant to the sign.
    """
    d = np.linalg.norm(p[:, None, :] - p[None, :, :], axis=-1)
    np.fill_diagonal(d, np.inf)
    nb = np.argsort(d, axis=1)[:, :2]
    v = p[nb[:, 0]] - p[nb[:, 1]]
    n = np.hypot(v[:, 0], v[:, 1])
    bad = n < 1e-9
    v[bad] = (1.0, 0.0)
    n[bad] = 1.0
    return v / n[:, None]


def _tangential_order(p, v, ctr, R, r_floor):
    """
    -> (r/R, cos(2*alpha)) per member, dropping members too close to the centre
    for a tangent to be defined (there the radial unit vector is pure noise).
    """
    rel = p - ctr
    r = np.hypot(rel[:, 0], rel[:, 1])
    ok = r > r_floor * R
    if not ok.any():
        return None
    u = rel[ok] / r[ok, None]
    tg = np.stack([-u[:, 1], u[:, 0]], axis=1)      # tangent = radial rotated 90 deg
    ct = np.sum(v[ok] * tg, axis=1)
    return r[ok] / R, 2.0 * ct * ct - 1.0


def chain_profile(pos_all_csv, groups_df, geom, nbins=12, min_size=3,
                  n_frames=400, n_rot=12, r_floor=0.15, seed=0, force=False):
    """
    Binned tangential order vs radius, observed and null, for one trial.

    Cached as <prefix>_chains_m<minsize>_b<bins>.csv holding per-bin sums, so
    trials pool by adding rows -- no interpolation, and a longer trial
    contributes proportionally more observations.

    Frames are subsampled (n_frames evenly spaced): the statistic is an average
    over chains, consecutive frames are nearly identical, and the null needs
    n_rot re-measurements of every chain. 400 frames of a 30k-frame trial is
    already tens of thousands of chain observations per experiment.
    """
    cx, cy, R = geom[0], geom[1], geom[2]
    ctr = np.array([cx, cy])
    cache = pos_all_csv.replace("_POS_ALL.csv",
                                f"_chains_m{min_size}_b{nbins}.csv")
    if os.path.isfile(cache) and not force:
        c = pd.read_csv(cache)
        return {k: c[k].to_numpy(dtype=float)
                for k in ("r_mid", "obs_sum", "obs_n", "null_sum", "null_n")}

    df = pd.read_csv(pos_all_csv, usecols=["Step", "Agent_ID", "X", "Y"])
    df.columns = df.columns.str.strip()
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    steps, starts = np.unique(df["Step"].to_numpy(), return_index=True)
    bounds = np.append(starts, len(df))
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    aid = df["Agent_ID"].to_numpy()

    big = groups_df[groups_df["size"] >= min_size]
    by_step = {int(s): grp["member_ids"].astype(str).tolist()
               for s, grp in big.groupby("Step", sort=False)}

    edges = np.linspace(0.0, 1.0, int(nbins) + 1)
    obs_sum = np.zeros(nbins); obs_n = np.zeros(nbins)
    nul_sum = np.zeros(nbins); nul_n = np.zeros(nbins)
    rng = np.random.default_rng(seed)

    pick = np.unique(np.linspace(0, len(steps) - 1,
                                 min(int(n_frames), len(steps))).astype(int))
    for k in pick:
        sl = slice(bounds[k], bounds[k + 1])
        members = by_step.get(int(steps[k]))
        if not members:
            continue
        pos_of = {int(a): i for i, a in enumerate(aid[sl])}
        frame_xy = xy[sl]
        for mem in members:
            idx = [pos_of[int(x)] for x in mem.split() if int(x) in pos_of]
            if len(idx) < min_size:
                continue
            p = frame_xy[idx]
            v = _local_chain_dirs(p)
            got = _tangential_order(p, v, ctr, R, r_floor)
            if got is not None:
                b = np.clip(np.digitize(got[0], edges) - 1, 0, nbins - 1)
                np.add.at(obs_sum, b, got[1])
                np.add.at(obs_n, b, 1.0)

            # Null: same chain, same centroid, random orientation that still fits
            c = p.mean(axis=0)
            off = p - c
            for ang in rng.uniform(0.0, 2.0 * np.pi, int(n_rot)):
                ca, sa = math.cos(ang), math.sin(ang)
                M = np.array([[ca, -sa], [sa, ca]])
                q = c + off @ M.T
                if np.hypot(q[:, 0] - cx, q[:, 1] - cy).max() > R:
                    continue                      # would stick out of the ring
                got = _tangential_order(q, v @ M.T, ctr, R, r_floor)
                if got is not None:
                    b = np.clip(np.digitize(got[0], edges) - 1, 0, nbins - 1)
                    np.add.at(nul_sum, b, got[1])
                    np.add.at(nul_n, b, 1.0)

    mid = 0.5 * (edges[:-1] + edges[1:])
    pd.DataFrame({"r_mid": mid, "obs_sum": obs_sum, "obs_n": obs_n,
                  "null_sum": nul_sum, "null_n": nul_n}).to_csv(cache, index=False)
    return {"r_mid": mid, "obs_sum": obs_sum, "obs_n": obs_n,
            "null_sum": nul_sum, "null_n": nul_n}


def merge_chains(profiles):
    """Pool trials: the cached values are sums and counts, so they add."""
    return {"r_mid": profiles[0]["r_mid"],
            **{k: sum(p[k] for p in profiles)
               for k in ("obs_sum", "obs_n", "null_sum", "null_n")}}


def fig_chains(profiles, labels, path, title="", min_size=3, min_n=200):
    """
    profiles  one merged chain profile per experiment
    min_n     bins with fewer observations than this are left out rather than
              drawn as a spike -- the outermost bin is always thin
    """
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    n = len(profiles)

    for i, (p, lab) in enumerate(zip(profiles, labels)):
        c = CAT_COLORS[i % len(CAT_COLORS)] if n > 1 else C_LARGEST
        x = p["r_mid"]
        ok_o = p["obs_n"] >= min_n
        ok_n = p["null_n"] >= min_n
        obs = np.where(ok_o, p["obs_sum"] / np.maximum(p["obs_n"], 1), np.nan)
        nul = np.where(ok_n, p["null_sum"] / np.maximum(p["null_n"], 1), np.nan)
        exc = np.where(ok_o & ok_n, obs - nul, np.nan)
        tot = (p["obs_sum"].sum() / max(p["obs_n"].sum(), 1)
               - p["null_sum"].sum() / max(p["null_n"].sum(), 1))

        a1.plot(x, obs, lw=2.0, color=c, label=lab)
        a1.plot(x, nul, lw=1.2, color=c, ls="--", alpha=0.65)
        # The legend carries name + overall figure next to a colour swatch, and
        # the per-trial *_chains_*.csv caches are the table view. No end label:
        # it would sit at the last bin while quoting the overall number, which
        # reads as if the curve ended there.
        a2.plot(x, exc, lw=2.2, color=c, label=f"{lab}  ({tot:+.2f} overall)")

    for ax in (a1, a2):
        ax.axhline(0.0, color=C_REF, lw=1.1, ls=":")
        ax.grid(alpha=0.25)
        ax.set_ylim(-1, 1)
    a1.set_ylabel("tangential order  <cos 2α>")
    a1.set_title(f"Do the chains run along the ring?{title}\n"
                 f"+1 = along the wall,  0 = no preference,  "
                 f"-1 = pointing at it\n"
                 f"solid = measured   |   dashed = the same chains spun to a "
                 f"random orientation that still fits inside", fontsize=9)
    a1.legend(frameon=False, fontsize=8, loc="lower left", ncol=min(3, n))

    a2.set_ylabel("excess over the null")
    a2.set_xlabel("distance from arena centre  r / R")
    a2.set_xlim(0, 1)
    a2.set_title("Measured minus null: the part that is preference, not "
                 "geometry\n(the only form comparable across arena sizes)",
                 fontsize=9)
    a2.legend(frameon=False, fontsize=8, loc="lower left", ncol=min(2, n))

    fig.tight_layout()
    save_fig(fig, path)
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
    def _stack_2d(key):
        """Interpolate a (T, k) block per trial -> (trials, T, k)."""
        k = all_sr[0][key].shape[1]
        return np.stack([
            np.stack([np.interp(grid, sr["secs"], sr[key][:, b])
                      for b in range(k)], axis=1)
            for sr in all_sr])

    stacked["frac"] = _stack_2d("frac")
    # Only when every trial has a gait log -- averaging a subset would quietly
    # compare different numbers of trials in different figures
    if all("pattern" in sr for sr in all_sr):
        stacked["pattern"] = _stack_2d("pattern")
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
    save_fig(fig, path)
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
    save_fig(fig, path)
    plt.close(fig)


# =============================================================================
# Process one trial / one group of experiments
# =============================================================================

def process_trial(folder, exp_name, bins, labels, args, fps, geom=None):
    pos_all = find_pos_all(folder)
    name = os.path.basename(os.path.normpath(folder))

    gdf, cache, cached = group_table(pos_all, args.max_dist,
                                     not args.no_singletons, args.force)
    if gdf.empty:
        print(f"    [skip] {name}: group table is empty")
        return None

    # frames is the expensive intermediate (1.1M group rows -> Python dicts), and
    # only two figures actually need it; everything else consumes the time series
    # derived from it. So cache the series and build frames only when the cache
    # misses or when those two figures are asked for. Without this, selecting any
    # figure at all -- even an experiment-level one, even radial, which never
    # looks at frames -- paid the full reconstruction for every trial.
    need_frames = bool({"kymograph", "snapshots"} & args.figs)
    spath = series_cache_path(pos_all, args.max_dist,
                              not args.no_singletons, bins)
    sr = None if args.force else read_series_cache(spath)
    series_cached = sr is not None

    frames = None
    if sr is None or need_frames:
        frames = load_frames(gdf, fps, with_members=need_frames)
    if sr is None:
        sr = trial_series(frames, bins)
        write_series_cache(spath, sr)

    # Gait-pattern shares, sampled on the same time axis as everything else so
    # they can be averaged across trials alongside the rest
    if {"patterns", "mean_patterns"} & args.figs and args.pattern_order:
        glog = find_gait_log(folder) or reconstruct_gait_log(folder, fps,
                                                             args.force)
        if glog:
            sr["pattern"] = pattern_shares(glog, sr["secs"], args.pattern_order)
            sr["pattern_rebuilt"] = glog.endswith("_gait_log_rebuilt.csv")

    if "radial" in args.figs and geom is not None:
        sr["radial"] = radial_profile(pos_all, gdf, geom, args.radial_bins,
                                      args.radial_min_size, args.force)
    if "chains" in args.figs and geom is not None:
        sr["chains"] = chain_profile(pos_all, gdf, geom, args.chain_bins,
                                     args.chain_min_size, args.chain_frames,
                                     args.chain_rotations, force=args.force)

    out_dir = args.out or folder
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{exp_name}_{name}" if args.out else name
    title = f"  [{exp_name}/{name}, max_dist={args.max_dist:g}]" if args.title else ""

    made = []

    def p(fig_name):
        fn = os.path.join(out_dir, f"{prefix}_{fig_name}.png")
        made.append(fn)
        return fn

    if "composition" in args.figs:
        fig_composition(sr, labels, p("composition"), title, args.smooth_s)
    if "groupsize" in args.figs:
        fig_groupsize(sr, p("groupsize"), title, args.smooth_s)
    if "counts" in args.figs:
        fig_counts(sr, p("counts"), title, args.smooth_s)
    if "patterns" in args.figs and "pattern" in sr:
        fig_patterns(sr["secs"], sr["pattern"], args.pattern_order,
                     bool(sr["pattern"][:, -1].any()), p("patterns"), title,
                     args.smooth_s, rebuilt=sr.get("pattern_rebuilt", False))
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
        # one entry per frame, so this is len(frames) without needing frames
        "frames": int(len(sr["secs"])), "duration_s": float(sr["secs"][-1]),
        "largest_mean": float(sr["largest"].mean()),
        "largest_steady": float(sr["largest"][tail].mean()),
        "largest_peak": int(sr["largest"].max()),
        "mean_size_steady": float(sr["mean_size"][tail].mean()),
        "n_groups_steady": float(sr["n_groups"][tail].mean()),
        "largest_frac_steady": float(sr["largest_frac"][tail].mean()),
        "alignment_steady": float(sr["alignment"][tail].mean()),
    }
    src = ("series cached" if series_cached else
           ("groups cached" if cached else "computed"))
    print(f"    {name}: {stats['frames']} frames / {stats['duration_s']:.1f}s, "
          f"largest group mean {stats['largest_mean']:.1f} / last {args.steady_s:g}s "
          f"{stats['largest_steady']:.1f} / peak {stats['largest_peak']}"
          f"   [{src}]")
    for fn in made:
        print(f"      -> {written_name(fn)}")
    return sr, stats


STEADY_COLS = ("largest_mean", "largest_steady", "largest_peak",
               "mean_size_steady", "n_groups_steady",
               "largest_frac_steady", "alignment_steady")


def process_experiment(exp_dir, args, radial_sink=None, chain_sink=None):
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    trials = find_trials(exp_dir)
    if not trials:
        print(f"  [skip] no trial subdirectories containing *_POS_ALL.csv")
        return None

    fps, fps_src = (args.fps, "--fps") if args.fps else detect_fps(exp_dir)
    print(f"  {len(trials)} trials, fps={fps:g} ({fps_src})")

    geom = None
    if {"radial", "chains"} & args.figs:
        geom = ring_geometry(exp_dir)
        if geom is None:
            print("    [skip] radial/chains: no readable arena geometry "
                  "(need W / H / INNER_R in config_snapshot.json)")
        else:
            print(f"    arena: centre ({geom[0]:.0f}, {geom[1]:.0f}), "
                  f"R={geom[2]:g}, {geom[3]}"
                  + (f"({geom[4]} sides)" if geom[3].startswith("poly") else "")
                  + f"   [from {geom[5]}]"
                  + ("  <- no config_snapshot.json here; verify this is the "
                     "arena these trials actually ran in"
                     if geom[5] == "config.py" else ""))

    # Bin edges must be shared across the whole experiment, otherwise the
    # trials' stacked-area plots aren't comparable and can't be averaged
    # together. Scan each trial's robot count first and use the max to set the bins.
    # The experiment's own snapshot already records this; counting distinct
    # Agent_IDs means parsing every trial's POS_ALL (108MB each at N=100) on
    # every run just to learn a number that is sitting in a 7KB json. Verified
    # equal to the scan on the 0916 runs. Falls back to the scan when there is
    # no snapshot, or when it does not name N_SMARTICLES.
    n_max = 0
    snap_n = read_snapshot(exp_dir).get("N_SMARTICLES")
    if isinstance(snap_n, int) and snap_n > 0:
        n_max = snap_n
    else:
        for t in trials:
            n_max = max(n_max, int(pd.read_csv(find_pos_all(t),
                                               usecols=["Agent_ID"])
                                   ["Agent_ID"].nunique()))
    bins, labels = size_bins(n_max)

    all_sr, all_stats = [], []
    for t in trials:
        r = process_trial(t, exp_name, bins, labels, args, fps, geom)
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
    if "pattern" in stacked:
        # The numbers behind the patterns figure: several categorical slots are
        # low-contrast on white, so the table view is not optional
        cols = [str(c) for c in args.pattern_order] + ["other"]
        for b, lab in enumerate(cols):
            if b < stacked["pattern"].shape[2]:
                ts[f"pattern_{lab}_mean"] = stacked["pattern"][:, :, b].mean(axis=0)
                ts[f"pattern_{lab}_std"] = stacked["pattern"][:, :, b].std(axis=0)
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
            print(f"    -> {written_name(fp)}")

    if "mean_patterns" in args.figs:
        if "pattern" in stacked:
            share = stacked["pattern"].mean(axis=0)
            fp = os.path.join(out_dir, f"{exp_name}_mean_patterns.png")
            fig_patterns(grid, share, args.pattern_order,
                         bool(share[:, -1].any()), fp, title, sm,
                         note=f"(mean of {stacked['pattern'].shape[0]} trials, "
                              f"{sm:g}s moving average)" if sm > 0 else
                              f"(mean of {stacked['pattern'].shape[0]} trials)",
                         rebuilt=any(sr.get("pattern_rebuilt") for sr in all_sr))
            print(f"    -> {written_name(fp)}")
        else:
            n_log = sum(1 for t in trials if find_gait_log(t))
            print(f"    [skip] mean_patterns: {n_log}/{len(trials)} trials have "
                  f"a *_gait_log.csv (need all of them; it is written by runs "
                  f"from 2026-09 on)")

    # Pool this experiment's trials into one radial profile: the histograms are
    # counts, so they add -- no interpolation onto a common axis needed, and a
    # longer trial legitimately contributes more observations.
    if "radial" in args.figs and all("radial" in sr for sr in all_sr):
        prof = merge_radial([sr["radial"] for sr in all_sr])
        fp = os.path.join(out_dir, f"{exp_name}_radial.png")
        fig_radial([prof], [exp_name], [geom[2]], fp, title,
                   args.radial_min_size)
        print(f"    -> {written_name(fp)}")
        if radial_sink is not None:
            radial_sink.append((exp_name, prof, geom[2]))

    if "chains" in args.figs and all("chains" in sr for sr in all_sr):
        prof = merge_chains([sr["chains"] for sr in all_sr])
        fp = os.path.join(out_dir, f"{exp_name}_chains.png")
        fig_chains([prof], [exp_name], fp, title, args.chain_min_size)
        print(f"    -> {written_name(fp)}")
        if chain_sink is not None:
            chain_sink.append((exp_name, prof))

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

def open_pickled_figure(path):
    """
    Reopen a .fig.pickle in an interactive window -- the MATLAB `openfig`
    equivalent.

    The module forces the Agg backend at import so batch runs never need a
    display, so this has to switch to an interactive one first. matplotlib
    allows that after pyplot is imported, but only to a backend whose GUI
    toolkit is actually installed, hence the walk through the candidates.
    """
    if not os.path.isfile(path):
        print(f"No such figure: {path}")
        return 1
    for backend in ("QtAgg", "TkAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(backend, force=True)
            break
        except Exception:
            continue
    else:
        print("No interactive matplotlib backend available (tried QtAgg, "
              "TkAgg, Qt5Agg, MacOSX). Install PyQt5 or tkinter, or load the "
              "pickle yourself:\n"
              "    import pickle, matplotlib.pyplot as plt\n"
              f"    fig = pickle.load(open(r'{path}', 'rb'))\n"
              "    plt.show()")
        return 1

    import matplotlib.pyplot as ipl                     # re-bind to the new backend
    try:
        with open(path, "rb") as f:
            fig = pickle.load(f)
    except Exception as e:
        print(f"Could not load {path}: {type(e).__name__}: {e}\n"
              f"Pickled figures are tied to the matplotlib that wrote them "
              f"(this is {matplotlib.__version__}); re-run the plot to "
              f"regenerate, or use the svg/pdf copy instead.")
        return 1

    # A figure built through plt.subplots() carries a flag that makes unpickling
    # re-register it with pyplot on its own, and then plt.show() finds it. Don't
    # rely on that: if it did not happen, the figure has no GUI manager and
    # show() would open nothing at all. Borrow a throwaway figure's canvas.
    if getattr(fig.canvas, "manager", None) is None:
        shim = ipl.figure()
        mgr = shim.canvas.manager
        mgr.canvas.figure = fig
        fig.set_canvas(mgr.canvas)
    print(f"{os.path.basename(path)} — the toolbar's sliders button opens "
          f"'Edit axis, curves and images parameters'. Close the window to exit.")
    ipl.show()
    return 0


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
                    default=["all"],
                    help="Only plot these figures. Default 'all' = everything except the opt-in patterns/mean_patterns/radial; use 'everything' for those too")
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
    ap.add_argument("--max-patterns", type=int, default=8,
                    help="How many gait commands get their own colour in the patterns figure; the rest fold into 'other' (default 8)")
    ap.add_argument("--radial-bins", type=int, default=20,
                    help="Number of radial bins in the radial figure (default 20)")
    ap.add_argument("--radial-min-size", type=int, default=3,
                    help="Smallest group counted as 'grouped' in the radial figure's P(...) panel (default 3). "
                         "Low values saturate -- if nearly every robot qualifies everywhere the curve goes flat "
                         "and hides the effect; try 8 or 12 at N=100. The mean-group-size panel has no threshold "
                         "and is unaffected")
    ap.add_argument("--radial-compare", default=None,
                    help="Also write one radial figure overlaying every experiment, to this path")
    ap.add_argument("--chain-min-size", type=int, default=3,
                    help="Smallest group treated as a chain in the chains figure (default 3)")
    ap.add_argument("--chain-bins", type=int, default=12,
                    help="Number of radial bins in the chains figure (default 12)")
    ap.add_argument("--chain-frames", type=int, default=400,
                    help="Frames subsampled per trial for the chains figure (default 400)")
    ap.add_argument("--chain-rotations", type=int, default=12,
                    help="Random orientations tried per chain to build the geometric null (default 12)")
    ap.add_argument("--chain-compare", default=None,
                    help="Also write one chains figure overlaying every experiment, to this path")
    ap.add_argument("--formats", nargs="+", choices=list(FIG_FORMATS),
                    default=["png"], metavar="FMT",
                    help="Which formats to write each figure in (default png). "
                         "svg/pdf keep text as live text, so fonts and colours stay editable in "
                         "Inkscape or Illustrator. 'pickle' writes <name>.fig.pickle, matplotlib's "
                         "answer to a MATLAB .fig -- reopen it with --open. Several may be given")
    ap.add_argument("--font", default=None, metavar="NAME",
                    help="Font for every figure (default: the first installed of "
                         + "/".join(FONT_PREFERENCE[:4]) + "/...). Only one name is written "
                         "into the SVG, because Illustrator reports an error for every name "
                         "in a fallback chain it does not have, rather than walking it")
    ap.add_argument("--open", dest="open_fig", default=None, metavar="FIG.pickle",
                    help="Reopen a .fig.pickle in an interactive window and exit. "
                         "The toolbar's green sliders button edits line colours, styles, "
                         "labels and axis limits in place; the figure is a normal matplotlib "
                         "Figure, so it can also be changed from code before re-saving")
    args = ap.parse_args()

    if args.open_fig:
        return open_pickled_figure(args.open_fig)

    global OUTPUT_FORMATS
    OUTPUT_FORMATS = list(dict.fromkeys(args.formats))   # de-dup, keep order

    font = choose_font(args.font)
    if font and {"svg", "pdf", "eps"} & set(OUTPUT_FORMATS):
        print(f"Vector output in {font} (one font name per label, so it opens "
              f"without missing-font errors)")

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

    # Which command gets which colour is decided **once, across every
    # experiment in this run**, so a given gait is the same colour in every
    # figure produced. Deciding it per trial would repaint the same command
    # whenever a trial happened to use a different set -- colour has to follow
    # the command, not its rank in one file. Gait logs are a few hundred rows,
    # so scanning them all up front is free.
    args.pattern_order = ()
    if {"patterns", "mean_patterns"} & args.figs:
        totals, n_reb = {}, 0
        for exp in exps:
            fps_e = args.fps or detect_fps(exp)[0]
            for t in find_trials(exp):
                # Rebuilding here rather than lazily in process_trial: the
                # colour assignment below has to see every command in the run
                # before the first figure is drawn
                if not find_gait_log(t):
                    try:
                        if reconstruct_gait_log(t, fps_e, args.force):
                            n_reb += 1
                    except Exception as e:
                        print(f"  [note] could not rebuild the gait log for "
                              f"{t}: {type(e).__name__}: {e}")
                for cmd, n in gait_commands(t).items():
                    totals[cmd] = totals.get(cmd, 0) + n
        if n_reb:
            print(f"  rebuilt the gait log for {n_reb} trial(s) that predate "
                  f"*_gait_log.csv (cached as *_gait_log_rebuilt.csv)")
        if totals:
            keep = sorted(totals, key=lambda c: (-totals[c], c))[:max(1, args.max_patterns)]
            args.pattern_order = tuple(sorted(keep))   # stable, by command value
            extra = len(totals) - len(keep)
            print(f"  gait patterns: {', '.join(str(c) for c in args.pattern_order)}"
                  + (f"  (+{extra} folded into 'other')" if extra > 0 else ""))
        else:
            print("  [note] no *_gait_log.csv found in any trial; "
                  "the patterns figures will be skipped")

    rows, radial_sink, chain_sink = [], [], []
    for i, exp in enumerate(exps, 1):
        print(f"[{i}/{len(exps)}] {exp}")
        try:
            r = process_experiment(exp, args, radial_sink, chain_sink)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  [FAIL] {exp}: {type(e).__name__}: {e}")

    if args.radial_compare and radial_sink:
        names = [x[0] for x in radial_sink]
        fig_radial([x[1] for x in radial_sink], names, [x[2] for x in radial_sink],
                   args.radial_compare,
                   f"  [{len(names)} experiments, max_dist={args.max_dist:g}]"
                   if args.title else "",
                   args.radial_min_size)
        print(f"\nRadial comparison written to {args.radial_compare}")
    elif args.radial_compare:
        print("\n[note] --radial-compare: no experiment produced a radial "
              "profile (is 'radial' in --figs?)")

    if args.chain_compare and chain_sink:
        fig_chains([x[1] for x in chain_sink], [x[0] for x in chain_sink],
                   args.chain_compare,
                   f"  [{len(chain_sink)} experiments, max_dist={args.max_dist:g}]"
                   if args.title else "",
                   args.chain_min_size)
        print(f"Chain-orientation comparison written to {args.chain_compare}")
    elif args.chain_compare:
        print("\n[note] --chain-compare: no experiment produced a chain "
              "profile (is 'chains' in --figs?)")

    if args.summary and rows:
        pd.DataFrame(rows).to_csv(args.summary, index=False)
        print(f"\nSummary written to {args.summary}")
    print(f"\nDone: {len(rows)}/{len(exps)} experiments")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

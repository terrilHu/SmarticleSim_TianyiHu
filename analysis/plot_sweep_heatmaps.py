"""
plot_sweep_heatmaps.py  -  amplitude x frequency heat maps for the gait sweep.

Two figures, one per quantity, from sweep_largest_groupsize.csv and
sweep_mean_groupsize.csv. The grid is the 6 x 8 gait table; only the 25 cells
the pilot certified are filled, and the 23 it rejected are drawn as empty
rather than left to look like low values -- they are not low, they were never
measured, because the arms could not deliver the commanded amplitude there.

If group_persistence.py has been run on the sweep, a third figure shows the
membership memory q(30 s), in green so it is not read as a size.

With the steady-state method, a fourth figure shows how long the steady states
last, from _sweep_steady_segments.csv, in orange so it is read as a duration
rather than a size.

    python plot_sweep_heatmaps.py datafile/1014_sweep [--only largest mean q30 duration]
"""

import math
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import t as student_t

# Keep SVG text as text and pin one installed font name. matplotlib's default
# writes a ten-name CSS fallback chain into every label, and Illustrator tries
# to resolve each one instead of walking the list, so it opens with a missing
# font error per name it lacks.
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.unicode_minus"] = False
for _f in ("Arial", "Helvetica", "Liberation Sans", "Segoe UI", "DejaVu Sans"):
    if _f in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [_f]
        break

# The documented single-hue sequential ramp, light -> dark. A sequential scale
# is one hue by rule: a rainbow map invents ordering the data does not have and
# makes equal steps look unequal.
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
        "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281",
        "#0d366b"]
CMAP = LinearSegmentedColormap.from_list("seq_blue", BLUE)
# Same span of lightness in green, around the palette's series green #008300,
# so a stability figure is never read as another group-size figure.
GREEN = ["#d8f0d6", "#c3e7c0", "#acdda9", "#94d291", "#7bc679", "#62b961",
         "#49ab4a", "#2f9c35", "#178c22", "#007c15", "#006b12", "#005a10",
         "#004a0d"]
CMAP_GREEN = LinearSegmentedColormap.from_list("seq_green", GREEN)
# The palette's second sequential hue (orange, #eb6834), stepped at the same
# OKLab lightness as BLUE so a darker cell means the same "more" in each figure.
ORANGE = ["#ffd5c6", "#ffbfa8", "#ffa98a", "#f8946f", "#f37e52", "#eb6834",
          "#da5821", "#c94908", "#b43f01", "#9e3703", "#8a2e01", "#752601",
          "#611e01"]
CMAP_ORANGE = LinearSegmentedColormap.from_list("seq_orange", ORANGE)
C_EMPTY = "#e8e6e1"      # a cell that was never measured
INK, INK_LIGHT, MUTED = "#1a1a1a", "#ffffff", "#55534e"

# Layout of the hand-finished figures in <sweep>/analysis (Illustrator, from
# heatmap_largest_groupsize.svg), in points on the 8.6 x 7.8 in page, so a
# freshly drawn figure drops in beside them without re-editing.
PAGE_W, PAGE_H = 619.2, 561.6
AX_BOX = (40.89, 73.6, 510.55, 448.73)       # x, y from top, w, h
CBAR_BOX = (562.08, 109.0, 19.2, 378.2)
NOTE_Y = (6.9, 23.7, 34.4)                   # baselines of the note lines
TITLE_Y = 62.1
FS_TEXT, FS_TITLE, FS_NOTE = 13, 12, 8.5


def _fig_rect(box):
    x, y, w, h = box
    return [x / PAGE_W, 1 - (y + h) / PAGE_H, w / PAGE_W, h / PAGE_H]


def heatmap(df, value_col, sem_col, title, cbar_label, note, path,
            status_col=None, vmin=None, vmax=None, digits=1, cmap=CMAP,
            cbar_ticks=None, sem_digits=None):
    amps = sorted(df.cmd_ampli_deg.unique())
    freqs = sorted(df.cmd_freq_hz.unique())
    # Rows are frequency, columns are amplitude: amplitude is the variable
    # that actually moves these numbers, and a horizontal axis is the one a
    # reader scans first.
    V = np.full((len(freqs), len(amps)), np.nan)
    S = np.full((len(freqs), len(amps)), np.nan)
    flag = np.zeros((len(freqs), len(amps)), dtype=bool)
    for _, r in df.iterrows():
        i, j = freqs.index(r.cmd_freq_hz), amps.index(r.cmd_ampli_deg)
        V[i, j] = r[value_col]
        S[i, j] = r[sem_col]
        # Anything the stats step did not certify as settled gets marked.
        # What counts as not-settled depends on the method, so key off the
        # one good value rather than enumerate the bad ones.
        if status_col and str(r.get(status_col, "steady")) != "steady":
            flag[i, j] = True

    fig = plt.figure(figsize=(PAGE_W / 72, PAGE_H / 72))
    ax = fig.add_axes(_fig_rect(AX_BOX))
    # Paint the whole grid as "not measured" first, then the data on top, so an
    # absent cell reads as absent instead of as the bottom of the colour scale.
    ax.imshow(np.zeros_like(V), cmap=LinearSegmentedColormap.from_list(
        "e", [C_EMPTY, C_EMPTY]), aspect="auto", origin="lower")
    # vmin/vmax are passed in so the synchronised and desynchronised sweeps
    # share one scale. Letting each figure autoscale would make the same blue
    # mean 13.6 in one and 8.5 in the other, which is exactly the comparison
    # these two figures exist to support.
    im = ax.imshow(np.ma.masked_invalid(V), cmap=cmap, aspect="auto",
                   origin="lower", vmin=vmin, vmax=vmax)

    # An unmeasured cell is the flat grey alone; the method note says why.
    for i in range(len(freqs)):
        for j in range(len(amps)):
            if np.isnan(V[i, j]):
                continue
            # Whichever ink has the higher WCAG contrast on this fill. A fixed
            # cut on gamma-encoded luminance put white on mid orange at ~2.5:1.
            rgb = np.asarray(im.cmap(im.norm(V[i, j]))[:3])
            lin = np.where(rgb <= 0.04045, rgb / 12.92,
                           ((rgb + 0.055) / 1.055) ** 2.4)
            lum = float(lin @ [0.2126, 0.7152, 0.0722])
            col = INK if (lum + 0.05) / 0.05 > 1.05 / (lum + 0.05) else INK_LIGHT
            mark = " *" if flag[i, j] else ""
            sd = digits + 1 if sem_digits is None else sem_digits
            ax.text(j, i, f"{V[i, j]:.{digits}f}{mark}\n±{S[i, j]:.{sd}f}",
                    ha="center", va="center", fontsize=FS_TEXT, color=col,
                    linespacing=0.8)

    ax.set_xticks(range(len(amps)))
    ax.set_xticklabels([f"{a:g}°" for a in amps])
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f"{f:g}" for f in freqs])
    ax.tick_params(labelsize=FS_TEXT)
    ax.set_xlabel("Commanded amplitude", fontsize=FS_TEXT, labelpad=2)
    ax.set_ylabel("Commanded frequency (Hz)", fontsize=FS_TEXT, labelpad=2)
    # Note above the title, both on the page rather than the axes, at the
    # reference figure's baselines.
    xc = (AX_BOX[0] + AX_BOX[2] / 2) / PAGE_W
    lines = note.splitlines()
    for k, line in enumerate(lines):
        y = NOTE_Y[k] if k < len(NOTE_Y) else NOTE_Y[-1] + (k - len(NOTE_Y) + 1) * 10.8
        fig.text(xc, 1 - y / PAGE_H, line, ha="center", va="baseline",
                 fontsize=FS_NOTE, color=MUTED)
    fig.text(xc, 1 - TITLE_Y / PAGE_H, title, ha="center", va="baseline",
             fontsize=FS_TITLE, color="#000000")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(amps), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(freqs), 1), minor=True)
    ax.grid(which="minor", color="#fcfcfb", linewidth=2)
    ax.tick_params(which="minor", length=0)

    # The colourbar label sits at the page edge, so wide tick labels push it
    # off; a fractional scale passes coarser ticks.
    cb = fig.colorbar(im, cax=fig.add_axes(_fig_rect(CBAR_BOX)), ticks=cbar_ticks)
    cb.set_label(cbar_label, fontsize=FS_TEXT, labelpad=1)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=4, width=0.8, labelsize=FS_TEXT)

    for ext in ("svg", "png"):
        fig.savefig(f"{os.path.splitext(path)[0]}.{ext}", dpi=150)
    plt.close(fig)
    return f"{os.path.splitext(path)[0]}.svg"


# One scale per quantity, spanning both the synchronised and desynchronised
# sweeps (observed 5.30-13.64 and 2.23-6.21), rounded outward. Fixed rather than
# computed from whatever happens to be on disk, so a figure redrawn on its own
# still lands on the same scale as its counterpart.
SCALE_LARGEST = (5.0, 13.0)
SCALE_MEAN = (2.0, 6.0)
SCALE_Q30 = (0.4, 1.0)   # observed 0.66-1.00 synchronised, 0.50-0.98 desynchronised
# Mean steady-segment length, in MINUTES on the figure: one window (1 min) is
# the shortest possible segment and a trial is 8.3 min, and single-digit ticks
# leave the colourbar label room on the page, where 3-digit seconds did not.
SCALE_DUR = (1.0, 8.0)

# q(30 s) comes from group_persistence.py's per-trial table; keep its window
# settings quoted from the module so the note cannot drift from the numbers.
Q_LAG_S = 30


def persistence_table(root):
    """
    Per-gait mean / sd / sem of q(30 s) over trials, or None if
    group_persistence.py has not been run on this sweep. Also written out as
    sweep_q30.csv beside the other sweep_*.csv tables.
    """
    path = os.path.join(root, "_persistence_per_trial.csv")
    if not os.path.isfile(path):
        return None
    p = pd.read_csv(path)
    col = f"q_{Q_LAG_S}s"
    g = p.groupby("pattern")[col]
    t = pd.DataFrame({"n_trials": g.size(), "q30_mean": g.mean(), "q30_sd": g.std()})
    t["q30_sem"] = t.q30_sd / np.sqrt(t.n_trials)
    t["q30_ci95_halfwidth"] = t.q30_sem * t.n_trials.map(lambda n: student_t.ppf(0.975, n - 1))
    gait = pd.read_csv(os.path.join(root, "_sweep_per_trial.csv"))[
        ["pattern", "Y", "Z", "cmd_ampli_deg", "cmd_freq_hz"]].drop_duplicates("pattern")
    t = gait.merge(t.reset_index(), on="pattern")
    t.to_csv(os.path.join(root, "sweep_q30.csv"), index=False)
    return t

def duration_table(root):
    """
    Per-gait mean / sd / sem over trials of the mean steady-segment length, or
    None if the sweep has no per-segment table (fixed-window method). Each trial
    is first reduced to the plain mean length of its segments, so a trial with
    four short plateaus counts once, like a trial with one long one. Written out
    as sweep_steady_duration.csv beside the other sweep_*.csv tables.
    """
    path = os.path.join(root, "_sweep_steady_segments.csv")
    if not os.path.isfile(path):
        return None
    sg = pd.read_csv(path)
    per = (sg.groupby(["pattern", "trial_id"])
             .agg(seg_dur=("dur", "mean"), n_segments=("dur", "size"),
                  steady_s=("dur", "sum"))
             .reset_index())
    g = per.groupby("pattern")
    t = pd.DataFrame({"n_trials": g.size(),
                      "seg_dur_mean": g.seg_dur.mean(),
                      "seg_dur_sd": g.seg_dur.std(),
                      "n_segments_mean": g.n_segments.mean(),
                      "steady_s_mean": g.steady_s.mean()})
    t["seg_dur_sem"] = t.seg_dur_sd / np.sqrt(t.n_trials)
    t["seg_dur_ci95_halfwidth"] = t.seg_dur_sem * t.n_trials.map(
        lambda n: student_t.ppf(0.975, n - 1))
    t["seg_dur_mean_min"] = t.seg_dur_mean / 60.0
    t["seg_dur_sem_min"] = t.seg_dur_sem / 60.0
    gait = sg[["pattern", "Y", "Z", "cmd_ampli_deg", "cmd_freq_hz"]
              ].drop_duplicates("pattern")
    t = gait.merge(t.reset_index(), on="pattern").round(3)
    t.to_csv(os.path.join(root, "sweep_steady_duration.csv"), index=False)
    return t


def duration_note(d):
    """Method-note section for the steady-duration figure; empty when not drawn."""
    if d is None:
        return ""
    NL = chr(10)
    return (
        "## Steady-state duration" + NL + NL +
        "From `_sweep_steady_segments.csv`, summarised in "
        "`sweep_steady_duration.csv`. Each steady segment found above has a "
        "length (s). Each trial is reduced to the plain mean length of its "
        "segments -- so a trial that held four one-minute plateaus scores about "
        "60 s, one that held a single five-minute plateau about 300 s -- and the "
        "cell shows the mean over trials, with the SEM over trials. The figure "
        "is in minutes; the csv keeps seconds (`seg_dur_*`) beside the minute "
        "columns it plots (`*_min`)." + NL + NL +
        "The minimum possible length is one window (" + WIN_TXT + "), and a "
        "500 s trial bounds it from above, so long values are partly censored "
        "by the end of the run. The csv also carries the mean number of "
        "segments per trial and the mean total steady time per trial." + NL + NL +
        "Colour scale fixed at " + f"{SCALE_DUR[0]:g}-{SCALE_DUR[1]:g}" +
        " min, shared with the other sweep; orange, so it is not read as a "
        "group size." + NL + NL)


# Shown verbatim in the generated method note, so the note always quotes the
# settings this build actually used.
WIN_TXT = "60 s"
DRIFT_TXT = "8%"
DEV_TXT = "15%"
STEP_TXT = "8%"


def persistence_note(q):
    """Method-note section for the q(30 s) figure; empty when it was not drawn."""
    if q is None:
        return ""
    from group_persistence import BLOCK_S, MIN_SIZE, SAMPLE_HZ, T0_S
    NL = chr(10)
    m, lag, t0 = str(MIN_SIZE), str(Q_LAG_S), f"{T0_S:g}"
    per = int(round(BLOCK_S * SAMPLE_HZ))
    return (
        "## Membership memory q(" + lag + " s)" + NL + NL +
        "From `_persistence_per_trial.csv` (`group_persistence.py`), summarised "
        "in `sweep_q30.csv`. q asks each robot: of the robots I am grouped with "
        "now, what share am I still grouped with " + lag + " s later?" + NL + NL +
        "### Who is grouped with whom" + NL + NL +
        "Contacts blink on and off at sub-second scale, so time is cut into "
        f"{BLOCK_S:g} s blocks of {per} samples (positions sampled at "
        f"{SAMPLE_HZ:g} Hz, grouped with the same Voronoi rule as above). Robots "
        "i and j are bonded in block b if they shared a group in more than half "
        "of its samples:" + NL + NL +
        r"$$s_{ij}(b) = 1 \iff \#\{\text{samples of } b \text{ in which } i, j "
        r"\text{ share a group}\} > " + str(per // 2) + "$$" + NL + NL +
        "The groups of block b are the connected components of that bond graph. "
        "Only groups of at least " + m + " robots count; a robot in a pair or "
        "alone has no groupmates. Robot i's groupmates at time t (the block "
        "starting at t) are" + NL + NL +
        r"$$G_i(t) = \{\, j \ne i : j \text{ is in } i\text{'s group at } t "
        r"\,\}\quad\text{if that group has} \ge " + m + r"\text{ members, else } "
        r"G_i(t) = \varnothing$$" + NL + NL +
        "i itself is left out of the set, so that a robot never counts as its "
        "own surviving partner." + NL + NL +
        "### Overlap of one robot's groupmates" + NL + NL +
        "For a robot with groupmates at t, compare the two sets with the "
        "Jaccard index (size of the intersection over size of the union):" + NL + NL +
        r"$$J_i(t, \tau) = \frac{|G_i(t) \cap G_i(t+\tau)|}"
        r"{|G_i(t) \cup G_i(t+\tau)|}$$" + NL + NL +
        "- J = 1: exactly the same groupmates." + NL +
        "- J = 0: none of the old groupmates left, including the case where i is "
        "no longer in a group of >= " + m + " at all (then the intersection is empty "
        "and the union is just G_i(t))." + NL +
        r"- The union is never empty, because $G_i(t) \neq \varnothing$." + NL + NL +
        "Example: at t, i is in a group of five, G_i(t) = {a, b, c, d}. At "
        "t + tau, c and d have left and e has joined, G_i(t + tau) = {a, b, e}. "
        "The intersection {a, b} has 2 robots and the union {a, b, c, d, e} has "
        "5, so J = 2/5 = 0.4." + NL + NL +
        "Why the union in the denominator rather than |G_i(t)| (the share of old "
        "groupmates retained): retention ignores who joined. A group of four "
        "that absorbs ten newcomers would still score 1 on retention, although "
        "the robot now sits in a quite different group; Jaccard scores it "
        "4/14. The union penalises losses and gains alike, and J is symmetric "
        "in the two times." + NL + NL +
        "### Average" + NL + NL +
        "q(tau) is the plain mean of J over every (robot, origin) pair where the "
        "robot has groupmates at the origin and both times are inside the "
        "analysed stretch of the trial:" + NL + NL +
        r"$$q(\tau) = \frac{1}{|P_\tau|} \sum_{(i,t) \in P_\tau} J_i(t, \tau), "
        r"\qquad P_\tau = \{(i, t) : G_i(t) \ne \varnothing,\ t \ge " + t0 +
        r"\text{ s},\ t + \tau \le T\}$$" + NL + NL +
        "with t running over block starts and T the end of the run. Each robot "
        "at each origin has weight 1, whatever the size of its group. This is "
        "done per trial; the heat map shows q(" + lag + " s) averaged over trials, "
        "with the SEM over trials." + NL + NL +
        "q(tau) = 1 means the partition is frozen. Under a full random "
        "reshuffle q is close to 0 (two groups of five out of 100 robots share "
        "about 0.16 robots by chance), so the 0-1 range can be read as an "
        "absolute scale. The size-weighted pair version, "
        r"$\sum_{(i,t)} |G_i(t) \cap G_i(t+\tau)| \,/\, \sum_{(i,t)} |G_i(t)|$"
        ", is the probability that two robots grouped together now are grouped "
        "together again tau later (the lagged association rate of "
        "animal-behaviour work, Whitehead 1995); it is kept in the csv as "
        "`pair_q_30s`." + NL + NL +
        "No group is tracked, so q has none of Kaplan-Meier's problems: no "
        "group identity to decide at merges and splits, no right-censoring of "
        "groups that outlive the run, and the sample is the trial, not the "
        "pooled (non-independent) groups." + NL + NL +
        "The lag is a choice, not a fit. q is not a single exponential in lag, "
        "so its value depends on the lag, but the ordering of gaits does not: "
        "q at 10, 30 and 60 s rank the gaits almost identically (Spearman 0.99 "
        "in the synchronised sweep). Error bars as above: one number per trial, "
        "SEM over trials. Colour scale fixed at " + f"{SCALE_Q30[0]:g}-{SCALE_Q30[1]:g}" +
        ", shared with the other sweep." + NL + NL +
        "Caveat: q close to 1 at small amplitude partly reflects robots that "
        "barely move, not a structure that is actively held together." + NL + NL)


def write_method_note(out, root, lg, mn, q=None, dur=None):
    """
    Write the method note next to the figures, filling in the parameters this
    particular set was actually produced with rather than restating defaults.
    """
    import datetime
    NL = chr(10)
    name = os.path.basename(os.path.normpath(root))
    method = str(lg["method"].iloc[0]) if "method" in lg.columns else "fixed"
    n_tr = int(lg["n_trials"].iloc[0])
    n_cells = len(lg)
    n_fr = int(lg["n_frames_per_trial"].iloc[0])

    if method == "steady":
        how = (
            "**Steady-state segments.** A window of " + WIN_TXT + " is slid over the "
            "whole trial. A window counts as steady when the least-squares trend "
            "across it changes the signal by no more than " + DRIFT_TXT + " of that "
            "window's own mean -- a fractional change, so the threshold means the "
            "same thing at every group size. The trend test only sees the slope, "
            "so a window that rises and falls back would pass it; the 10 s moving "
            "average must therefore also stay within " + DEV_TXT + " of the window "
            "mean everywhere in the window. Overlapping steady windows are merged "
            "into one segment only while their means stay within " + STEP_TXT +
            " of the segment's, so two plateaus at different levels whose windows "
            "touch remain two segments." + NL + NL +
            "Every segment counts: a trial may settle at several levels in turn, "
            "and all of them are steady states. A trial's value is the time "
            "average over all its steady time (segments weighted by duration). "
            "`_sweep_steady_segments.csv` lists every segment and "
            "`sweep_steady_occurrence.csv` summarises how many, how long and at "
            "what level, per cell." + NL + NL +
            "Steadiness is judged on the largest-group signal only, and both "
            "quantities are measured over those same segments, so the pair always "
            "describes one state of the system rather than two different "
            "stretches of time." + NL + NL +
            "Within a segment each quantity is the time average over that segment, "
            "not the per-frame extreme: a maximum of a maximum would be a single "
            "frame out of thousands, with no averaging behind it." + NL + NL +
            "A trial with no steady segment is dropped rather than falling back to "
            "a whole-trial average, which would be indistinguishable from a real "
            "steady value. `steady_frac` in the csv is the share of the trial that "
            "was steady; a cell below 0.5 is marked `*` on the figure."
        )
        window = "sliding, " + WIN_TXT
    else:
        how = (
            "**Fixed window.** Every trial is averaged over t >= 400 s, the last "
            "100 s of a 500 s run. `drift_sigma` reports how far the residual "
            "trend inside that window is from zero; a cell at 2 sigma or more is "
            "marked `*` on the figure, meaning the value was still rising when the "
            "trial ended and is therefore a lower bound."
        )
        window = "fixed, t >= 400 s"

    txt = (
        "# " + name + " -- group-size heat maps" + NL + NL +
        "Generated " + datetime.date.today().isoformat() +
        " by `plot_sweep_heatmaps.py` from `sweep_largest_groupsize.csv` and "
        "`sweep_mean_groupsize.csv` (`sweep_stats.py`)." + NL + NL +
        "## What is plotted" + NL + NL +
        "A " + str(n_cells) + "-cell slice of the gait table: commanded amplitude "
        "across, commanded frequency up. The remaining cells of the 6x8 table are "
        "drawn as *not measured* -- the pilot found the arms could not deliver the "
        "commanded amplitude there, so those gaits were never run. They are not "
        "low values." + NL + NL +
        "- **largest group size** -- the biggest group present, per frame." + NL +
        "- **mean group size** -- size-weighted, `sum(s^2)/sum(s)`: pick a robot at "
        "random, how big is its group. Singletons included; the csv also carries "
        "the version that excludes them." + NL + NL +
        "## Grouping" + NL + NL +
        "Voronoi (Delaunay) adjacency cut at `max_dist = 65 px`, then connected "
        "components -- the same function the live strategy uses. Body length is "
        "105 px in these runs, so the threshold is a constant 0.62 body lengths." + NL + NL +
        "## Which part of the trial" + NL + NL +
        how + NL + NL +
        "## Error bars" + NL + NL +
        "Each trial is reduced to one number, and the statistics are taken across "
        "the " + str(n_tr) + " trials. Frames are not independent samples: the "
        "largest-group series has an autocorrelation time of about 27 s, so a "
        "trial contributes a handful of independent samples, not thousands. "
        "Treating frames as independent gives an error bar ~59x too small." + NL + NL +
        "- `sd` -- spread of individual trials." + NL +
        "- `sem` -- `sd/sqrt(" + str(n_tr) + ")`, the uncertainty of the cell's "
        "mean. This is the +/- printed in each cell." + NL +
        "- `ci95_halfwidth` -- `sem * 2.262`, the t factor for df = " + str(n_tr - 1) +
        ". Not 1.96; with n = " + str(n_tr) + " the normal approximation is wrong." + NL + NL +
        "All " + str(n_cells) + " cells use the same ten initial conditions, so a "
        "comparison between two cells -- or between this sweep and its counterpart "
        "-- should be made on the paired differences in `_sweep_per_trial.csv`, "
        "not from the individual error bars, which double-count the shared "
        "initial-condition variation." + NL + NL +
        "## Colour scale" + NL + NL +
        "Fixed at " + str(SCALE_LARGEST[0]) + "-" + str(SCALE_LARGEST[1]) +
        " for largest and " + str(SCALE_MEAN[0]) + "-" + str(SCALE_MEAN[1]) +
        " for mean, shared with the other sweep so the same blue means the same "
        "number in both. Single-hue sequential: a rainbow map would invent "
        "ordering the data does not have." + NL + NL +
        persistence_note(q) +
        duration_note(dur) +
        "## Not covered by the error bars" + NL + NL +
        "`max_dist = 65 px`; the window choice (" + window + "); and the gap "
        "between commanded and delivered amplitude, which the pilot measured at "
        "up to 6.7% even among the cells that were kept." + NL
    )
    path = os.path.join(out, "METHOD.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(txt)
    return path


OUT_SUBDIR = "analysis"


FIGURES = ("largest", "mean", "q30", "duration")


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default=os.path.join("datafile", "1014_sweep"))
    # The figures in analysis/ get finished by hand; --only redraws one
    # without overwriting the others.
    ap.add_argument("--only", nargs="+", choices=FIGURES, default=list(FIGURES))
    args = ap.parse_args()
    root = args.root
    lg = pd.read_csv(os.path.join(root, "sweep_largest_groupsize.csv"))
    mn = pd.read_csv(os.path.join(root, "sweep_mean_groupsize.csv"))
    # Figures and their method note go in one folder, so a set can be handed on
    # without the reader having to reconstruct how the numbers were made.
    out = os.path.join(root, OUT_SUBDIR)
    os.makedirs(out, exist_ok=True)
    n = int(lg.n_trials.iloc[0])
    # The subtitle has to follow the method the csv was produced with --
    # stating a fixed window over steady-state numbers would misdescribe
    # the figure.
    method = str(lg["method"].iloc[0]) if "method" in lg.columns else "fixed"
    span = (f"all steady stretches (sliding {WIN_TXT} window, drift ≤ {DRIFT_TXT}, band ±{DEV_TXT})"
            if method == "steady" else "t≥400s of a 500s trial")
    flag_txt = ("* = under half the trial was steady, so the value rests on little data"
                if method == "steady"
                else "* = still drifting at the end of the window (residual trend ≥ 2σ), so a lower bound")
    base = (f"N=100, no strategy, {span}, max_dist=65px"
            "  |  colour scale shared with the other sweep" + chr(10) +
            f"cell: mean ± SEM over {n} trials, the same {n} initial "
            "conditions in every cell")

    figs = []
    if "largest" in args.only:
        figs.append(heatmap(lg, "largest_mean", "largest_sem",
                            "Largest steady-state group size",
                            "Robots in the largest group",
                            base + chr(10) + flag_txt,
                            os.path.join(out, "heatmap_largest_groupsize"),
                            status_col="status",
                            vmin=SCALE_LARGEST[0], vmax=SCALE_LARGEST[1]))
    if "mean" in args.only:
        figs.append(heatmap(mn, "mean_size_w_mean", "mean_size_w_sem",
                            "Mean steady-state group size",
                            "Size-weighted mean group size",
                            base + "\nsize-weighted, singletons included "
                                   "(∑s²/∑s: pick a robot, how big is its group)",
                            os.path.join(out, "heatmap_mean_groupsize"),
                            vmin=SCALE_MEAN[0], vmax=SCALE_MEAN[1]))
    q = persistence_table(root)
    if q is not None and "q30" in args.only:
        from group_persistence import BLOCK_S, MIN_SIZE, T0_S
        nq = int(q.n_trials.iloc[0])
        figs.append(heatmap(
            q, "q30_mean", "q30_sem",
            f"Group membership memory q({Q_LAG_S} s)",
            f"Groupmates still shared after {Q_LAG_S} s",
            f"N=100, no strategy, time origins t≥{T0_S:g}s, max_dist=65px, "
            f"groups ≥{MIN_SIZE}, {BLOCK_S:g}s majority blocks"
            " | colour scale shared with the other sweep" + chr(10) +
            f"cell: mean ± SEM over {nq} trials, the same {nq} initial "
            "conditions in every cell" + chr(10) +
            f"q = share of a robot's groupmates still grouped with it {Q_LAG_S} s later "
            "(1 = frozen, 0 = fully reshuffled)",
            os.path.join(out, f"heatmap_q{Q_LAG_S}"),
            vmin=SCALE_Q30[0], vmax=SCALE_Q30[1], digits=2, cmap=CMAP_GREEN,
            cbar_ticks=np.arange(SCALE_Q30[0], SCALE_Q30[1] + 1e-9, 0.1)))
    dur = duration_table(root) if method == "steady" else None
    if dur is not None and "duration" in args.only:
        figs.append(heatmap(
            dur, "seg_dur_mean_min", "seg_dur_sem_min",
            "Steady-state duration",
            "Length of a steady state (min)",
            base + chr(10) +
            "a trial's value = mean length of its steady segments "
            "(each ≥ 1 min; a trial lasts 8.3 min)",
            os.path.join(out, "heatmap_steady_duration"),
            vmin=SCALE_DUR[0], vmax=SCALE_DUR[1], digits=1,
            cmap=CMAP_ORANGE,
            cbar_ticks=np.arange(SCALE_DUR[0], SCALE_DUR[1] + 1e-9, 1)))
    doc = write_method_note(out, root, lg, mn, q, dur)
    for x in (*figs, doc):
        print("->", x)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
batch_grouping.py
-----------------
Batch driver around pos_all_grouping.py.

Walks the sweep tree under {ROOT}/datafile, runs the group-level alignment
analysis on every trial's POS_ALL file, writes per-trial result CSVs (and an
A(n) plot) next to each POS_ALL file, and aggregates one steady-state row per
trial into a single summary CSV at {ROOT}.

For trials whose id is 0002 it ALSO renders the group-overlay video.

Expected layout (folder names exactly as in the sweep):
    {ROOT}/datafile/<exp_name>/trial_0000/trial_0000_POS_ALL.csv
    {ROOT}/datafile/<exp_name>/trial_0001/trial_0000_POS_ALL.csv
    {ROOT}/datafile/<exp_name>/trial_0002/trial_0000_POS_ALL.csv
    {ROOT}/videos/<exp_name>/trial_0000.mp4          (one video per exp_name)

This file must sit in the same folder as pos_all_grouping.py and
pos_all_alignment.py so the imports below resolve.

Run:
    python batch_grouping.py                 # uses ROOT set below
    python batch_grouping.py --root D:/data   # override ROOT
"""

import argparse
import os
import re
import time
import traceback
from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from pos_all_grouping import (
    process_pos_all_groups,
    compute_frame_summary,
    compute_alignment_vs_size,
    steady_state_summary,
    plot_alignment_vs_size,
    render_group_video,
)


# =============================================================================
# User settings  (edit ROOT, or pass --root on the command line)
# =============================================================================

ROOT = r"C:/Users/tianyihu/Pictures/Camera Roll/0604_unsync"

MAX_DIST           = 90.0     # Voronoi adjacency threshold (pixels)
FPS                = 60.0     # data frame rate (Step per second)
LAST_SECONDS       = 10.0     # trailing window for steady-state A(n)
INCLUDE_SINGLETONS = True     # keep size-1 groups in A(n) / frame summary

PLOT_AN            = True     # save an A(n) PNG next to each POS_ALL file
RENDER_VIDEOS      = True     # render group video for trial_0002
VIDEO_STRIDE       = 2        # video-frame -> data-step stride (as in heatmap.py)
VIDEO_TRIAL_ID     = 2        # which trial id gets a video (2 == trial_0002)

POS_ALL_SUFFIX     = "_POS_ALL.csv"             # appended to trial folder name to form filename
OUT_PREFIX         = "grouping"                 # prefix for per-trial result files
SUMMARY_NAME       = "grouping_summary.csv"     # aggregate CSV written at ROOT


# =============================================================================
# Helpers
# =============================================================================

def parse_exp_name(name: str) -> dict:
    """Pull sweep parameters out of a folder name like
    'trial_N17_J1p0_J2p0_W1f6_W2f6_A1a2_A2a2'. Missing fields become None."""
    def grab(pat):
        m = re.search(pat, name)
        return int(m.group(1)) if m else None
    return {
        "exp_name": name,
        "N": grab(r"N(\d+)"),
        "freq_slot": grab(r"W1f(\d+)"),
        "ampli_slot": grab(r"A1a(\d+)"),
    }


def iter_trials(datafile_dir: Path):
    """Yield (exp_name, trial_tid, trial_dir, pos_csv) for every trial found."""
    for exp_dir in sorted(p for p in datafile_dir.iterdir() if p.is_dir()):
        for trial_dir in sorted(p for p in exp_dir.iterdir()
                                if p.is_dir() and p.name.startswith("trial_")):
            m = re.search(r"trial_(\d+)", trial_dir.name)
            if not m:
                continue
            #yield exp_dir.name, int(m.group(1)), trial_dir, trial_dir / (trial_dir.name + POS_ALL_SUFFIX)
            yield exp_dir.name, int(m.group(1)), trial_dir, trial_dir / ("trial_0000" + POS_ALL_SUFFIX)


# =============================================================================
# Per-trial analysis
# =============================================================================

def analyze_trial(pos_csv: Path, trial_dir: Path, last_n_frames: int) -> dict:
    """Run the grouping analysis for one trial; write per-trial CSVs (+ PNG).
    Returns the one-row steady-state record (as a dict)."""
    groups_df = process_pos_all_groups(str(pos_csv), MAX_DIST, INCLUDE_SINGLETONS)
    frame_df  = compute_frame_summary(groups_df)
    an_df     = compute_alignment_vs_size(groups_df, last_n_frames, INCLUDE_SINGLETONS)
    steady    = steady_state_summary(frame_df, last_n_frames)

    groups_df.to_csv(trial_dir / f"{OUT_PREFIX}_groups.csv", index=False)
    frame_df.to_csv(trial_dir / f"{OUT_PREFIX}_frame_summary.csv", index=False)
    an_df.to_csv(trial_dir / f"{OUT_PREFIX}_alignment_vs_size.csv", index=False)
    steady.to_csv(trial_dir / f"{OUT_PREFIX}_steady_state.csv", index=False)

    if PLOT_AN:
        plot_alignment_vs_size(
            an_df, str(trial_dir / f"{OUT_PREFIX}_alignment_vs_size.png"),
            title=f"{trial_dir.parent.name}/{trial_dir.name}  "
                  f"(last {LAST_SECONDS:g}s, max_dist={MAX_DIST:g})")

    return steady.iloc[0].to_dict()


# =============================================================================
# Main batch
# =============================================================================

def run_batch(root: Path):
    datafile = root / "datafile"
    if not datafile.is_dir():
        raise FileNotFoundError(f"datafile dir not found: {datafile}")

    base_dir = str(datafile)                  # render_group_video expects this
    group_video_dir = root / "group_videos"   # outputs kept separate from originals
    last_n_frames = int(LAST_SECONDS * FPS) if LAST_SECONDS > 0 else None

    trials = list(iter_trials(datafile))
    print(f"Found {len(trials)} trials under {datafile}")
    print(f"max_dist={MAX_DIST}  fps={FPS}  window={last_n_frames} frames  "
          f"singletons={'kept' if INCLUDE_SINGLETONS else 'dropped'}\n")

    summary_rows = []
    n_ok = n_skip = n_err = n_vid = 0
    t0 = time.time()

    for k, (exp_name, trial_tid, trial_dir, pos_csv) in enumerate(trials, 1):
        tag = f"[{k}/{len(trials)}] {exp_name}/trial_{trial_tid:04d}"

        if not pos_csv.is_file():
            print(f"{tag}  SKIP (no {pos_csv.name})")
            n_skip += 1
            continue

        try:
            rec = analyze_trial(pos_csv, trial_dir, last_n_frames)
            rec.update(parse_exp_name(exp_name))
            rec["trial_tid"] = trial_tid
            summary_rows.append(rec)
            n_ok += 1
            print(f"{tag}  ok  "
                  f"n_groups={rec.get('n_groups_mean'):.2f}  "
                  f"A_sw={rec.get('align_size_weighted_mean'):.3f}")
        except Exception as e:
            n_err += 1
            print(f"{tag}  ERROR analysis: {e}")
            traceback.print_exc()
            continue

        # Render the group video only for the designated trial id.
        if RENDER_VIDEOS and trial_tid == VIDEO_TRIAL_ID:
            out_path = group_video_dir / exp_name / f"trial_{trial_tid:04d}_group.mp4"
            # Match the background video BY TRIAL NAME (trial_0002 -> trial_0002.mp4).
            video_path = root / "videos" / exp_name / f"trial_{trial_tid:04d}.mp4"
            try:
                render_group_video(
                    exp_name=exp_name,
                    trial_tid=trial_tid,
                    base_dir=base_dir,
                    out_path=str(out_path),
                    max_dist=MAX_DIST,
                    fps=int(FPS),
                    video_stride=VIDEO_STRIDE,
                    include_singletons=INCLUDE_SINGLETONS,
                    video_path=str(video_path),
                )
                n_vid += 1
            except FileNotFoundError as e:
                print(f"{tag}  video skipped: {e}")
            except Exception as e:
                print(f"{tag}  ERROR video: {e}")
                traceback.print_exc()

    # Aggregate summary (one row per trial), ordered for readability.
    if summary_rows:
        df = pd.DataFrame(summary_rows)
        lead = ["exp_name", "trial_tid", "freq_slot", "ampli_slot", "N", "n_frames"]
        cols = [c for c in lead if c in df.columns] + \
               [c for c in df.columns if c not in lead]
        df = df[cols].sort_values(["freq_slot", "ampli_slot", "trial_tid"])
        out_csv = root / SUMMARY_NAME
        df.to_csv(out_csv, index=False)
        print(f"\nSummary ({len(df)} rows) -> {out_csv}")
    else:
        print("\nNo successful trials; summary not written.")

    dt = time.time() - t0
    print(f"\nDone in {dt:.1f}s  |  ok={n_ok}  skipped={n_skip}  "
          f"errors={n_err}  videos={n_vid}")


def main():
    ap = argparse.ArgumentParser(description="Batch group-level alignment analysis over a sweep tree")
    ap.add_argument("--root", default=ROOT, help="sweep root containing datafile/ and videos/")
    ap.add_argument("--no_video", action="store_true", help="disable video rendering")
    args = ap.parse_args()

    if args.no_video:
        global RENDER_VIDEOS
        RENDER_VIDEOS = False

    run_batch(Path(args.root))


if __name__ == "__main__":
    main()

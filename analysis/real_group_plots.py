"""
real_group_plots.py
-------------------
The `groupsize` / `mean_groupsize` figures of batch_group_plots.py, drawn from
hardware runs instead of simulation trials.

A hardware experiment directory holds one trial per stamp, flat rather than in
trial_XXXX/ subfolders:

    11_12/
        group1112_185734.csv      one row per group per detected frame
        command1112_185734.csv    the preset command sequence with wall-clock times
        config1112_185734.txt
        ...

Usage:
    python analysis/real_group_plots.py D:/UMich-Smarticles/videos/1105_sim2real_exp/11_12
    python analysis/real_group_plots.py <dir> --out <dir>/plots --formats png svg
    python analysis/real_group_plots.py <dir> --window all     # keep every frame

Output (default <dir>/group_plots/):
    <stamp>_groupsize.png          per trial, same figure as the simulation's
    <exp>_mean_groupsize.png       across trials: thin = trial, bold = mean, band = ±1 std
    <exp>_trial_stats.csv          scalar stats per trial
    <exp>_mean_timeseries.csv      mean/std of each quantity on the common time axis

What differs from the simulation side, and why
==============================================
* Time. The group csv has no fixed fps (recognition runs at ~6 Hz with jitter),
  so seconds come from the Time column, not Step / fps.
* t = 0. In simulation t=0 is the first frame after warm-up, i.e. the moment
  the gait under study starts. The hardware equivalent is the second row of
  command<stamp>.csv (the first is the a4 "settle" command a couple of seconds
  before), and the run ends at the last row (the closing a4). With
  --window command (default) only frames inside that window are kept and t=0 is
  the main command; that is what makes trials, and hardware vs simulation,
  line up. --window all keeps everything with t=0 at the first frame, as the
  hardware-side plot_group_evolution.py does.
* Detection gaps. Undetected robots are simply absent from a frame.
  trial_series normalises the per-robot mean group size by the robots seen in
  that frame, so a missing robot does not dilute it; it can, however, break a
  chain in two if it was the link -- a real limitation of the input, not
  something this script can repair.

Grouping is taken as recorded (GROUP_MAX_DIST in the config snapshot, in
pixels); nothing is regrouped here.
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_group_plots as bgp                       # noqa: E402

# Only the groupsize figures are drawn, which never read the size-bin shares;
# the bins still have to exist for trial_series/align_series. These are the
# hardware script's bins (20 robots).
BINS = [(1, 1), (2, 3), (4, 5), (6, 8), (9, 99)]
LABELS = ["1", "2-3", "4-5", "6-8", "9+"]


def find_real_trials(exp_dir):
    """-> [(stamp, group_csv, command_csv or None), ...] sorted by stamp."""
    out = []
    for g in sorted(glob.glob(os.path.join(exp_dir, "group*.csv"))):
        stamp = os.path.splitext(os.path.basename(g))[0][len("group"):]
        c = os.path.join(exp_dir, f"command{stamp}.csv")
        out.append((stamp, g, c if os.path.isfile(c) else None))
    return out


def command_window(cmd_csv):
    """
    -> (t_start, t_end, main_command) from the command log, or None.

    Rows are (index, one command per robot..., Time). The main command is the
    second row; the window closes at the last row when there are at least 3.
    """
    if cmd_csv is None:
        return None
    c = pd.read_csv(cmd_csv)
    if len(c) < 2 or "Time" not in c.columns:
        return None
    t = pd.to_datetime(c["Time"])
    robot_cols = [k for k in c.columns if re.fullmatch(r"\d+", str(k))]
    main = c.loc[1, robot_cols].astype(str).unique()
    main = main[0] if len(main) == 1 else "/".join(main)
    t_end = t.iloc[-1] if len(c) >= 3 else None
    return t.iloc[1], t_end, f"a{main}" if str(main).isdigit() else str(main)


def real_trial_series(group_csv, cmd_csv, window="command"):
    """
    trial_series() on a hardware group csv, with seconds from the Time column.
    -> (series dict, info dict)
    """
    df = pd.read_csv(group_csv, usecols=["Step", "Time", "size", "alignment"])
    df["Time"] = pd.to_datetime(df["Time"])
    info = {"command": "", "t0_source": "first frame"}

    win = command_window(cmd_csv) if window == "command" else None
    if win is not None:
        t0, t_end, info["command"] = win
        keep = df["Time"] >= t0
        if t_end is not None:
            keep &= df["Time"] <= t_end
        df = df[keep]
        info["t0_source"] = f"command {info['command']}"
    else:
        if window == "command":
            print(f"    [note] {os.path.basename(group_csv)}: no usable command "
                  f"log, t=0 is the first frame")
        t0 = df["Time"].min()

    # fps is a dummy here: trial_series only uses it to build secs, which is
    # replaced below by each Step's own timestamp
    sr = bgp.trial_series(df, BINS, fps=1.0)
    step_time = df.groupby("Step")["Time"].first().sort_index()
    sr["secs"] = (step_time - t0).dt.total_seconds().to_numpy()
    return sr, info


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+",
                    help="hardware experiment directories holding group<stamp>.csv")
    ap.add_argument("--out", default=None,
                    help="output directory (default <exp>/group_plots)")
    ap.add_argument("--window", choices=("command", "all"), default="command",
                    help="command: keep the main-command window, t=0 at its start "
                         "(default); all: every frame, t=0 at the first frame")
    ap.add_argument("--smooth-s", type=float, default=2.0,
                    help="moving-average window in seconds (0 disables)")
    ap.add_argument("--steady-s", type=float, default=100.0,
                    help="length of the tail averaged into the *_steady stats")
    ap.add_argument("--figs", nargs="+", default=["groupsize", "mean_groupsize"],
                    choices=("groupsize", "mean_groupsize"))
    ap.add_argument("--formats", nargs="+", default=["png"],
                    choices=list(bgp.FIG_FORMATS))
    ap.add_argument("--font", default=None)
    ap.add_argument("--title", action="store_true",
                    help="append [exp/trial] to figure titles")
    args = ap.parse_args()

    bgp.OUTPUT_FORMATS = list(args.formats)     # save_fig reads the module global
    bgp.choose_font(args.font)

    for exp_dir in args.experiments:
        exp_name = os.path.basename(os.path.normpath(exp_dir))
        trials = find_real_trials(exp_dir)
        print(f"[{exp_name}] {len(trials)} trials")
        if not trials:
            continue
        out_dir = args.out or os.path.join(exp_dir, "group_plots")
        os.makedirs(out_dir, exist_ok=True)

        all_sr, rows = [], []
        for stamp, gcsv, ccsv in trials:
            sr, info = real_trial_series(gcsv, ccsv, args.window)
            if len(sr["secs"]) == 0:
                print(f"    [skip] {stamp}: no frames in the window")
                continue
            all_sr.append(sr)
            tail = sr["secs"] >= sr["secs"][-1] - args.steady_s
            rows.append({
                "exp": exp_name, "trial": stamp, "command": info["command"],
                "t0": info["t0_source"],
                "frames": len(sr["secs"]),
                "duration_s": float(sr["secs"][-1]),
                "fps_mean": (len(sr["secs"]) - 1) / max(sr["secs"][-1] - sr["secs"][0], 1e-9),
                "n_seen_median": float(np.median(sr["n_seen"])),
                "n_seen_max": int(sr["n_seen"].max()),
                "largest_mean": float(sr["largest"].mean()),
                "largest_steady": float(sr["largest"][tail].mean()),
                "largest_peak": int(sr["largest"].max()),
                "mean_size_mean": float(sr["mean_size"].mean()),
                "mean_size_steady": float(sr["mean_size"][tail].mean()),
                "alignment_steady": float(sr["alignment"][tail].mean()),
            })
            r = rows[-1]
            print(f"    {stamp} ({r['command'] or 'no command log'}): "
                  f"{r['frames']} frames / {r['duration_s']:.1f}s, "
                  f"~{r['fps_mean']:.1f} fps, robots seen median {r['n_seen_median']:g}; "
                  f"largest mean {r['largest_mean']:.2f} / last {args.steady_s:g}s "
                  f"{r['largest_steady']:.2f} / peak {r['largest_peak']}; "
                  f"mean size last {args.steady_s:g}s {r['mean_size_steady']:.2f}")
            if "groupsize" in args.figs:
                title = f"  [{exp_name}/{stamp}]" if args.title else ""
                fp = os.path.join(out_dir, f"{stamp}_groupsize.png")
                bgp.fig_groupsize(sr, fp, title, args.smooth_s)
                print(f"      -> {bgp.written_name(fp)}")

        if not all_sr:
            continue
        stats = pd.DataFrame(rows)
        sp = os.path.join(out_dir, f"{exp_name}_trial_stats.csv")
        stats.to_csv(sp, index=False)
        print(f"    -> {os.path.basename(sp)}")

        grid, stacked, t_end = bgp.align_series(all_sr)
        durs = [float(s["secs"][-1]) for s in all_sr]
        if max(durs) - min(durs) > 1e-6:
            print(f"    [note] durations {min(durs):.1f}~{max(durs):.1f}s, "
                  f"cross-trial average truncated to {t_end:.1f}s")
        ts = pd.DataFrame({"time": grid})
        for k in ("largest", "mean_size", "alignment", "n_groups", "n_multi",
                  "n_seen", "largest_frac"):
            ts[f"{k}_mean"] = stacked[k].mean(axis=0)
            ts[f"{k}_std"] = stacked[k].std(axis=0)
        tp = os.path.join(out_dir, f"{exp_name}_mean_timeseries.csv")
        ts.to_csv(tp, index=False)
        print(f"    -> {os.path.basename(tp)}")

        if "mean_groupsize" in args.figs:
            cmds = sorted({r["command"] for r in rows if r["command"]})
            title = (f"  [{exp_name}" + (f", {'/'.join(cmds)}" if cmds else "")
                     + "]") if args.title else ""
            fp = os.path.join(out_dir, f"{exp_name}_mean_groupsize.png")
            bgp.fig_mean_groupsize(grid, stacked, fp, title, args.smooth_s)
            print(f"    -> {bgp.written_name(fp)}")

        print(f"    across trials (n={len(stats)}): largest last {args.steady_s:g}s "
              f"{stats['largest_steady'].mean():.2f} ± {stats['largest_steady'].std(ddof=0):.2f}, "
              f"mean size last {args.steady_s:g}s "
              f"{stats['mean_size_steady'].mean():.2f} ± {stats['mean_size_steady'].std(ddof=0):.2f}")


if __name__ == "__main__":
    main()

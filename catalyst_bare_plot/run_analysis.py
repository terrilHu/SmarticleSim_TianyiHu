"""
run_analysis.py
---------------
Compare baseline experiments against their strategy experiments, trial by trial
(same trial names, same initial conditions). A pair is <base> in --baseline-root
against <base>_<suffix> in --strategy-root. With no options this regenerates
videos/1105_catalyst_amount_xend/datafile/hypotheses_compare_xend_36.

Steps, in order (--steps picks a subset):

    tracks      group_lifetimes.py on any experiment with no pooled tracks yet
                (the only step that writes into the input folders)
    compare     compare_<base>_vs_<strat>_lifetime_d65_m3_w1.csv/.png
    hypotheses  hypotheses_<base>_vs_<strat>_last100*, plus _t50* for --t50-bases
    overview    hypotheses_overview_last100.csv/.png
    bars        composition_delta_*_last100[_bare].csv/.png/.svg, three pairs per chart
    snapshots   snapshots_<trial>/<base>_vs_<strat>/, for --snapshot-bases
    ic          snapshots_<trial>/IC<k>_initial[_annotated].png

Every trial needs trial_*/trial_*_groups_d65.csv from batch_group_plots.py (see
README.md); inputs are checked before anything runs.

Usage (conda env smarticle-3.11, for pygame):
    python run_analysis.py
    python run_analysis.py --steps bars overview
    python run_analysis.py --baseline-root D:/data/base --strategy-root D:/data/strat --suffix xend_12
    python run_analysis.py --bases a016 a034 --out-root D:/results/test
"""

import argparse
import ctypes
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))

# Defaults: the runs behind hypotheses_compare_xend_36. All can be set on the command line.
VIDEOS = r"D:\UMich-Smarticles\videos"
BASELINE_ROOT = os.path.join(VIDEOS, "1029_test_catalyst", "datafile")
STRATEGY_ROOT = os.path.join(VIDEOS, "1105_catalyst_amount_xend", "datafile")
SUFFIX = "xend_36"
BASES = ["a016", "a034", "a062", "a816", "a834", "a862"]
T50_BASES = ["a834"]
SNAPSHOT_TRIAL = "trial_0002"
YLIM = ["-30", "30"]                # shared y range of the composition bars

TAG = "d65_m3_w1"                   # max_dist 65, min group size 3, 1 s hysteresis
PAIRS_PER_CHART = 3
MIN_FREE_RAM_GB, MIN_FREE_DISK_GB = 1.5, 5.0

STEPS = ["tracks", "compare", "hypotheses", "overview", "bars", "snapshots", "ic"]


def free_ram_gb():
    class Mem(ctypes.Structure):
        _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
            (n, ctypes.c_ulonglong) for n in ("tp", "ap", "tpf", "apf", "tv", "av", "ae")]
    m = Mem()
    m.length = ctypes.sizeof(Mem)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ap / 2**30


def run(args, script, *script_args):
    """Run a sibling script, stopping first if RAM or disk is short and on any failure."""
    ram, disk = free_ram_gb(), shutil.disk_usage(args.out_root).free / 2**30
    print(f"\n>>> {script} {' '.join(script_args)}\n    free RAM {ram:.1f} GB, disk {disk:.0f} GB", flush=True)
    if ram < MIN_FREE_RAM_GB or disk < MIN_FREE_DISK_GB:
        raise SystemExit("stopping: not enough free RAM or disk")
    p = subprocess.run([sys.executable, script, *script_args], cwd=HERE,
                       env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=subprocess.PIPE,
                       stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
    print("\n".join(p.stdout.splitlines()[-4:]), flush=True)
    if p.returncode:
        print(p.stdout[-3000:])
        raise SystemExit(f"{script} failed ({p.returncode})")


def strat(args, base):
    return f"{base}_{args.suffix}"


def base_dir(args, base):
    return os.path.join(args.baseline_root, base)


def strat_dir(args, base):
    return os.path.join(args.strategy_root, strat(args, base))


def pair_dir(args, base):
    return os.path.join(args.out_root, f"{base}_vs_{strat(args, base)}")


def snapshot_dir(args):
    return os.path.join(args.out_root, f"snapshots_{args.snapshot_trial}")


def trial_names(exp_dir):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(exp_dir, "trial_*")) if os.path.isdir(d))


def check_inputs(args):
    """Report every missing input at once, instead of failing partway through a long run."""
    problems = []
    for b in args.bases:
        dirs = (base_dir(args, b), strat_dir(args, b))
        for d in dirs:
            if not os.path.isdir(d):
                problems.append(f"missing experiment folder: {d}")
                continue
            if not os.path.isfile(os.path.join(d, "config_snapshot.json")):
                problems.append(f"missing config_snapshot.json: {d}")
            trials = trial_names(d)
            if not trials:
                problems.append(f"no trial_* folders: {d}")
            for t in trials:
                # by pattern: a trial merged by merge_batches.py keeps its files' original names
                if "hypotheses" in args.steps and not glob.glob(os.path.join(d, t, "*_groups_d65.csv")):
                    problems.append(f"missing groups cache, run batch_group_plots.py: {os.path.join(d, t)}")
        if all(os.path.isdir(d) for d in dirs) and trial_names(dirs[0]) != trial_names(dirs[1]):
            problems.append(f"trial names differ: {b} {trial_names(dirs[0])} vs {strat(args, b)} {trial_names(dirs[1])}")
    if "snapshots" in args.steps:
        for b in args.snapshot_bases:
            for d in (base_dir(args, b), strat_dir(args, b)):
                if not os.path.isdir(os.path.join(d, args.snapshot_trial)):
                    problems.append(f"missing {args.snapshot_trial}: {d}")
                if not os.path.isfile(os.path.join(d, f"{os.path.basename(d)}_trial_stats.csv")):
                    problems.append(f"missing {os.path.basename(d)}_trial_stats.csv: {d}")
    if problems:
        raise SystemExit("input check failed:\n  " + "\n  ".join(problems))


def step_tracks(args):
    for d in [base_dir(args, b) for b in args.bases] + [strat_dir(args, b) for b in args.bases]:
        name = os.path.basename(d)
        if not os.path.exists(os.path.join(d, f"{name}_tracks_{TAG}.csv")):
            run(args, "group_lifetimes.py", d, "--jobs", "2")


def step_compare(args):
    import pandas as pd
    sys.path.insert(0, HERE)
    from group_lifetimes import compare_experiments

    def one(d):
        name = os.path.basename(d)
        return (name, pd.read_csv(os.path.join(d, f"{name}_tracks_{TAG}.csv")),
                pd.read_csv(os.path.join(d, f"{name}_lifetime_trials_{TAG}.csv")))

    for b in args.bases:
        s = strat(args, b)
        os.makedirs(pair_dir(args, b), exist_ok=True)
        out = os.path.join(pair_dir(args, b), f"compare_{b}_vs_{s}_lifetime_{TAG}")
        compare_experiments([one(base_dir(args, b)), one(strat_dir(args, b))], out,
                            f"{b}  vs  {s}  -  min size 3, hysteresis 1 s, d65")
        print("wrote", out, flush=True)


def step_hypotheses(args):
    for b in args.bases:
        os.makedirs(pair_dir(args, b), exist_ok=True)
        run(args, "strategy_hypotheses.py", base_dir(args, b), strat_dir(args, b),
            "--last", "100", "--out-dir", pair_dir(args, b))
    for b in args.t50_bases:     # t >= 50 s window
        if b in args.bases:
            run(args, "strategy_hypotheses.py", base_dir(args, b), strat_dir(args, b), "--out-dir", pair_dir(args, b))


def step_overview(args):
    run(args, "hypotheses_overview.py", args.out_root, "--exp-root", args.strategy_root)


def step_bars(args):
    for i in range(0, len(args.bases), PAIRS_PER_CHART):
        group = args.bases[i:i + PAIRS_PER_CHART]
        common = [args.out_root, *group, "--exp-root", args.strategy_root, "--ylim", *args.ylim]
        run(args, "composition_delta_bars.py", *common)
        run(args, "composition_delta_bars.py", *common, "--bare")


def step_snapshots(args):
    for b in args.snapshot_bases:
        run(args, "render_snapshots.py", base_dir(args, b), strat_dir(args, b),
            "--trial", args.snapshot_trial, "--times", "0", "300", "500",
            "--out-dir", os.path.join(snapshot_dir(args), f"{b}_vs_{strat(args, b)}"))


def step_ic(args):
    """Draw the initial condition the snapshot trial started from, as its config records it."""
    if not args.snapshot_bases:
        return
    like = strat_dir(args, args.snapshot_bases[0])
    cfg = json.load(open(os.path.join(like, "config_snapshot.json"), encoding="utf-8"))
    index = cfg["INIT_INDICES"][int(args.snapshot_trial.split("_")[1])]
    with tempfile.TemporaryDirectory() as work:
        run(args, "render_ic.py", args.ic_file or cfg["INIT_FILE"], str(index),
            "--like", os.path.join(like, args.snapshot_trial), "--out-dir", snapshot_dir(args), "--work-dir", work)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline-root", default=BASELINE_ROOT, help="folder holding the baseline experiments")
    ap.add_argument("--strategy-root", default=STRATEGY_ROOT, help="folder holding the strategy experiments")
    ap.add_argument("--suffix", default=SUFFIX, help="the strategy experiment of <base> is <base>_<suffix>")
    ap.add_argument("--bases", nargs="+", default=BASES, help="baseline experiment names")
    ap.add_argument("--out-root", default=None,
                    help="output folder (default: <strategy-root>/hypotheses_compare_<suffix>). "
                         "Files of the same name are overwritten")
    ap.add_argument("--t50-bases", nargs="*", default=T50_BASES, help="pairs that also get the t >= 50 s window")
    ap.add_argument("--snapshot-bases", nargs="*", default=None,
                    help="pairs to draw snapshots of (default: the first three of --bases)")
    ap.add_argument("--snapshot-trial", default=SNAPSHOT_TRIAL)
    ap.add_argument("--ic-file", default=None,
                    help="initial-conditions JSON (default: INIT_FILE in the strategy's config_snapshot.json)")
    ap.add_argument("--ylim", nargs=2, default=YLIM, metavar=("LO", "HI"),
                    help="y range of the composition bars. Bars beyond it are cut off")
    ap.add_argument("--steps", nargs="+", choices=STEPS, default=STEPS)
    args = ap.parse_args()

    args.baseline_root = os.path.abspath(args.baseline_root)
    args.strategy_root = os.path.abspath(args.strategy_root)
    args.out_root = os.path.abspath(args.out_root or os.path.join(args.strategy_root,
                                                                  f"hypotheses_compare_{args.suffix}"))
    if args.snapshot_bases is None:
        args.snapshot_bases = args.bases[:PAIRS_PER_CHART]
    check_inputs(args)
    os.makedirs(args.out_root, exist_ok=True)
    print(f"baseline root  {args.baseline_root}\nstrategy root  {args.strategy_root}\noutput         {args.out_root}",
          flush=True)
    for name in STEPS:      # always in pipeline order, however --steps was typed
        if name in args.steps:
            globals()[f"step_{name}"](args)
    print("\nALL DONE", flush=True)


if __name__ == "__main__":
    main()

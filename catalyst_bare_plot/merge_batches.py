"""
merge_batches.py
----------------
Combine several batches of the same experiment (e.g. ICs 11-15 in one folder,
ICs 16-20 in another, each with trial_0000..trial_0004) into one experiment
folder with renumbered trials, so run_analysis.py can treat them as one
experiment of 10 trials.

No data is copied. In <out>/<exp>/:

    trial_0000 .. trial_NNNN   directory junctions to the batches' trial folders,
                               numbered in batch order (batch 1's trials first)
    <exp>_trial_stats.csv      the batches' tables, `trial` renamed
    <exp>_tracks_<tag>.csv     the batches' pooled group_lifetimes output, `trial` renamed
    <exp>_lifetime_trials_<tag>.csv   (both skipped, with a note, when a batch has no pooled
                               tracks and --run-tracks is not given)
    config_snapshot.json       batch 1's, with INIT_INDICES concatenated (and INIT_FILE
                               replaced by --init-file if given), plus MERGED_FROM
    _merge_map.csv             new trial -> batch, original trial, IC index, gait-switch mode

The files inside a junctioned trial keep their original names (a trial_0005
folder may hold trial_0000_groups_d65.csv). strategy_hypotheses.py and
render_snapshots.py find files by pattern and take the trial name from the
folder, so that works; group_lifetimes.py does not, so it is run on each batch
in its own folder (--run-tracks) and the pooled tables are merged here.

The junctions need an NTFS drive for <out> (exFAT cannot hold them); the
batches themselves can be on any drive. To delete <out>, remove the junctions
with `rmdir` (cmd) or os.rmdir, never with a recursive delete, which could
follow them into the original data.

Usage:
    python merge_batches.py --out C:/merged/base --batches E:/1107_baseline/sync E:/1107_baseline/16-20 \
        --exps a816 a834 a862 [--run-tracks] [--init-file E:/.../init_conditions_200_p_N500.json]
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TAG = "d65_m3_w1"
# config keys that legitimately differ between batches of one experiment
EXPECTED_DIFF = {"INIT_FILE", "INIT_INDICES", "OUT_ROOT", "_snapshot", "_skipped", "COMMAND_ARRAY",
                 "RECORD_VIDEO", "SNAPSHOT_DIVISIONS"}


def trials_of(exp_dir):
    return sorted(os.path.basename(d) for d in glob.glob(os.path.join(exp_dir, "trial_*")) if os.path.isdir(d))


def junction(link, target):
    r = subprocess.run(["cmd", "/c", "mklink", "/J", link, target], capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"mklink /J failed for {link} -> {target}: {r.stdout}{r.stderr}")


def merge_exp(exp, batches, out, run_tracks, init_file):
    srcs = [os.path.join(b, exp) for b in batches]
    for s in srcs:
        if not os.path.isfile(os.path.join(s, "config_snapshot.json")):
            raise SystemExit(f"missing {s}\\config_snapshot.json")
    dst = os.path.join(out, exp)
    if os.path.exists(dst) and os.listdir(dst):
        raise SystemExit(f"{dst} already exists and is not empty; remove it first (rmdir the junctions)")

    with_tracks = True
    for s in srcs:
        if not os.path.isfile(os.path.join(s, f"{exp}_tracks_{TAG}.csv")) or \
           not os.path.isfile(os.path.join(s, f"{exp}_lifetime_trials_{TAG}.csv")):
            if not run_tracks:
                # enough for analyses that read only the groups caches (e.g. the catalyst dose curves);
                # run_analysis.py's compare/hypotheses steps need the merged tracks
                print(f"  NOTE {exp}: {s} has no pooled tracks; merged tracks tables skipped "
                      "(use --run-tracks for run_analysis.py)", flush=True)
                with_tracks = False
                continue
            print(f">>> group_lifetimes.py {s}", flush=True)
            subprocess.run([sys.executable, "group_lifetimes.py", s, "--jobs", "2"], cwd=HERE, check=True,
                           env=dict(os.environ, PYTHONIOENCODING="utf-8"), stdout=subprocess.DEVNULL)

    cfgs = [json.load(open(os.path.join(s, "config_snapshot.json"), encoding="utf-8")) for s in srcs]
    for s, c in zip(srcs[1:], cfgs[1:]):
        ignore = set(EXPECTED_DIFF)
        if not c.get("ENABLE_RUNTIME_GAIT_CONTROL") and not cfgs[0].get("ENABLE_RUNTIME_GAIT_CONTROL"):
            ignore |= {k for k in set(c) | set(cfgs[0]) if k.startswith("RUNTIME_GAIT_")}   # unused when off
        diff = sorted(k for k in set(c) | set(cfgs[0]) if c.get(k) != cfgs[0].get(k) and k not in ignore)
        if diff:
            print(f"  NOTE {exp}: {s} differs from batch 1 in " +
                  ", ".join(f"{k} ({cfgs[0].get(k)} vs {c.get(k)})" for k in diff), flush=True)

    os.makedirs(dst, exist_ok=True)
    rows, k = [], 0
    for s, c in zip(srcs, cfgs):
        for t in trials_of(s):
            new = f"trial_{k:04d}"
            junction(os.path.join(dst, new), os.path.join(s, t))
            idx = c.get("INIT_INDICES")
            rows.append(dict(trial=new, batch=os.path.dirname(s), src_trial=t,
                             init_index=idx[int(t.split("_")[1])] if idx else None,
                             init_file=c.get("INIT_FILE"),
                             gait_switch_mode=c.get("RUNTIME_GAIT_SWITCH_MODE")))
            k += 1
    mp = pd.DataFrame(rows)
    mp.to_csv(os.path.join(dst, "_merge_map.csv"), index=False)
    rename = {(r.batch, r.src_trial): r.trial for r in mp.itertuples()}

    tables = ["_trial_stats.csv"] + ([f"_tracks_{TAG}.csv", f"_lifetime_trials_{TAG}.csv"] if with_tracks else [])
    for suffix in tables:
        parts = []
        for s in srcs:
            p = os.path.join(s, exp + suffix)
            if not os.path.isfile(p):
                raise SystemExit(f"missing {p}")
            t = pd.read_csv(p)
            t["trial"] = [rename[(os.path.dirname(s), x)] for x in t["trial"]]
            parts.append(t)
        pd.concat(parts, ignore_index=True).to_csv(os.path.join(dst, exp + suffix), index=False)

    cfg = dict(cfgs[0])
    cfg["INIT_INDICES"] = [r.init_index for r in mp.itertuples()]
    if init_file:
        cfg["INIT_FILE"] = init_file
    cfg["MERGED_FROM"] = srcs
    with open(os.path.join(dst, "config_snapshot.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"{exp}: {k} trials from {len(srcs)} batches -> {dst}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="merged root, on an NTFS drive")
    ap.add_argument("--batches", nargs="+", required=True, help="batch roots, in trial order")
    ap.add_argument("--exps", nargs="+", required=True, help="experiment folder names present in every batch")
    ap.add_argument("--run-tracks", action="store_true",
                    help="run group_lifetimes.py on any batch experiment without pooled tracks "
                         "(writes its usual cache files into that batch folder)")
    ap.add_argument("--init-file", default=None,
                    help="INIT_FILE for the merged config (when the batches' IC files are the same file)")
    args = ap.parse_args()
    out = os.path.abspath(args.out)
    os.makedirs(out, exist_ok=True)
    batches = [os.path.abspath(b) for b in args.batches]
    for e in args.exps:
        merge_exp(e, batches, out, args.run_tracks, args.init_file)


if __name__ == "__main__":
    main()

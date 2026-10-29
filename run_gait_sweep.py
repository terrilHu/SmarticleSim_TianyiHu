"""
run_gait_sweep.py  -  the production gait sweep.

One condition per gait pattern, 10 trials each, all from the same ten initial
conditions, no strategy. Which patterns are included is decided by the pilot,
not by hand: pilot_summary.csv measured what the arms actually did, and a
pattern is only run if the delivered amplitude and frequency were within
--tol of what was commanded.

That filter is the point of the pilot. The actuator cannot track large
amplitudes at high frequency -- at 90 deg / 4 Hz the arms reach 27 deg, 30% of
the setpoint -- so those cells of the grid are not "90 deg at 4 Hz" at all, and
including them would put a fictitious amplitude axis into the results. At a 7%
tolerance 23 of the 48 patterns drop out, leaving a staircase along
amplitude x frequency = const, which is the bandwidth limit.

    python run_gait_sweep.py --pilot datafile/pilot_gait/pilot_summary.csv \
        --out datafile/1014_sweep --list
    python run_gait_sweep.py --pilot ... --out datafile/1014_sweep

Each condition writes to <out>/<pattern>/trial_XXXX/, which is the layout
batch_group_plots.py and the scaling scripts already expect.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

DEFAULT_INIT = ("D:/UMich-Smarticles/videos/1014_sweep/init_conditions/"
                "init_conditions_200_p_N100.json")
DEFAULT_PILOT = os.path.join("datafile", "pilot_gait", "pilot_summary.csv")


def select_patterns(pilot_csv, tol):
    """
    -> (kept, dropped) as lists of dicts, from the pilot's measured fidelity.

    Keeps a pattern when BOTH the amplitude and the frequency came out within
    tol of the command. Amplitude is the one that actually bites; the frequency
    check is here because a pattern that failed to run at all would show up in
    it, not because frequency is ever marginally off.
    """
    import pandas as pd
    d = pd.read_csv(pilot_csv)
    need = {"pattern", "command", "ampli_ratio", "freq_ratio"}
    missing = need - set(d.columns)
    if missing:
        raise ValueError(f"{pilot_csv} is missing {sorted(missing)}; "
                         f"is it a pilot_summary.csv?")
    bad = ((d.ampli_ratio - 1).abs() > tol) | ((d.freq_ratio - 1).abs() > tol)
    bad |= ~d.ampli_ratio.notna()
    return d[~bad].to_dict("records"), d[bad].to_dict("records")


def retarget(kept, x_slot):
    """
    Keep the Y/Z cells the pilot certified, but run them at a different
    initial-phase slot X.

    X only sets the phase every robot starts its cycle at; it has nothing to do
    with the amplitude and frequency the pilot measured, so the selection
    carries over. X=8 starts the whole swarm in phase, X=0 gives each robot a
    random phase -- same gait, synchronised or not.

    What does NOT carry over is the fidelity itself. The pilot measured
    delivered amplitude with the swarm in phase; desynchronised, the arms fight
    each other instead of moving together, so the joint load differs and the
    bandwidth edge may sit elsewhere. The numbers echoed in the plan below are
    the X=8 measurements, kept only to show which cells were chosen and why.
    """
    out = []
    for k in kept:
        k = dict(k)
        y, z = int(k["Y"]), int(k["Z"])
        k["command"] = x_slot * 100 + 10 * y + z
        k["pattern"] = f"a{x_slot}{y}{z}"
        out.append(k)
    return out


def write_condition(path, pattern, command, out_root, args):
    body = {
        "N_SMARTICLES": 100,
        "COMMAND_ARRAY": f"a{int(command)}",
        "MAX_RUNTIME": float(args.seconds),
        "N_TRIALS_GLOBAL": len(args.ic_indices),
        "ALREADY_SPWANED": True,
        "INIT_FILE": args.init_file,
        # The same ten layouts for every pattern: a paired design, so a
        # difference between patterns is not a difference between layouts.
        "INIT_SELECTION": "explicit",
        "INIT_INDICES": list(args.ic_indices),
        # No strategy -- every robot holds its commanded gait for the whole run.
        # The probe below is not one: it returns no requests and changes no
        # dynamics, it only records what the joints did.
        "ENABLE_RUNTIME_GAIT_CONTROL": bool(args.probe),
        "RUNTIME_GAIT_CONTROLLER": "gait_probe:measure",
        # Record trial 0 only. Not RECORD_POLICY="first_n": that path tests
        # trial_id <= RECORD_FIRST_N on a 0-based id, so 1 records two trials
        # and 0 records none. "mod" with K = n_trials hits only trial 0.
        "RECORD_VIDEO": True,
        "RECORD_POLICY": "mod",
        "RECORD_EVERY_K": max(1, len(args.ic_indices)),
        "PREVIEW": False,
        "PARALLEL_WORKERS": int(args.workers),
        "EXP_NAME": pattern,
        "OUT_ROOT": out_root,
        "RING_COLOR_BANDS": 0,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return path


def done_trials(out_root, pattern):
    return len(glob.glob(os.path.join(out_root, pattern, "trial_*",
                                      "*_POS_ALL.csv")))


def main():
    ap = argparse.ArgumentParser(
        description="Production gait sweep: the pilot decides which patterns run.")
    ap.add_argument("--pilot", default=DEFAULT_PILOT,
                    help="pilot_summary.csv that decides which patterns are usable")
    ap.add_argument("--out", default=os.path.join("datafile", "1014_sweep"),
                    help="Output root; one subfolder per pattern")
    ap.add_argument("--init-file", default=DEFAULT_INIT,
                    help="Initial-conditions json every condition draws from")
    ap.add_argument("--ic-from", type=int, default=31,
                    help="First initial-condition index (inclusive)")
    ap.add_argument("--ic-to", type=int, default=40,
                    help="Last initial-condition index (inclusive)")
    ap.add_argument("--seconds", type=float, default=500.0,
                    help="Trial length, matching the earlier scaling runs")
    ap.add_argument("--x-slot", type=int, default=8, choices=list(range(0, 9)),
                    metavar="X",
                    help="Initial-phase slot for every pattern. 8 = the whole "
                         "swarm starts in phase (what the pilot ran); 0 = each "
                         "robot gets a random phase, i.e. desynchronised. The "
                         "pilot's Y/Z selection is reused either way")
    ap.add_argument("--tol", type=float, default=0.07,
                    help="Allowed fractional error on delivered amplitude and "
                         "frequency; 0.07 drops 23 of the 48 patterns")
    ap.add_argument("--workers", type=int, default=1,
                    help="Trials run concurrently within a condition")
    ap.add_argument("--no-probe", dest="probe", action="store_false",
                    help="Do not record joint angles. The probe costs a few MB "
                         "per trial and is what lets the results quote the "
                         "delivered amplitude rather than the commanded one")
    ap.add_argument("--only", nargs="+", default=None, metavar="PAT",
                    help="Run only these patterns")
    ap.add_argument("--list", action="store_true",
                    help="Show the plan and the kept/dropped split, then exit")
    ap.add_argument("--resume", action="store_true",
                    help="Skip conditions that already have all their trials")
    ap.add_argument("--timeout", type=float, default=None,
                    help="Per-condition wall-clock limit in seconds")
    args = ap.parse_args()
    args.ic_indices = list(range(args.ic_from, args.ic_to + 1))

    if not os.path.isfile(args.pilot):
        print(f"No pilot summary at {args.pilot}. Run pilot_gait_sweep.py first "
              f"-- the point of the sweep is that the pilot chooses the patterns.")
        return 1
    if not os.path.isfile(args.init_file):
        print(f"No initial-conditions file at {args.init_file}")
        return 1

    kept, dropped = select_patterns(args.pilot, args.tol)
    if args.x_slot != 8:
        kept = retarget(kept, args.x_slot)
    if args.only:
        want = set(args.only)
        kept = [k for k in kept if k["pattern"] in want]

    out_root = os.path.abspath(args.out)
    cond_dir = os.path.join(out_root, "_conditions")
    probe_dir = os.path.join(out_root, "_probe")
    os.makedirs(cond_dir, exist_ok=True)
    if args.probe:
        os.makedirs(probe_dir, exist_ok=True)

    n_tr = len(args.ic_indices)
    print(f"pilot: {args.pilot}   tolerance {args.tol:.0%}")
    print(f"  kept {len(kept)} patterns, dropped {len(dropped)}")
    if dropped:
        print(f"  dropped: {' '.join(sorted(d['pattern'] for d in dropped))}")
    if args.x_slot != 8:
        print(f"  running them at X={args.x_slot} "
              f"({'random initial phase, desynchronised' if args.x_slot == 0 else 'phase slot ' + str(args.x_slot)}); "
              f"the fidelity figures below are the X=8 pilot's, not re-measured")
    print(f"plan: {len(kept)} conditions x {n_tr} trials x {args.seconds:g}s, "
          f"IC {args.ic_from}..{args.ic_to}, video on trial 0 only, "
          f"probe={'on' if args.probe else 'off'}")
    print(f"      output -> {out_root}")

    todo = []
    for k in kept:
        have = done_trials(out_root, k["pattern"])
        if args.resume and have >= n_tr:
            print(f"  {k['pattern']}: {have}/{n_tr} trials present, skipping")
            continue
        todo.append(k)
    if args.list:
        for k in todo:
            print(f"  will run {k['pattern']}  A={k['cmd_ampli_deg']:.0f}deg "
                  f"f={k['cmd_freq_hz']:g}Hz  "
                  f"(delivered {k['ampli_ratio']:.3f} of commanded)")
        return 0
    if not todo:
        print("Nothing to do.")
        return 0

    t0 = time.time()
    failed = []
    for i, k in enumerate(todo, 1):
        pat = k["pattern"]
        cfg = write_condition(os.path.join(cond_dir, f"{pat}.json"), pat,
                              k["command"], out_root, args)
        env = dict(os.environ)
        env["SMARTICLE_CONFIG"] = cfg
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        env["PYTHONIOENCODING"] = "utf-8"
        if args.probe:
            env["GAIT_PROBE_OUT"] = os.path.join(probe_dir, f"{pat}.csv")
            env["GAIT_PROBE_IDS"] = "6"
            if args.workers > 1:
                env["GAIT_PROBE_SPLIT_PID"] = "1"
        el = time.time() - t0
        eta = el / (i - 1) * (len(todo) - i + 1) if i > 1 else 0.0
        print(f"\n[{i}/{len(todo)}] {pat}  A={k['cmd_ampli_deg']:.0f}deg "
              f"f={k['cmd_freq_hz']:g}Hz   elapsed {el/3600:.1f}h  "
              f"eta {eta/3600:.1f}h", flush=True)
        sys.stdout.flush()
        try:
            r = subprocess.run([sys.executable, "simulation.py"], env=env,
                               cwd=HERE, timeout=args.timeout)
            if r.returncode != 0:
                print(f"    [FAIL] exit {r.returncode}", flush=True)
                failed.append(pat)
        except subprocess.TimeoutExpired:
            print(f"    [TIMEOUT]", flush=True)
            failed.append(pat)

    print(f"\nDone: {len(todo) - len(failed)}/{len(todo)} conditions in "
          f"{(time.time()-t0)/3600:.1f}h")
    if failed:
        print(f"failed: {' '.join(failed)}  (re-run with --resume)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())

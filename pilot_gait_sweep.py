"""
pilot_gait_sweep.py  -  one short trial per gait pattern, to find out which
patterns are actually usable before committing to the full sweep.

Covers a811..a868: X = 8 (initial phase fixed), Y = 1..6 (amplitude 15..90 deg),
Z = 1..8 (frequency 0.5..4.0 Hz) -> 48 patterns. Every run uses the SAME single
initial condition, so anything that differs between runs is the gait and only
the gait.

It answers two questions per pattern:

  1. Was the gait delivered?  The commanded amplitude is a setpoint, not a
     result. SIM_DT is 1/180 s, so a 4 Hz gait gets 45 integration steps per
     cycle, and in a dense swarm the joints are also fighting collisions. If the
     arms never reach the commanded angle then the amplitude axis of the full
     sweep is fiction. Measured with gait_probe, which logs the real joint
     angles -- POS_ALL carries only the body pose and cannot answer this.

  2. Did the run stay sane?  Counts non-finite coordinates and robots outside
     the ring, and reports the spread of the swarm. Blunt checks that catch a
     blown-up trial without prejudging what abnormal means.

Video is on by default: the numbers say whether the gait was tracked, the video
says whether the collective behaviour is the one you wanted, and those are
different questions.

    python pilot_gait_sweep.py --out datafile/pilot_gait --seconds 60
    python pilot_gait_sweep.py --out datafile/pilot_gait --only a811 a868
    python pilot_gait_sweep.py --out datafile/pilot_gait --analyse-only
"""

import argparse
import glob
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
X_SLOT = 8
Y_RANGE = range(1, 7)
Z_RANGE = range(1, 9)


def patterns():
    return [(f"a8{y}{z}", 800 + 10 * y + z, y, z)
            for y in Y_RANGE for z in Z_RANGE]


def _adeg(y):
    from gait import AMPLI_TABLE
    return round(math.degrees(AMPLI_TABLE[y - 1]))


def _fhz(z):
    from gait import FREQ_TABLE
    return FREQ_TABLE[z - 1]


def write_condition(path, cmd_text, exp_name, out_root, seconds, ic_index,
                    video):
    """One override file per pattern; only COMMAND_ARRAY and EXP_NAME differ."""
    body = {
        "N_SMARTICLES": 100,
        "COMMAND_ARRAY": cmd_text,
        "MAX_RUNTIME": float(seconds),
        # One trial, and the SAME initial condition for every pattern -- this is
        # a paired comparison, so the layout must not be a variable.
        "N_TRIALS_GLOBAL": 1,
        "ALREADY_SPWANED": True,
        "INIT_FILE": "init_conditions/init_conditions_200_p.json",
        "INIT_SELECTION": "explicit",
        "INIT_INDICES": [int(ic_index)],
        # The sweep itself is strategy-free; the probe is a passive observer
        # that returns no requests, so it changes nothing about the dynamics.
        "ENABLE_RUNTIME_GAIT_CONTROL": True,
        "RUNTIME_GAIT_CONTROLLER": "gait_probe:measure",
        "RECORD_VIDEO": bool(video),
        "PREVIEW": False,
        "PARALLEL_WORKERS": 1,
        "EXP_NAME": exp_name,
        "OUT_ROOT": out_root,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(body, f, indent=2)
    return path


def sanity(trial_dir, inner_r, cx, cy):
    """Blunt health checks on a finished trial, straight from POS_ALL."""
    import numpy as np
    import pandas as pd
    pa = glob.glob(os.path.join(trial_dir, "*_POS_ALL.csv"))
    if not pa:
        return {"ok": False, "why": "no POS_ALL"}
    df = pd.read_csv(pa[0], usecols=["Step", "Agent_ID", "X", "Y"])
    if df.empty:
        return {"ok": False, "why": "empty POS_ALL"}
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    finite = np.isfinite(xy).all(axis=1)
    r = np.full(len(xy), np.nan)
    r[finite] = np.hypot(xy[finite, 0] - cx, xy[finite, 1] - cy)
    last = (df["Step"] == df["Step"].max()).to_numpy()
    return {
        "ok": True,
        "frames": int(df["Step"].nunique()),
        "nonfinite_frac": float((~finite).mean()),
        "escaped_frac": float(np.nansum(r > inner_r) / max(int(finite.sum()), 1)),
        "final_spread_over_R": float(np.nanstd(r[last]) / inner_r),
        "final_mean_r_over_R": float(np.nanmean(r[last]) / inner_r),
    }


def main():
    ap = argparse.ArgumentParser(
        description="One trial per gait pattern (a811..a868), same initial "
                    "condition, measuring whether the gait was delivered.")
    ap.add_argument("--out", default=os.path.join("datafile", "pilot_gait"),
                    help="Output root for the 48 runs")
    ap.add_argument("--seconds", type=float, default=60.0,
                    help="Trial length. 60s is ~30 cycles even at the slowest "
                         "gait, plenty to measure fidelity; raise it if the "
                         "failure you are looking for takes longer to appear")
    ap.add_argument("--ic", type=int, default=1,
                    help="Initial-condition index every pattern starts from")
    ap.add_argument("--probe-from", type=float, default=None,
                    help="Ignore the first N seconds when measuring amplitude; "
                         "default is one fifth of the trial, skipping the ramp-up")
    ap.add_argument("--no-video", dest="video", action="store_false",
                    help="Skip video: much faster and smaller, but then only the "
                         "numbers tell you whether a pattern is usable")
    ap.add_argument("--only", nargs="+", default=None, metavar="PAT",
                    help="Run just these patterns, e.g. --only a811 a868")
    ap.add_argument("--analyse-only", action="store_true",
                    help="Skip the runs, re-analyse whatever is already on disk")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="Per-pattern wall-clock limit in seconds")
    args = ap.parse_args()

    out_root = os.path.abspath(args.out)
    cond_dir = os.path.join(out_root, "_conditions")
    probe_dir = os.path.join(out_root, "_probe")
    os.makedirs(cond_dir, exist_ok=True)
    os.makedirs(probe_dir, exist_ok=True)

    pats = patterns()
    if args.only:
        want = set(args.only)
        pats = [p for p in pats if p[0] in want]
        if not pats:
            print(f"No pattern matches {sorted(want)}")
            return 1

    t_probe = (args.probe_from if args.probe_from is not None
               else 0.2 * args.seconds)

    if not args.analyse_only:
        print(f"{len(pats)} patterns x 1 trial x {args.seconds:g}s, "
              f"IC #{args.ic}, video={'on' if args.video else 'off'}")
        t0 = time.time()
        for i, (name, cmd, y, z) in enumerate(pats, 1):
            cfg = write_condition(
                os.path.join(cond_dir, f"{name}.json"), f"a{cmd}", name,
                out_root, args.seconds, args.ic, args.video)
            env = dict(os.environ)
            env["SMARTICLE_CONFIG"] = cfg
            env["GAIT_PROBE_OUT"] = os.path.join(probe_dir, f"{name}.csv")
            env.setdefault("SDL_VIDEODRIVER", "dummy")
            env["PYTHONIOENCODING"] = "utf-8"
            el = time.time() - t0
            eta = el / (i - 1) * (len(pats) - i + 1) if i > 1 else 0.0
            print(f"[{i}/{len(pats)}] {name}  A={_adeg(y)}deg f={_fhz(z)}Hz"
                  f"   elapsed {el/60:.1f}min  eta {eta/60:.0f}min", flush=True)
            try:
                r = subprocess.run([sys.executable, "simulation.py"], env=env,
                                   cwd=HERE, timeout=args.timeout)
                if r.returncode != 0:
                    print(f"    [FAIL] exit {r.returncode}", flush=True)
            except subprocess.TimeoutExpired:
                print(f"    [TIMEOUT] after {args.timeout:g}s", flush=True)

    from gait_probe import analyse_fidelity
    import pandas as pd

    rows = []
    for name, cmd, y, z in pats:
        rec = {"pattern": name, "command": cmd, "Y": y, "Z": z,
               "cmd_ampli_deg": _adeg(y), "cmd_freq_hz": _fhz(z)}
        pl = os.path.join(probe_dir, f"{name}.csv")
        fid = analyse_fidelity(pl, t_min=t_probe) if os.path.isfile(pl) else None
        if fid:
            rec.update({k: v for k, v in fid.items() if k != "command"})
        exp = os.path.join(out_root, name)
        snap = os.path.join(exp, "config_snapshot.json")
        trials = sorted(glob.glob(os.path.join(exp, "trial_*")))
        if os.path.isfile(snap) and trials:
            s = json.load(open(snap, encoding="utf-8"))
            rec.update(sanity(trials[0], s["INNER_R"], s["W"] / 2.0, s["H"] / 2.0))
        rows.append(rec)

    df = pd.DataFrame(rows)
    path = os.path.join(out_root, "pilot_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n-> {path}")

    show = [c for c in ("pattern", "cmd_ampli_deg", "cmd_freq_hz",
                        "meas_ampli_deg", "ampli_ratio", "meas_freq_hz",
                        "freq_ratio", "escaped_frac", "nonfinite_frac")
            if c in df.columns]
    if show:
        print(df[show].to_string(index=False,
                                 float_format=lambda v: f"{v:8.3f}"))
    return 0


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    sys.exit(main())

"""
gait_probe.py  -  a runtime callback that measures nothing but the gait itself.

Point it at a trial to find out whether the robots are actually performing the
gait they were commanded:

    ENABLE_RUNTIME_GAIT_CONTROL = True
    RUNTIME_GAIT_CONTROLLER     = "gait_probe:measure"
    (and set GAIT_PROBE_OUT in the environment to the csv to write)

It never returns a request, so it changes no dynamics -- GaitController.poll
calls it, gets None, and nothing is queued. It exists because the commanded
amplitude is a *setpoint*: at SIM_DT = 1/180 s a 4 Hz gait gets only 45
integration steps per cycle, and in a dense swarm the joints are also fighting
collisions, so the arms may never reach the angle that was asked for. Nothing
in the normal trial output can reveal that -- POS_ALL records the body pose
(X, Y, Theta), not the joints -- so without this the amplitude axis of a gait
sweep would be the requested amplitude rather than the delivered one.

Output is one row per sampled robot per sampled frame:

    t, robot_id, thL, thR, command, cmd_ampli_rad, cmd_freq_hz

Only GAIT_PROBE_IDS robots are logged (default 10, evenly spaced by id) and
only every GAIT_PROBE_EVERY-th poll, because this is a diagnostic, not a
dataset: all 100 robots at 60 Hz for 500 s is 3M rows for a measurement that a
handful of robots over a few dozen cycles already settles.

analyse_fidelity() turns that log into one row per trial: the achieved
amplitude as a fraction of the commanded one, and the achieved frequency from
the mean crossings of the joint angle.
"""

import atexit
import math
import os

_STATE = {}
_REGISTERED = False


def _cfg_int(name, default):
    try:
        return int(os.environ.get(name, "") or default)
    except ValueError:
        return default


def measure(t, robots, ctx):
    """
    Runtime-gait callback: records joint angles, requests nothing.

    Buffers in memory and flushes at the end rather than appending per frame --
    the trial loop calls this every frame, and a flush per call would make the
    probe the slowest thing in the run.
    """
    out = os.environ.get("GAIT_PROBE_OUT")
    if not out:
        return None

    st = _STATE.get(out)
    if st is None:
        # Flush at process exit rather than from the trial loop: the loop has
        # no idea the probe exists, and the driver runs one pattern per
        # subprocess, so process exit is exactly "this trial is over".
        global _REGISTERED
        if not _REGISTERED:
            atexit.register(_flush_all)
            _REGISTERED = True
        n = len(robots)
        k = max(1, _cfg_int("GAIT_PROBE_IDS", 10))
        step = max(1, n // k)
        st = _STATE[out] = {
            "ids": set(range(0, n, step)),
            "every": max(1, _cfg_int("GAIT_PROBE_EVERY", 1)),
            "calls": 0,
            "rows": [],
        }

    st["calls"] += 1
    if (st["calls"] - 1) % st["every"]:
        return None
    # A run with several trials calls this for each in turn, so the trial
    # has to be on the row or the traces would be spliced together.
    tid = int(ctx.get("trial_id", -1)) if isinstance(ctx, dict) else -1
    rows = st["rows"]
    for r in robots:
        if r.id in st["ids"]:
            rows.append((t, tid, r.id, r.thL, r.thR, r.command, r.ampli, r.freq_hz))
    return None


def _flush_all():
    for p in list(_STATE):
        try:
            flush(p)
        except Exception as e:
            print(f"[gait_probe] flush failed for {p}: {type(e).__name__}: {e}")


def flush(path=None):
    """Write the buffered rows. Call once a trial is over; safe to call twice."""
    path = path or os.environ.get("GAIT_PROBE_OUT")
    st = _STATE.pop(path, None)
    if not st or not st["rows"]:
        return None
    # PARALLEL_WORKERS>1 runs trials in separate processes that all inherit
    # the same GAIT_PROBE_OUT; without the pid they would overwrite each
    # other. The analyser globs the stem, so the split costs it nothing.
    if os.environ.get("GAIT_PROBE_SPLIT_PID"):
        stem, ext = os.path.splitext(path)
        path = f"{stem}_p{os.getpid()}{ext}"
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("t,trial_id,robot_id,thL,thR,command,cmd_ampli_rad,cmd_freq_hz\r\n")
        f.write("".join(f"{t:.6f},{g},{i},{l:.6f},{r:.6f},{c},{a:.6f},{z:.4f}\r\n"
                        for t, g, i, l, r, c, a, z in st["rows"]))
    return path


# =============================================================================
# Turning a probe log into a fidelity measurement
# =============================================================================

def analyse_fidelity(path, t_min=None):
    """
    -> dict describing how well the commanded gait was delivered.

    Amplitude is half the peak-to-peak swing per robot, taken from the 2nd/98th
    percentiles rather than min/max: a collision can slam a joint past its
    setpoint for a frame or two, and min/max would report that excursion as the
    amplitude.

    Frequency comes from counting mean crossings of each robot's own trace.
    That needs no window function and does not care that the probe window is a
    non-integer number of cycles -- unlike an FFT, whose resolution over a short
    window would be coarser than the 0.5 Hz spacing of the frequency table this
    sweep steps through.

    t_min discards the start of the trial: the arms begin at rest and the
    warm-up ramps them in, so the first cycles understate the amplitude.
    """
    import numpy as np
    import pandas as pd

    df = pd.read_csv(path)
    if df.empty:
        return None
    if t_min is not None:
        df = df[df["t"] >= t_min]
    if df.empty:
        return None

    cmd_a = float(df["cmd_ampli_rad"].iloc[0])
    cmd_f = float(df["cmd_freq_hz"].iloc[0])
    cmd = int(df["command"].iloc[0])

    amps, freqs = [], []
    # Group by trial as well as robot: concatenating two trials for the same
    # robot would put a discontinuity in the middle of the trace, which the
    # crossing count would read as an extra half cycle.
    keys = ["trial_id", "robot_id"] if "trial_id" in df.columns else ["robot_id"]
    for _k, g in df.groupby(keys):
        tt = g["t"].to_numpy(dtype=float)
        for col in ("thL", "thR"):
            v = g[col].to_numpy(dtype=float)
            if len(v) < 8:
                continue
            lo, hi = np.percentile(v, [2.0, 98.0])
            amps.append(0.5 * (hi - lo))
            c = v - v.mean()
            s = np.signbit(c)
            ncross = int((s[1:] != s[:-1]).sum())
            span = tt[-1] - tt[0]
            if span > 0:
                freqs.append(0.5 * ncross / span)

    if not amps:
        return None
    amps = np.array(amps)
    freqs = np.array(freqs) if freqs else np.array([float("nan")])
    return {
        "command": cmd,
        "cmd_ampli_deg": math.degrees(cmd_a),
        "cmd_freq_hz": cmd_f,
        "meas_ampli_deg": math.degrees(float(np.median(amps))),
        "meas_ampli_deg_sd": math.degrees(float(amps.std(ddof=1))) if len(amps) > 1 else 0.0,
        "ampli_ratio": float(np.median(amps) / cmd_a) if cmd_a > 0 else float("nan"),
        "meas_freq_hz": float(np.nanmedian(freqs)),
        "freq_ratio": float(np.nanmedian(freqs) / cmd_f) if cmd_f > 0 else float("nan"),
        "n_samples": int(len(amps)),
    }

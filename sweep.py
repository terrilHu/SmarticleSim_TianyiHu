"""
sweep.py  -  Parameter sweep over frequency x amplitude x phase space.

Key difference from the original version
-----------------------------------------
The original version passed parameters by monkey-patching global variables in
config/simulation/smarticle.  The updated run_trial decodes everything from
COMMAND_ARRAY (integer-encoded per-robot commands) directly.
This file builds COMMAND_ARRAY and writes it to config; no float globals are
patched anymore.

Integer encoding (consistent with simulation.py / naming.py)
    +/-XYZ
      X (hundreds, 1-8): initial phase slot; 0 = randomise at runtime
      Y (tens,     1-6): amplitude slot
      Z (units,    1-9): frequency slot
      positive sign: both joints in-phase
      negative sign: joints in anti-phase (phase2 = phase1 + pi)

Batch definition
----------------
Add or edit dictionaries in SWEEP_BATCHES (near the top of this file) to
define experiment batches:
    {
        "name":        str,         # batch identifier, used in directory names
        "phase_slots": list[int],   # X values (0 = random, 1-8 from table)
        "ampli_slots": list[int],   # Y values (1-6)
        "freq_slots":  list[int],   # Z values (1-9)
        "antiphase":   bool,        # True -> anti-phase (negative sign)
        "trials":      int,         # number of trials per parameter combination
    }

Usage:
    python sweep.py                   # run all batches
    python sweep.py --batch uniform   # run only batches whose name contains "uniform"
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from config import (
    N_SMARTICLES, TRIAL_SEED_BASE,
    W, H, INNER_R, WALL_THICK, WALL_SEGMENTS, WALL_FRICTION, WALL_ELASTICITY,
    RENDER_FPS_HEADLESS, RATE_LIM, V_MAX, W_MAX, ANG_DAMP, SPACE_DAMP, LIN_DAMP,
    L, L_s, WC, R0, a0, a1, g0,
    RECORD_VIDEO, RECORD_POLICY, RECORD_EVERY_K, RECORD_FIRST_N, VIDEO_STRIDE,
    ALREADY_SPWANED, SIM_DT,
)
from spawn import load_initial_conditions
from analysis import actuationimpactCalculation, write_results_csv
from naming import generate_trial_name, _FREQ_DICT, _AMPLI_DICT, _INITIAL_DICT
from simulation import run_trial, print_progress_bar

# =============================================================================
# Lookup tables (mirrors naming.py so slot indices are consistent)
# =============================================================================

# Frequency table (Hz); 1-based slot index maps to the Z digit
FREQ_TABLE  = _FREQ_DICT     # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

# Amplitude table (rad); 1-based slot index maps to the Y digit
AMPLI_TABLE = _AMPLI_DICT    # [pi/12, pi/6, pi/4, pi/3, 5pi/12, pi/2]

# Initial phase table (rad); 1-based slot index maps to the X digit; 0 = random
PHASE_TABLE = _INITIAL_DICT  # [pi/4 ... 2*pi]

# =============================================================================
# Batch definitions  -  edit here to configure experiments
# =============================================================================

SWEEP_BATCHES = [
    # Example batch 1: full amplitude x frequency grid, fixed in-phase (X=8 = 2*pi)
    # {
    #     "name":        "full_sweep_inphase",
    #     "phase_slots": [8],                          # X=8 -> phase = 2*pi (aligned)
    #     "ampli_slots": [4],           # all amplitude slots
    #     "freq_slots":  [1, 2, 3], # all frequency slots
    #     "antiphase":   True,
    #     "trials":      3,
    # },
]

# =============================================================================
# Mixture batch definitions  -  edit here to configure mixture experiments
# =============================================================================
# Each entry describes one mixture sweep:
#   "name"     : batch identifier used in directory names and filenames
#   "cmd_a"    : integer command assigned to the n selected robots
#   "cmd_b"    : integer command assigned to the remaining (N_SMARTICLES - n) robots
#   "n_values" : list of counts n to sweep over (each defines one group of trials)
#   "trials"   : number of trials per value of n (robots are re-sampled each trial)
#
# For every trial the n robots receiving cmd_a are chosen uniformly at random
# from all N_SMARTICLES robots (without replacement), so different trials with
# the same n produce different spatial arrangements.

MIXTURE_BATCHES = [
    # Example: sweep the number of "026" robots against a background of "062" robots
    {
        "name":     "mix_-041_vs_-024",
        "cmd_a":    -41,                              # command for the minority group
        "cmd_b":    -24,                              # command for the majority group
        "n_values": [0, 2, 4, 6, 8, 9, 11, 13, 15, 17],
        "trials":   5,
    },
    # Add further mixture batches here
]

# =============================================================================
# Global trial settings
# =============================================================================

INIT_FILE      = "init_conditions/init_conditions_200.json"
LAST_N_SECONDS = 10.0                               # statistics window at end of trial
FRAMES_PER_SEC = RENDER_FPS_HEADLESS                # recorded frames per second
LAST_N_FRAMES  = int(LAST_N_SECONDS * FRAMES_PER_SEC)  # number of frames in window


# =============================================================================
# Helper: build COMMAND_ARRAY from slot indices
# =============================================================================

def make_command_array(
    n: int,
    phase_slot: int,
    ampli_slot: int,
    freq_slot:  int,
    antiphase:  bool,
) -> list:
    """
    Build a uniform COMMAND_ARRAY of length n from slot indices.

    Parameters
    ----------
    n           : number of robots
    phase_slot  : X digit (0 = random at runtime, 1-8 = table lookup)
    ampli_slot  : Y digit (1-6)
    freq_slot   : Z digit (1-9)
    antiphase   : if True, encode as negative (joint2 phase = joint1 + pi)

    Returns
    -------
    List of n integers, each equal to +/-XYZ.
    """
    assert 0 <= phase_slot <= 8, f"phase_slot must be 0-8, got {phase_slot}"
    assert 1 <= ampli_slot <= 6, f"ampli_slot must be 1-6, got {ampli_slot}"
    assert 1 <= freq_slot  <= 9, f"freq_slot must be 1-9, got {freq_slot}"

    code = phase_slot * 100 + ampli_slot * 10 + freq_slot
    if antiphase:
        code = -code
    return [code] * n


def decode_command(cmd: int) -> dict:
    """
    Decode a single integer command into a human-readable parameter dict.
    phase_rad is None when phase_slot=0 (randomised at runtime).
    """
    sign     = -1 if cmd < 0 else 1
    abs_cmd  = abs(cmd)
    z        = abs_cmd % 10
    y        = (abs_cmd % 100 - z) // 10
    x        = abs_cmd // 100
    freq_hz  = FREQ_TABLE[z - 1]  if 1 <= z <= 9 else None
    ampli_rad= AMPLI_TABLE[y - 1] if 1 <= y <= 6 else None
    phase_rad= PHASE_TABLE[x - 1] if 1 <= x <= 8 else None  # None means random
    return {
        "phase_slot": x,
        "ampli_slot": y,
        "freq_slot":  z,
        "antiphase":  sign < 0,
        "freq_hz":    freq_hz,
        "ampli_rad":  ampli_rad,
        "ampli_deg":  math.degrees(ampli_rad) if ampli_rad is not None else None,
        "phase_rad":  phase_rad,
    }


# =============================================================================
# Helper: recover naming parameters from COMMAND_ARRAY
# =============================================================================

def _naming_params_from_commands(commands: list):
    """
    Extract the arguments required by generate_trial_name from a COMMAND_ARRAY.
    When phase_slot=0 (random), init_phases uses 0.0 as a placeholder, which
    causes the filename to show J1p0/J2p0.
    """
    decoded = [decode_command(c) for c in commands]
    init_phases = []
    for d in decoded:
        ph  = d["phase_rad"] if d["phase_rad"] is not None else 0.0
        ph2 = ph + math.pi if d["antiphase"] else ph
        init_phases.append((ph, ph2))

    # All robots share the same omega/amplitude in a sweep combination
    d0    = decoded[0]
    omega = (d0["freq_hz"] * 2 * math.pi, d0["freq_hz"] * 2 * math.pi)
    ampli = (d0["ampli_deg"], d0["ampli_deg"])
    return init_phases, omega, ampli


# =============================================================================
# Alignment statistics
# =============================================================================

def alignment_stats_from_csv(alignment_csv: str):
    """
    Read *_alignment.csv, take the last LAST_N_FRAMES rows, and return
    (mean, std) over that window.
    Returns (nan, nan) if the file is missing or contains too few rows.
    """
    if not os.path.isfile(alignment_csv):
        return float("nan"), float("nan")
    try:
        df     = pd.read_csv(alignment_csv)
        vals   = df["mean_alignment"].to_numpy(dtype=float)
        window = vals[-LAST_N_FRAMES:]
        if len(window) == 0:
            return float("nan"), float("nan")
        return float(np.mean(window)), float(np.std(window))
    except Exception:
        return float("nan"), float("nan")


# =============================================================================
# Heatmap
# =============================================================================

def plot_heatmap(heatmap_mean, heatmap_std, ampli_labels, freq_labels,
                 out_path, title="Mean alignment (last 10 s)",
                 cmap="viridis", vmin=0, vmax=1):
    """
    Plot a freq x ampli heatmap coloured by mean alignment.
    Each cell is annotated with "mean / +-std".
    x-axis: amplitude (ampli_labels); y-axis: frequency (freq_labels, low at bottom).
    """
    n_freq, n_ampli = heatmap_mean.shape
    fig, ax = plt.subplots(figsize=(max(6, n_ampli * 1.4), max(4, n_freq * 0.8)))

    im = ax.imshow(heatmap_mean, origin="lower", aspect="auto",
                   vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(n_ampli))
    ax.set_xticklabels(ampli_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_freq))
    ax.set_yticklabels(freq_labels, fontsize=9)
    ax.set_xlabel("Amplitude (rad)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title(title, fontsize=11)
    plt.colorbar(im, ax=ax, label="Mean alignment")

    # Annotate each cell: top line = mean, bottom line = +/-std
    for fi in range(n_freq):
        for ai in range(n_ampli):
            m     = heatmap_mean[fi, ai]
            s     = heatmap_std [fi, ai]
            txt   = "-" if np.isnan(m) else f"{m:.2f}\n+/-{s:.2f}"
            color = "white" if (np.isnan(m) or m < 0.5) else "black"
            ax.text(ai, fi, txt, ha="center", va="center",
                    fontsize=7.5, color=color, linespacing=1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[heatmap] Saved -> {out_path}")


# =============================================================================
# Single-batch runner
# =============================================================================

def run_batch(batch: dict, ALL_INIT: list, init_cursor_start: int,
              global_trial_counter_start: int, n_total: int,
              start_time: float) -> tuple:
    """
    Run all parameter combinations for one batch.

    Parameters
    ----------
    batch                      : batch configuration dict
    ALL_INIT                   : full list of pre-generated initial conditions
    init_cursor_start          : index into ALL_INIT where this batch begins
    global_trial_counter_start : starting value for the global progress counter
    n_total                    : total number of trials across all batches (for progress bar)
    start_time                 : wall-clock start time (for ETA calculation)

    Returns
    -------
    (all_results, init_cursor_end, global_trial_counter_end)
    """
    name        = batch["name"]
    phase_slots = batch["phase_slots"]
    ampli_slots = batch["ampli_slots"]
    freq_slots  = batch["freq_slots"]
    antiphase   = batch.get("antiphase", False)
    n_trials    = batch.get("trials", 3)

    all_results          = []
    init_cursor          = init_cursor_start
    global_trial_counter = global_trial_counter_start

    n_a = len(ampli_slots)
    n_f = len(freq_slots)
    n_p = len(phase_slots)

    # Heatmap accumulators (freq x ampli); only filled when n_p == 1
    heatmap_mean = np.full((n_f, n_a), float("nan"))
    heatmap_std  = np.full((n_f, n_a), float("nan"))

    print(f"\n{'='*60}")
    print(f"[batch] {name}  -  "
          f"{n_p} phase(s) x {n_a} ampli(s) x {n_f} freq(s) x {n_trials} trials"
          f" = {n_p * n_a * n_f * n_trials} total")
    print(f"{'='*60}")

    for p_idx, p_slot in enumerate(phase_slots):
        for a_idx, a_slot in enumerate(ampli_slots):
            for f_idx, f_slot in enumerate(freq_slots):

                # Build COMMAND_ARRAY for this combination and inject into both
                # config and simulation.  config.COMMAND_ARRAY must be set so
                # that the snapshot and any future imports see the right value.
                # simulation.COMMAND_ARRAY must also be patched explicitly because
                # simulation.py binds the name at import time via
                # "from config import COMMAND_ARRAY", so reassigning config's
                # attribute alone does not update the reference inside run_trial.
                import simulation as _sim
                cmd_array = make_command_array(
                    N_SMARTICLES, p_slot, a_slot, f_slot, antiphase)
                config.COMMAND_ARRAY = cmd_array
                _sim.COMMAND_ARRAY   = cmd_array

                # Derive naming parameters from the command array
                init_phases, omega, ampli = _naming_params_from_commands(cmd_array)
                exp_name  = generate_trial_name(
                    N_SMARTICLES, init_phases, omega=omega, amplitude=ampli,
                    prefix=f"batch_{name}",
                )
                out_dir   = os.path.join("datafile", exp_name)
                video_dir = os.path.join("videos",   exp_name)
                os.makedirs(out_dir,   exist_ok=True)
                os.makedirs(video_dir, exist_ok=True)

                # Save a JSON snapshot of config for reproducibility
                _save_config_snapshot(out_dir, cmd_array, batch)

                # Compute actuation impact values (still required by run_trial)
                freq_hz    = FREQ_TABLE[f_slot - 1]
                ampli_rad  = AMPLI_TABLE[a_slot - 1]
                actuations = actuationimpactCalculation(
                    freq_hz, freq_hz, ampli_rad, ampli_rad)

                trial_means   = []  # per-trial mean alignment for this combination
                combo_results = []

                for local_tid in range(n_trials):
                    global_init_idx = init_cursor
                    init_cursor    += 1
                    seed      = TRIAL_SEED_BASE + global_init_idx
                    trial_out = os.path.join(out_dir, f"trial_{local_tid:04d}")

                    try:
                        res = run_trial(
                            trial_id  = local_tid,
                            seed      = seed,
                            preview   = False,
                            video_dir = video_dir,
                            out_dir   = trial_out,
                            actuations= actuations,
                            ALL_INIT  = [ALL_INIT[global_init_idx]],
                            init_idx  = 0,  # the list passed in has exactly one entry
                        )
                        res["exp_name"]   = exp_name
                        res["batch"]      = name
                        res["local_tid"]  = local_tid
                        res["phase_slot"] = p_slot
                        res["ampli_slot"] = a_slot
                        res["freq_slot"]  = f_slot
                        res["antiphase"]  = antiphase

                        # Read alignment statistics from the output CSV
                        align_csv = os.path.join(
                            trial_out, f"trial_0000_alignment.csv")
                        t_mean, t_std = alignment_stats_from_csv(align_csv)
                        res["align_last10s_mean"] = t_mean
                        res["align_last10s_std"]  = t_std
                        if not np.isnan(t_mean):
                            trial_means.append(t_mean)
                        combo_results.append(res)
                    except Exception as e:
                        print(f"\n[WARN] {exp_name} trial {local_tid} failed: {e}")

                    global_trial_counter += 1
                    print_progress_bar(global_trial_counter, n_total, start_time)

                # Compute combination-level statistics across all trials
                if trial_means:
                    combo_mean = float(np.mean(trial_means))
                    combo_std  = float(np.std(trial_means))
                else:
                    combo_mean = combo_std = float("nan")

                # Only the first phase slot populates the 2-D heatmap;
                # multi-phase batches do not produce a single 2-D summary plot.
                if p_idx == 0:
                    heatmap_mean[f_idx, a_idx] = combo_mean
                    heatmap_std [f_idx, a_idx] = combo_std

                for r in combo_results:
                    r["combo_align_mean"] = combo_mean
                    r["combo_align_std"]  = combo_std

                all_results.extend(combo_results)
                if combo_results:
                    write_results_csv(
                        os.path.join(out_dir, "summary.csv"), combo_results)

    # Save a heatmap only for single-phase batches (2-D grid is well-defined)
    if n_p == 1:
        ampli_labels = [f"{AMPLI_TABLE[a-1]/math.pi:.3g}pi" for a in ampli_slots]
        freq_labels  = [f"{FREQ_TABLE[f-1]}" for f in freq_slots]
        heatmap_path = os.path.join("datafile", f"heatmap_{name}.png")
        plot_heatmap(
            heatmap_mean, heatmap_std,
            ampli_labels, freq_labels,
            out_path=heatmap_path,
            title=(f"[{name}] Mean alignment (last {LAST_N_SECONDS:.0f} s)  -  "
                   f"{'anti-phase' if antiphase else 'in-phase'}"),
        )

    return all_results, init_cursor, global_trial_counter


# =============================================================================
# Mixture experiment: n robots with cmd_a, the rest with cmd_b
# =============================================================================

def make_mixture_command_array(
    n_total: int,
    n_a: int,
    cmd_a: int,
    cmd_b: int,
    rng: "np.random.Generator",
) -> list:
    """
    Build a COMMAND_ARRAY where exactly n_a robots (chosen uniformly at random)
    receive cmd_a and the remaining (n_total - n_a) robots receive cmd_b.

    Parameters
    ----------
    n_total : total number of robots (must equal N_SMARTICLES)
    n_a     : number of robots assigned cmd_a (0 <= n_a <= n_total)
    cmd_a   : integer command for the selected robots
    cmd_b   : integer command for the remaining robots
    rng     : numpy random Generator used for reproducible sampling

    Returns
    -------
    List of n_total integers with exactly n_a copies of cmd_a at random positions
    and (n_total - n_a) copies of cmd_b at the remaining positions.
    """
    assert 0 <= n_a <= n_total, (
        f"n_a must be between 0 and n_total ({n_total}), got {n_a}")

    cmd_array = [cmd_b] * n_total
    # Choose which robot indices receive cmd_a
    indices_a = rng.choice(n_total, size=n_a, replace=False)
    for idx in indices_a:
        cmd_array[idx] = cmd_a
    return cmd_array


def run_mixture_batch(
    batch: dict,
    ALL_INIT: list,
    init_cursor_start: int,
    global_trial_counter_start: int,
    n_total_trials: int,
    start_time: float,
) -> tuple:
    """
    Run a mixture sweep: for each value of n in batch["n_values"], run
    batch["trials"] trials where n randomly selected robots are assigned
    cmd_a and the rest are assigned cmd_b.  The random selection is re-drawn
    independently for every trial, so spatial arrangements vary within a group.

    A line plot of mean alignment vs n is saved after all groups complete.

    Parameters
    ----------
    batch                      : mixture batch config dict (see MIXTURE_BATCHES)
    ALL_INIT                   : full list of pre-generated initial conditions
    init_cursor_start          : index into ALL_INIT where this batch begins
    global_trial_counter_start : starting value for the global progress counter
    n_total_trials             : total trials across all batches (for progress bar)
    start_time                 : wall-clock start time (for ETA)

    Returns
    -------
    (all_results, init_cursor_end, global_trial_counter_end)
    """
    import random as _random

    name     = batch["name"]
    cmd_a    = batch["cmd_a"]
    cmd_b    = batch["cmd_b"]
    n_values = batch["n_values"]
    n_trials = batch.get("trials", 5)

    all_results          = []
    init_cursor          = init_cursor_start
    global_trial_counter = global_trial_counter_start

    # Decode cmd_a and cmd_b once to build actuation objects and directory labels
    dec_a = decode_command(cmd_a)
    dec_b = decode_command(cmd_b)

    # Use cmd_a's frequency/amplitude for the actuation object passed to run_trial.
    # run_trial re-decodes per-robot commands internally, so this value is only
    # used for the interaction-model pre-computation; cmd_b values are also decoded
    # inside run_trial for each robot that carries cmd_b.
    freq_hz_a   = dec_a["freq_hz"]   or FREQ_TABLE[0]
    ampli_rad_a = dec_a["ampli_rad"] or AMPLI_TABLE[0]
    freq_hz_b   = dec_b["freq_hz"]   or FREQ_TABLE[0]
    ampli_rad_b = dec_b["ampli_rad"] or AMPLI_TABLE[0]

    # Accumulate per-n statistics for the summary line plot
    n_axis       = []   # n values that produced at least one valid trial
    means_axis   = []   # mean of per-trial alignment means
    stds_axis    = []   # std  of per-trial alignment means

    print(f"\n{'='*60}")
    print(f"[mixture batch] {name}")
    print(f"  cmd_a={cmd_a}  cmd_b={cmd_b}")
    print(f"  n_values={n_values}  trials per n={n_trials}")
    print(f"  total={len(n_values) * n_trials} trials")
    print(f"{'='*60}")

    for n_a in n_values:
        group_results = []
        trial_means   = []  # per-trial mean alignment for this value of n

        # One dedicated RNG per (batch, n_a) group for reproducible sampling.
        # Seed is derived from the global trial seed and the group index so that
        # results do not change when other n_values are added or removed.
        group_seed = TRIAL_SEED_BASE + abs(cmd_a) * 10000 + abs(cmd_b) * 1000 + n_a
        rng = np.random.default_rng(group_seed)

        # Build a human-readable label for directory naming, e.g. "na08_of_17"
        group_label = f"na{n_a:02d}_of_{N_SMARTICLES:02d}"
        exp_name    = f"mixture_{name}_{group_label}_a{abs(cmd_a)}_b{abs(cmd_b)}"
        out_dir     = os.path.join("datafile", exp_name)
        video_dir   = os.path.join("videos",   exp_name)
        os.makedirs(out_dir,   exist_ok=True)
        os.makedirs(video_dir, exist_ok=True)

        for local_tid in range(n_trials):
            # Re-sample which robots get cmd_a for every trial.
            # Patch both config and simulation for the same reason as run_batch:
            # simulation.py imports COMMAND_ARRAY by value at load time.
            import simulation as _sim
            cmd_array = make_mixture_command_array(
                N_SMARTICLES, n_a, cmd_a, cmd_b, rng)
            config.COMMAND_ARRAY = cmd_array
            _sim.COMMAND_ARRAY   = cmd_array

            # actuation object: use cmd_a parameters (symmetric case);
            # run_trial uses per-robot omega/A from the decoded commands directly
            actuations = actuationimpactCalculation(
                freq_hz_a, freq_hz_b, ampli_rad_a, ampli_rad_b)

            global_init_idx = init_cursor
            init_cursor    += 1
            seed      = TRIAL_SEED_BASE + global_init_idx
            trial_out = os.path.join(out_dir, f"trial_{local_tid:04d}")

            try:
                res = run_trial(
                    trial_id  = local_tid,
                    seed      = seed,
                    preview   = False,
                    video_dir = video_dir,
                    out_dir   = trial_out,
                    actuations= actuations,
                    ALL_INIT  = [ALL_INIT[global_init_idx]],
                    init_idx  = 0,
                )
                res["exp_name"]    = exp_name
                res["batch"]       = name
                res["local_tid"]   = local_tid
                res["n_a"]         = n_a
                res["cmd_a"]       = cmd_a
                res["cmd_b"]       = cmd_b
                res["cmd_array"]   = cmd_array   # full per-robot assignment

                align_csv = os.path.join(trial_out, f"trial_0000_alignment.csv")
                t_mean, t_std = alignment_stats_from_csv(align_csv)
                res["align_last10s_mean"] = t_mean
                res["align_last10s_std"]  = t_std
                if not np.isnan(t_mean):
                    trial_means.append(t_mean)
                group_results.append(res)

            except Exception as e:
                print(f"\n[WARN] {exp_name} trial {local_tid} failed: {e}")

            global_trial_counter += 1
            print_progress_bar(global_trial_counter, n_total_trials, start_time)

        # Compute group-level statistics
        if trial_means:
            g_mean = float(np.mean(trial_means))
            g_std  = float(np.std(trial_means))
            n_axis.append(n_a)
            means_axis.append(g_mean)
            stds_axis.append(g_std)
        else:
            g_mean = g_std = float("nan")

        for r in group_results:
            r["group_align_mean"] = g_mean
            r["group_align_std"]  = g_std

        all_results.extend(group_results)
        if group_results:
            write_results_csv(os.path.join(out_dir, "summary.csv"), group_results)

        # Save config snapshot for this group
        _save_config_snapshot(out_dir, cmd_array, {
            "name":     name,
            "n_a":      n_a,
            "cmd_a":    cmd_a,
            "cmd_b":    cmd_b,
            "antiphase": False,   # encoded in cmd sign, not a batch-level flag
        })

    # ── Summary line plot: mean alignment vs number of cmd_a robots ──────────
    if n_axis:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.errorbar(n_axis, means_axis, yerr=stds_axis,
                    fmt="o-", capsize=4, linewidth=1.5)
        ax.set_xlabel(f"Number of robots with cmd_a ({cmd_a})", fontsize=11)
        ax.set_ylabel("Mean alignment (last 10 s)", fontsize=11)
        ax.set_title(
            f"[{name}]  cmd_a={cmd_a}  cmd_b={cmd_b}\n"
            f"mean +/- std across {n_trials} trial(s) per n",
            fontsize=10,
        )
        ax.set_xlim(-0.5, N_SMARTICLES + 0.5)
        ax.set_ylim(0, 1)
        ax.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()
        plot_path = os.path.join("datafile", f"mixture_{name}_lineplot.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        print(f"[mixture] Line plot saved -> {plot_path}")

    return all_results, init_cursor, global_trial_counter


# =============================================================================
# Main entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Smarticle parameter sweep")
    parser.add_argument(
        "--batch", default=None,
        help="Run only batches whose name contains this string (case-insensitive). "
             "Applies to both SWEEP_BATCHES and MIXTURE_BATCHES.")
    args = parser.parse_args()

    # Filter regular and mixture batches by name if --batch was specified
    sweep_batches   = SWEEP_BATCHES
    mixture_batches = MIXTURE_BATCHES
    if args.batch:
        key = args.batch.lower()
        sweep_batches   = [b for b in SWEEP_BATCHES   if key in b["name"].lower()]
        mixture_batches = [b for b in MIXTURE_BATCHES if key in b["name"].lower()]
        if not sweep_batches and not mixture_batches:
            print(f"[ERROR] No batch whose name contains '{args.batch}'.")
            print("Available sweep batches:")
            for b in SWEEP_BATCHES:
                print(f"  - {b['name']}")
            print("Available mixture batches:")
            for b in MIXTURE_BATCHES:
                print(f"  - {b['name']}")
            sys.exit(1)

    # Count total trials across both batch types
    n_sweep = sum(
        len(b["phase_slots"]) * len(b["ampli_slots"]) *
        len(b["freq_slots"]) * b.get("trials", 3)
        for b in sweep_batches
    )
    n_mixture = sum(
        len(b["n_values"]) * b.get("trials", 5)
        for b in mixture_batches
    )
    n_total = n_sweep + n_mixture

    ALL_INIT = load_initial_conditions(INIT_FILE)
    if len(ALL_INIT) < n_total:
        raise RuntimeError(
            f"Not enough initial conditions: need {n_total}, "
            f"but '{INIT_FILE}' only contains {len(ALL_INIT)}."
        )
    print(f"Loaded {len(ALL_INIT)} initial conditions from '{INIT_FILE}'.")
    print(f"Running {n_total} trials  "
          f"({n_sweep} sweep + {n_mixture} mixture)  "
          f"across {len(sweep_batches) + len(mixture_batches)} batch(es).\n")

    os.makedirs("datafile", exist_ok=True)
    os.makedirs("videos",   exist_ok=True)

    all_results          = []
    init_cursor          = 0
    global_trial_counter = 0
    start_time           = time.time()

    # Run regular sweep batches
    for batch in sweep_batches:
        batch_results, init_cursor, global_trial_counter = run_batch(
            batch, ALL_INIT,
            init_cursor_start          = init_cursor,
            global_trial_counter_start = global_trial_counter,
            n_total                    = n_total,
            start_time                 = start_time,
        )
        all_results.extend(batch_results)

    # Run mixture batches
    for batch in mixture_batches:
        batch_results, init_cursor, global_trial_counter = run_mixture_batch(
            batch, ALL_INIT,
            init_cursor_start          = init_cursor,
            global_trial_counter_start = global_trial_counter,
            n_total_trials             = n_total,
            start_time                 = start_time,
        )
        all_results.extend(batch_results)

    # Write a single summary CSV covering all batches
    write_results_csv("datafile/sweep_summary.csv", all_results)
    elapsed = time.time() - start_time
    print(f"\nAll batches complete in {elapsed/60:.1f} min.")
    print("Global summary saved to datafile/sweep_summary.csv")


# =============================================================================
# Helper: save config snapshot
# =============================================================================

def _save_config_snapshot(out_dir: str, cmd_array: list, batch: dict):
    """
    Write a JSON snapshot of the current config module, the active COMMAND_ARRAY,
    and the batch metadata to out_dir/config_snapshot.json.
    """
    import config as _cfg
    snapshot = {}
    for k, v in vars(_cfg).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)):
            snapshot[k] = v
        elif isinstance(v, list):
            try:
                snapshot[k] = [list(x) if isinstance(x, tuple) else x for x in v]
            except Exception:
                pass
    snapshot["COMMAND_ARRAY"] = cmd_array
    snapshot["_batch_name"]   = batch["name"]
    snapshot["_antiphase"]    = batch.get("antiphase", False)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2)


if __name__ == "__main__":
    main()

"""
sweep.py  ─  Parameter sweep over frequency × amplitude space.

Iterates all 6×9 = 54 (amplitude, frequency) combinations defined in naming.py,
runs 3 trials per combination, and saves results under datafile/ and videos/.

Initial conditions are drawn sequentially from a pre-generated JSON file.
Each combination gets a non-overlapping block of 3 initial conditions.
Total trials: 54 × 3 = 162  (requires at least 162 entries in the JSON).

After all trials complete, a heatmap is saved showing mean alignment (last 10 s)
with amplitude on the x-axis and frequency on the y-axis.

Usage:
    python sweep.py
"""

import math
import os
import time
import json
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Patch config before importing simulation ──────────────────────────────────
import config
from config import (
    N_SMARTICLES, TRIAL_SEED_BASE, WARMUP_STEPS,
    W, H, INNER_R, WALL_THICK, WALL_SEGMENTS, WALL_FRICTION, WALL_ELASTICITY,
    RENDER_FPS_HEADLESS, RATE_LIM, V_MAX, W_MAX, ANG_DAMP, SPACE_DAMP, LIN_DAMP,
    L, L_s, WC, R0, a0, a1, g0,
    RECORD_VIDEO, RECORD_POLICY, RECORD_EVERY_K, RECORD_FIRST_N, VIDEO_STRIDE,
    ALREADY_SPWANED, SIM_DT,
)
from spawn import load_initial_conditions
from analysis import actuationimpactCalculation, write_results_csv
from naming import generate_trial_name
from simulation import run_trial, print_progress_bar

# =============================================================================
# Sweep parameters
# =============================================================================

INIT_FILE        = "init_conditions/init_conditions_200.json"
TRIALS_PER_COMBO = 3
INITIAL_PHASE    = 2 * math.pi     # fixed for all robots, both joints
LAST_N_SECONDS   = 10.0            # window for alignment statistics

# Data collection frame rate: one frame every (1/RENDER_FPS_HEADLESS) seconds
# (RENDER_FPS_HEADLESS physics steps are grouped into one recorded frame)
FRAMES_PER_SEC   = RENDER_FPS_HEADLESS          # = 60
LAST_N_FRAMES    = int(LAST_N_SECONDS * FRAMES_PER_SEC)   # = 600

# Amplitude table (radians), indices 1-6
AMPLI_DICT = [
    math.pi/12, math.pi/6, math.pi/4,
    math.pi/3,  math.pi*5/12, math.pi/2
]

# Frequency table (Hz), indices 1-9
FREQ_DICT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]


# =============================================================================
# Alignment statistics for one trial
# =============================================================================

def alignment_stats_from_csv(alignment_csv: str):
    """
    Read *_alignment.csv, take the last LAST_N_FRAMES rows,
    return (mean, std) over that window.
    Returns (nan, nan) if the file is missing or too short.
    """
    if not os.path.isfile(alignment_csv):
        return float("nan"), float("nan")
    try:
        df = pd.read_csv(alignment_csv)
        vals = df["mean_alignment"].to_numpy(dtype=float)
        window = vals[-LAST_N_FRAMES:]
        if len(window) == 0:
            return float("nan"), float("nan")
        return float(np.mean(window)), float(np.std(window))
    except Exception:
        return float("nan"), float("nan")


# =============================================================================
# Heatmap
# =============================================================================

def plot_heatmap(heatmap_mean, heatmap_std, ampli_labels, freq_labels, out_path):
    """
    Single heatmap coloured by mean alignment (last 10 s).
    Each cell is annotated with "mean\n±std".
    x-axis : amplitude (6 levels)
    y-axis : frequency (9 levels), low freq at bottom
    """
    n_freq, n_ampli = heatmap_mean.shape
    fig, ax = plt.subplots(figsize=(9, 6))

    im = ax.imshow(
        heatmap_mean,
        origin="lower",       # freq[0]=0.5 Hz at bottom
        aspect="auto",
        vmin=0, vmax=1,
        cmap="viridis",
    )
    ax.set_xticks(range(n_ampli))
    ax.set_xticklabels(ampli_labels, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(n_freq))
    ax.set_yticklabels(freq_labels, fontsize=9)
    ax.set_xlabel("Amplitude (rad)", fontsize=11)
    ax.set_ylabel("Frequency (Hz)", fontsize=11)
    ax.set_title("Mean alignment over last 10 s  (mean ± std across 3 trials)", fontsize=11)
    plt.colorbar(im, ax=ax, label="Mean alignment")

    # Annotate each cell: top line = mean, bottom line = ±std
    for fi in range(n_freq):
        for ai in range(n_ampli):
            m = heatmap_mean[fi, ai]
            s = heatmap_std [fi, ai]
            if np.isnan(m):
                txt = "—"
            else:
                txt = f"{m:.2f}\n±{s:.2f}"
            color = "white" if m < 0.5 or np.isnan(m) else "black"
            ax.text(ai, fi, txt, ha="center", va="center",
                    fontsize=7.5, color=color, linespacing=1.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"[heatmap] Saved → {out_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    # ── Load and validate initial conditions ─────────────────────────────────
    ALL_INIT   = load_initial_conditions(INIT_FILE)
    n_combos   = len(AMPLI_DICT) * len(FREQ_DICT)
    n_required = n_combos * TRIALS_PER_COMBO

    if len(ALL_INIT) < n_required:
        raise RuntimeError(
            f"Not enough initial conditions: need {n_required} "
            f"({n_combos} combos × {TRIALS_PER_COMBO} trials), "
            f"but {INIT_FILE} only has {len(ALL_INIT)}."
        )

    print(f"Loaded {len(ALL_INIT)} initial conditions from '{INIT_FILE}'.")
    print(f"Running {n_combos} combinations × {TRIALS_PER_COMBO} trials "
          f"= {n_required} total trials.\n")

    # heatmap_mean[f_idx, a_idx] = mean of (per-trial means) over 3 trials
    # heatmap_std [f_idx, a_idx] = std  of (per-trial means) over 3 trials
    n_a = len(AMPLI_DICT)
    n_f = len(FREQ_DICT)
    heatmap_mean = np.full((n_f, n_a), float("nan"))
    heatmap_std  = np.full((n_f, n_a), float("nan"))

    all_results = []
    start_time  = time.time()
    total_done  = 0
    init_cursor = 0

    for a_idx, ampli_rad in enumerate(AMPLI_DICT):
        for f_idx, freq_hz in enumerate(FREQ_DICT):

            omega     = freq_hz * 2 * math.pi
            ampli_deg = math.degrees(ampli_rad)
            #init_phases = [(INITIAL_PHASE, INITIAL_PHASE)] * N_SMARTICLES
            init_phases = [(p := random.random() * 2 * math.pi, p) for _ in range(N_SMARTICLES)]

            # Patch config and imported module namespaces
            config.OMEGA_NOM1  = omega;      config.OMEGA_NOM2  = omega
            config.A_DEG_NOM1  = ampli_deg;  config.A_DEG_NOM2  = ampli_deg
            config.INIT_PHASES = init_phases

            import simulation as _sim
            import smarticle  as _sm
            _sim.OMEGA_NOM1  = omega;   _sim.OMEGA_NOM2  = omega
            _sim.A_DEG_NOM1  = ampli_deg; _sim.A_DEG_NOM2 = ampli_deg
            _sim.INIT_PHASES = init_phases
            _sm.OMEGA_NOM1   = omega;   _sm.OMEGA_NOM2   = omega
            _sm.A_DEG_NOM1   = ampli_deg; _sm.A_DEG_NOM2 = ampli_deg

            exp_name  = generate_trial_name(
                N_SMARTICLES, init_phases,
                omega=(omega, omega), amplitude=(ampli_deg, ampli_deg),
            )
            out_dir   = os.path.join("datafile", exp_name)
            video_dir = os.path.join("videos",   exp_name)
            os.makedirs(out_dir,   exist_ok=True)
            os.makedirs(video_dir, exist_ok=True)
            _save_config_snapshot(out_dir, omega, ampli_deg, init_phases)

            actuations = actuationimpactCalculation(
                freq_hz, freq_hz, ampli_rad, ampli_rad)

            trial_means = []   # per-trial mean alignment (last 10 s)
            combo_results = []

            for local_tid in range(TRIALS_PER_COMBO):
                global_init_idx = init_cursor
                init_cursor    += 1
                seed      = TRIAL_SEED_BASE + global_init_idx
                trial_out = os.path.join(out_dir, f"trial_{local_tid:04d}")

                try:
                    res = run_trial(
                        trial_id  = 0,
                        seed      = seed,
                        preview   = False,
                        video_dir = video_dir,
                        out_dir   = trial_out,
                        actuations= actuations,
                        ALL_INIT  = [ALL_INIT[global_init_idx]],
                    )
                    res["exp_name"]   = exp_name
                    res["local_tid"]  = local_tid
                    res["ampli_slot"] = a_idx + 1
                    res["freq_slot"]  = f_idx + 1

                    # ── Alignment statistics for this trial ───────────────
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

                total_done += 1
                print_progress_bar(total_done, n_required, start_time)

            # ── Combination-level alignment statistics ────────────────────
            if trial_means:
                combo_mean = float(np.mean(trial_means))
                combo_std  = float(np.std(trial_means))
            else:
                combo_mean = combo_std = float("nan")

            heatmap_mean[f_idx, a_idx] = combo_mean
            heatmap_std [f_idx, a_idx] = combo_std

            for r in combo_results:
                r["combo_align_mean"] = combo_mean
                r["combo_align_std"]  = combo_std

            all_results.extend(combo_results)
            if combo_results:
                write_results_csv(os.path.join(out_dir, "summary.csv"), combo_results)

    # ── Global summary ────────────────────────────────────────────────────────
    write_results_csv("datafile/sweep_summary.csv", all_results)
    elapsed = time.time() - start_time
    print(f"\nSweep complete in {elapsed/60:.1f} min.")

    # ── Heatmap ───────────────────────────────────────────────────────────────
    ampli_labels = [f"{a/math.pi:.3g}π" for a in AMPLI_DICT]
    freq_labels  = [f"{f}" for f in FREQ_DICT]
    plot_heatmap(
        heatmap_mean, heatmap_std,
        ampli_labels, freq_labels,
        out_path="datafile/sweep_alignment_heatmap.png",
    )


# =============================================================================
# Helpers
# =============================================================================

def _save_config_snapshot(out_dir, omega, ampli_deg, init_phases):
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
    snapshot["OMEGA_NOM1"]  = omega
    snapshot["OMEGA_NOM2"]  = omega
    snapshot["A_DEG_NOM1"]  = ampli_deg
    snapshot["A_DEG_NOM2"]  = ampli_deg
    snapshot["INIT_PHASES"] = [list(p) for p in init_phases]
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_snapshot.json"), "w") as f:
        json.dump(snapshot, f, indent=2)


if __name__ == "__main__":
    main()

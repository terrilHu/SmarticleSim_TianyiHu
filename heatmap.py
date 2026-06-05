import math
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep import plot_heatmap, AMPLI_DICT, FREQ_DICT
from pos_all_alignment import process_pos_all

# =============================================================================
# Parameters
# =============================================================================

SUMMARY_CSV  = "C:/Users/tianyihu/Pictures/Camera Roll/0604_synchronized/datafile/sweep_summary.csv"
OUT_IMAGE    = "C:/Users/tianyihu/Pictures/Camera Roll/0604_synchronized/datafile/sweep_alignment_heatmap_160.png"
MAX_DIST     = 160.0     # Voronoi adjacency distance threshold

CMAP  = "plasma"
VMIN  = 0.4
VMAX  = 0.8

LAST_N_SECONDS = 10.0
FRAMES_PER_SEC = 60
LAST_N_FRAMES  = int(LAST_N_SECONDS * FRAMES_PER_SEC)   # 600


# =============================================================================
# Recompute alignment from POS_ALL for all trials
# =============================================================================

def recompute_alignment(summary_csv: str, max_dist: float):
    """
    For every trial in summary_csv, re-run process_pos_all on the raw
    POS_ALL.csv and overwrite the existing _alignment.csv.
    Returns the summary DataFrame with updated align_last10s_mean/std columns.
    """
    df = pd.read_csv(summary_csv)
    # Derive base datafile directory from the location of summary_csv
    base_dir = os.path.dirname(os.path.abspath(summary_csv))

    means, stds = [], []
    for row in df.itertuples():
        trial_dir = os.path.join(
            base_dir,
            row.exp_name,
            f"trial_{int(row.local_tid):04d}",
        )
        pos_all_csv  = os.path.join(trial_dir, "trial_0000_POS_ALL.csv")
        alignment_csv = os.path.join(trial_dir, "trial_0000_alignment.csv")

        if not os.path.isfile(pos_all_csv):
            print(f"[WARN] Missing: {pos_all_csv}")
            means.append(float("nan"))
            stds.append(float("nan"))
            continue

        # Recompute and overwrite alignment CSV
        align_df = process_pos_all(pos_all_csv, max_dist)
        align_df.to_csv(alignment_csv, index=False)

        # Extract last-10s statistics
        vals   = align_df["mean_alignment"].to_numpy(dtype=float)
        window = vals[-LAST_N_FRAMES:]
        if len(window) == 0:
            means.append(float("nan"))
            stds.append(float("nan"))
        else:
            means.append(float(np.mean(window)))
            stds.append(float(np.std(window)))

        print(f"  {row.exp_name} / trial_{int(row.local_tid):04d}  "
              f"mean={means[-1]:.3f}  std={stds[-1]:.3f}")

    df["align_last10s_mean"] = means
    df["align_last10s_std"]  = stds

    # Recompute combo-level statistics (mean of per-trial means)
    combo_stats = (
        df.groupby(["ampli_slot", "freq_slot"])["align_last10s_mean"]
        .agg(combo_align_mean="mean", combo_align_std="std")
        .reset_index()
    )
    df = df.drop(columns=["combo_align_mean", "combo_align_std"], errors="ignore")
    df = df.merge(combo_stats, on=["ampli_slot", "freq_slot"], how="left")

    # Overwrite summary CSV with updated values
    df.to_csv(summary_csv, index=False)
    print(f"\nUpdated summary CSV → {summary_csv}")
    return df


# =============================================================================
# Build heatmap matrices from summary DataFrame
# =============================================================================

def build_heatmap_matrices(df):
    n_a = len(AMPLI_DICT)
    n_f = len(FREQ_DICT)
    heatmap_mean = np.full((n_f, n_a), float("nan"))
    heatmap_std  = np.full((n_f, n_a), float("nan"))

    for row in df.itertuples():
        a_idx = int(row.ampli_slot) - 1
        f_idx = int(row.freq_slot)  - 1
        heatmap_mean[f_idx, a_idx] = row.combo_align_mean
        heatmap_std [f_idx, a_idx] = row.combo_align_std

    return heatmap_mean, heatmap_std


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print(f"Recomputing alignment from POS_ALL (max_dist={MAX_DIST})...\n")
    df = recompute_alignment(SUMMARY_CSV, MAX_DIST)

    heatmap_mean, heatmap_std = build_heatmap_matrices(df)

    ampli_labels = [f"{a/math.pi:.3g}π" for a in AMPLI_DICT]
    freq_labels  = [f"{f}" for f in FREQ_DICT]

    plot_heatmap(
        heatmap_mean, heatmap_std,
        ampli_labels, freq_labels,
        out_path=OUT_IMAGE,
        cmap=CMAP, vmin=VMIN, vmax=VMAX,
    )

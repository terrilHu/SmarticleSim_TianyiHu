"""
pos_all_alignment.py
---------------------
Process _POS_ALL.csv output from simulation, perform Voronoi adjacency analysis per frame,
compute per-robot alignment with neighbors, and output a global alignment time series.

Usage:
    python pos_all_alignment.py --input trial_0000_seed_0_POS_ALL.csv \
                                 --max_dist 60.0 \
                                 --output alignment_output.csv
"""

import argparse
import math
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


# =============================================================================
# Voronoi adjacency matrix (via Delaunay triangulation, equivalent to MATLAB version)
# =============================================================================

def voronoi_adjacency(positions: np.ndarray, max_dist: float):
    """
    Input:
        positions  - (N, 2) (x, y) coordinates of all robots in the current frame, may contain NaN
        max_dist   - distance threshold; pairs beyond this are set False in adjacency matrix

    Output:
        adj  - (N, N) symmetric boolean adjacency matrix
    """
    N = len(positions)
    adj = np.zeros((N, N), dtype=bool)

    # Filter NaN entries
    valid_mask = np.isfinite(positions).all(axis=1)
    valid_idx  = np.where(valid_mask)[0]
    pts        = positions[valid_idx]   # (M, 2)

    if len(pts) < 2:
        return adj

    try:
        tri = Delaunay(pts)
    except Exception:
        return adj

    # Extract unique edges from triangles (vectorised; same edge set as the
    # previous Python set-of-tuples, just built with NumPy).
    simplices = tri.simplices          # (T, 3)
    edges = np.concatenate((simplices[:, [0, 1]],
                            simplices[:, [1, 2]],
                            simplices[:, [0, 2]]), axis=0)
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)

    diff = pts[edges[:, 0]] - pts[edges[:, 1]]
    d    = np.sqrt(diff[:, 0] ** 2 + diff[:, 1] ** 2)
    keep = edges[d <= max_dist]

    ii = valid_idx[keep[:, 0]]
    jj = valid_idx[keep[:, 1]]
    adj[ii, jj] = True
    adj[jj, ii] = True

    return adj


# =============================================================================
# Single alignment calculation (equivalent to MATLAB singleAlignment)
# Input angles in radians (unlike the MATLAB version which accepts degrees)
# =============================================================================

def single_alignment(angles_rad: np.ndarray) -> float:
    """
    Compute alignment (vector order parameter) for a set of orientation angles.
    Equivalent to MATLAB singleAlignment, but input is in radians.

    alignment = |mean(exp(2i * theta))| ∈ [0, 1]
    1 means fully aligned, 0 means fully random.
    """
    if len(angles_rad) == 0:
        return 0.0
    V = np.sum(np.exp(2j * angles_rad))
    return abs(V) / len(angles_rad)


# =============================================================================
# Main processing function
# =============================================================================

def process_pos_all(csv_path: str, max_dist: float) -> pd.DataFrame:
    """
    Read _POS_ALL.csv and compute global alignment frame by frame.

    CSV format (after bug fix):
        Step, Agent_ID, X, Y, Theta
        0, 1, 312.14, 256.78, 1.2341
        ...

    Returns:
        DataFrame with columns: [Step, mean_alignment]
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    n_agents   = df["Agent_ID"].nunique()
    results    = []

    # The original re-scanned the whole table once per step
    # (df[df["Step"] == step]), i.e. O(frames * rows) = O(frames^2 * n) work.
    # Sorting once by (Step, Agent_ID) makes every frame a contiguous slice,
    # which yields exactly the same per-frame arrays in the same order.
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")
    step_col = df["Step"].to_numpy()
    xy_all   = df[["X", "Y"]].to_numpy(dtype=float)
    th_all   = df["Theta"].to_numpy(dtype=float)

    steps, starts = np.unique(step_col, return_index=True)
    bounds = np.append(starts, len(step_col))

    for _k, step in enumerate(steps):
        sl = slice(bounds[_k], bounds[_k + 1])

        positions = xy_all[sl]                                 # (N, 2)
        thetas    = th_all[sl]                                 # (N,) radians

        # 1. Voronoi adjacency matrix
        adj = voronoi_adjacency(positions, max_dist)
        # if step == 1:
        #     print(adj)

        # 2. Alignment of each robot with itself and its neighbors
        agent_alignments = []
        for i in range(n_agents):
            neighbor_idx = np.where(adj[i])[0]
            angles = np.concatenate([[thetas[i]], thetas[neighbor_idx]])
            a = single_alignment(angles)
            agent_alignments.append(a)

        # 3. Average across all robots
        mean_align = float(np.mean(agent_alignments))
        results.append({"Step": step, "mean_alignment": mean_align})

    return pd.DataFrame(results)


# =============================================================================
# Entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compute alignment time series from POS_ALL data")
    parser.add_argument("--input",    required=True,        help="path to _POS_ALL.csv")
    parser.add_argument("--max_dist", type=float, default=60.0, help="Voronoi adjacency distance threshold")
    parser.add_argument("--output",   default="alignment_timeseries.csv", help="output CSV path")
    args = parser.parse_args()

    print(f"Loading: {args.input}")
    result_df = process_pos_all(args.input, args.max_dist)

    result_df.to_csv(args.output, index=False)
    print(f"Done, total {len(result_df)} frames, saved to: {args.output}")
    print(result_df.describe())


if __name__ == "__main__":
    main()

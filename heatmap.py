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
OUT_IMAGE    = "C:/Users/tianyihu/Pictures/Camera Roll/0604_synchronized/datafile/sweep_alignment_heatmap_150.png"
MAX_DIST     = 150.0     # Voronoi adjacency distance threshold

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
# Voronoi overlay video
# =============================================================================

def render_voronoi_video(
    exp_name: str,
    trial_tid: int,
    base_dir: str,
    out_path: str,
    max_dist: float = MAX_DIST,
    fps: int = 60,
    video_stride = 2
):
    """
    Render a new video with Voronoi overlay and live alignment annotation.

    For each frame:
      - draws the original video frame as background
      - overlays Voronoi edges:
          white  = adjacent pair within max_dist
          red    = adjacent pair beyond max_dist (Delaunay neighbour but far)
      - draws each robot centre as a filled yellow circle
      - prints current mean alignment in the top-left corner

    Args:
        exp_name  : experiment folder name (e.g. 'trial_N17_J1p5_...')
        trial_tid : local trial index (0-based)
        base_dir  : root directory that contains exp_name/ and the videos/ folder
        out_path  : full path for the output video file
        max_dist  : Voronoi adjacency distance threshold (pixels)
        fps       : output video frame rate
    """
    import cv2
    from scipy.spatial import Delaunay
    from pos_all_alignment import voronoi_adjacency, single_alignment

    trial_dir   = os.path.join(base_dir, exp_name, f"trial_{trial_tid:04d}")
    pos_csv     = os.path.join(trial_dir, "trial_0000_POS_ALL.csv")
    # video_in    = os.path.join(base_dir, "..", "videos", exp_name,
    #                            f"trial_{trial_tid:04d}.mp4")
    video_in    = os.path.join(base_dir, "..", "videos", exp_name,
                               f"trial_0000.mp4")

    if not os.path.isfile(pos_csv):
        raise FileNotFoundError(f"POS_ALL not found: {pos_csv}")
    if not os.path.isfile(video_in):
        raise FileNotFoundError(f"Video not found: {video_in}")

    # ── Load position data ────────────────────────────────────────────────────
    df      = pd.read_csv(pos_csv)
    df.columns = df.columns.str.strip()
    steps   = sorted(df["Step"].unique())
    n_agents = df["Agent_ID"].nunique()

    # Pre-compute per-frame alignment
    align_series = []
    for step in steps:
        frame    = df[df["Step"] == step].sort_values("Agent_ID")
        positions = frame[["X", "Y"]].to_numpy(dtype=float)
        thetas    = frame["Theta"].to_numpy(dtype=float)
        adj       = voronoi_adjacency(positions, max_dist)
        vals      = []
        for i in range(n_agents):
            nb   = np.where(adj[i])[0]
            angs = np.concatenate([[thetas[i]], thetas[nb]])
            vals.append(single_alignment(angs))
        align_series.append(float(np.mean(vals)))

    # ── Open input video ──────────────────────────────────────────────────────
    cap     = cv2.VideoCapture(video_in)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    vid_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_vid_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # ── Open output video ─────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, vid_fps, (vid_w, vid_h))

    print(f"Rendering Voronoi overlay: {len(steps)} data frames, "
          f"{n_vid_frames} video frames → {out_path}")

    # Map each video frame index to the nearest data step index
    # (video may have been recorded at VIDEO_STRIDE, data is every frame)
    def nearest_step_idx(vfi):
        return min(vfi * video_stride, len(steps) - 1)

    vfi = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        si       = nearest_step_idx(vfi)
        step     = steps[si]
        frame_df = df[df["Step"] == step].sort_values("Agent_ID")
        positions = frame_df[["X", "Y"]].to_numpy(dtype=float)
        thetas    = frame_df["Theta"].to_numpy(dtype=float)
        alignment = align_series[si]

        # ── Compute Delaunay edges and classify ───────────────────────────────
        valid_mask = np.isfinite(positions).all(axis=1)
        valid_idx  = np.where(valid_mask)[0]
        pts        = positions[valid_idx]

        adj        = voronoi_adjacency(positions, max_dist)   # (N, N) full adjacency
        near_edges = []   # (p1, p2) within max_dist → white
        far_edges  = []   # (p1, p2) beyond max_dist → red

        if len(pts) >= 3:
            try:
                tri      = Delaunay(pts)
                edge_set = set()
                for s in tri.simplices:
                    for a, b in [(s[0],s[1]), (s[1],s[2]), (s[0],s[2])]:
                        edge_set.add((min(a,b), max(a,b)))
                for li, lj in edge_set:
                    p1 = tuple(pts[li].astype(int))
                    p2 = tuple(pts[lj].astype(int))
                    d  = np.linalg.norm(pts[li] - pts[lj])
                    if d <= max_dist:
                        near_edges.append((p1, p2))
                    else:
                        far_edges.append((p1, p2))
            except Exception:
                pass

        # ── Draw edges ────────────────────────────────────────────────────────
        overlay = bgr.copy()
        for p1, p2 in near_edges:
            cv2.line(overlay, p1, p2, (255, 255, 255), 1, cv2.LINE_AA)
        for p1, p2 in far_edges:
            cv2.line(overlay, p1, p2, (60, 60, 255), 1, cv2.LINE_AA)

        # ── Draw robot centres and per-robot alignment labels ───────────────
        for li, pos in enumerate(pts):
            cx, cy = tuple(pos.astype(int))
            cv2.circle(overlay, (cx, cy), 5, (0, 220, 255), -1, cv2.LINE_AA)

            # Per-robot alignment
            global_i = valid_idx[li]
            nb       = np.where(adj[global_i])[0]
            angs     = np.concatenate([[thetas[global_i]], thetas[nb]])
            robot_align = single_alignment(angs)
            robot_label = f"{robot_align:.2f}"

            # Draw label just above the centre dot
            cv2.putText(overlay, robot_label, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, robot_label, (cx + 6, cy - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 200), 1, cv2.LINE_AA)

        # ── Blend overlay onto background ─────────────────────────────────────
        bgr = cv2.addWeighted(overlay, 0.85, bgr, 0.15, 0)

        # ── Alignment text ────────────────────────────────────────────────────
        label = f"Alignment: {alignment:.3f}"
        cv2.putText(bgr, label, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, label, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(bgr)
        vfi += 1

    cap.release()
    writer.release()
    print(f"Done → {out_path}")



# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # print(f"Recomputing alignment from POS_ALL (max_dist={MAX_DIST})...\n")
    # df = recompute_alignment(SUMMARY_CSV, MAX_DIST)

    # heatmap_mean, heatmap_std = build_heatmap_matrices(df)

    # ampli_labels = [f"{a/math.pi:.3g}π" for a in AMPLI_DICT]
    # freq_labels  = [f"{f}" for f in FREQ_DICT]

    # plot_heatmap(
    #     heatmap_mean, heatmap_std,
    #     ampli_labels, freq_labels,
    #     out_path=OUT_IMAGE,
    #     cmap=CMAP, vmin=VMIN, vmax=VMAX,
    # )
    render_voronoi_video(
        exp_name  = "trial_N17_J1p0_J2p0_W1f8_W2f8_A1a1_A2a1",
        trial_tid = 2,
        base_dir  = r"C:/Users/tianyihu/Pictures/Camera Roll/0604_unsync/datafile",
        out_path  = r"videos/voronoi_trial2.mp4",
        max_dist  = 150.0,
    )

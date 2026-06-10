"""
pos_all_grouping.py
-------------------
Group-level alignment analysis built ON TOP of pos_all_alignment.py
(the original file is imported, never modified).

Per-frame pipeline:
    1. Voronoi (Delaunay) + max_dist adjacency   -> reuse voronoi_adjacency()
    2. Connected components of that adjacency     -> groups (clusters)
    3. Per group: member Agent_IDs, size, alignment, centroid

The analysis is LABEL-INVARIANT: it never assumes a group keeps its identity
across frames, so it needs no frame-to-frame correspondence.  Each group's
member Agent_IDs are nonetheless stored in the output table, which is the only
thing a tracking layer needs.  `track_groups()` (optional, unused by default)
can therefore be bolted on later WITHOUT recomputing the grouping.

Primary steady-state output is A(n): mean alignment as a function of group
size, pooled over a trailing time window.

Alignment convention is inherited unchanged from
pos_all_alignment.single_alignment: nematic / head-tail symmetric,
|<exp(2i*theta)>|, in [0, 1].

Usage:
    python pos_all_grouping.py --input trial_0000_POS_ALL.csv \
                               --max_dist 150 --fps 60 --last_seconds 10 \
                               --out_prefix trial_0000 --plot
"""

import argparse
import os

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# Reuse the original, unmodified routines.
from pos_all_alignment import voronoi_adjacency, single_alignment


# Separator used to serialise a member-id list into a single CSV cell.
_ID_SEP = " "


# =============================================================================
# Core: per-frame grouping
# =============================================================================

def compute_frame_groups(positions: np.ndarray,
                         thetas: np.ndarray,
                         agent_ids: np.ndarray,
                         max_dist: float,
                         include_singletons: bool = True):
    """
    Partition one frame's robots into groups (connected components of the
    Voronoi+max_dist adjacency graph) and compute per-group statistics.

    Args:
        positions          : (N, 2) float array; rows may be NaN (dropped robot)
        thetas             : (N,)   float array, orientation in radians
        agent_ids          : (N,)   array of Agent_ID, same row order as positions
        max_dist           : adjacency distance threshold
        include_singletons : keep size-1 groups (isolated robots) if True

    Returns:
        list of dicts, one per group:
            local_idx : np.ndarray of row indices into this frame's arrays
            agent_ids : np.ndarray of Agent_IDs in the group
            size      : int
            alignment : float in [0, 1] (nematic, head-tail symmetric)
            centroid  : (cx, cy)
    """
    positions = np.asarray(positions, dtype=float)
    agent_ids = np.asarray(agent_ids)
    N = len(positions)
    if N == 0:
        return []

    adj = voronoi_adjacency(positions, max_dist)          # (N, N) bool
    valid = np.isfinite(positions).all(axis=1)

    # Connected components over the adjacency graph. NaN/dropped robots have no
    # edges, so they would surface as singleton components; we discard them.
    _, labels = connected_components(csr_matrix(adj), directed=False)

    groups = []
    for c in np.unique(labels):
        comp = np.where(labels == c)[0]
        comp = comp[valid[comp]]                          # drop NaN members
        if comp.size == 0:
            continue
        if comp.size == 1 and not include_singletons:
            continue
        groups.append({
            "local_idx": comp,
            "agent_ids": agent_ids[comp],
            "size": int(comp.size),
            "alignment": float(single_alignment(thetas[comp])),
            "centroid": (float(np.mean(positions[comp, 0])),
                         float(np.mean(positions[comp, 1]))),
        })
    return groups


def process_pos_all_groups(csv_path: str,
                           max_dist: float,
                           include_singletons: bool = True) -> pd.DataFrame:
    """
    Read a _POS_ALL.csv and return a long-form table with one row per group per
    frame.  Member Agent_IDs are preserved (space-separated string) so a
    tracking layer can be added later without recomputing anything.

    Returns DataFrame columns:
        Step, group_local_id, size, alignment, centroid_x, centroid_y, member_ids

    `group_local_id` is a per-frame local index (resets each frame). A future
    `track_groups()` adds a persistent `group_global_id` on top of this table.
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    steps = sorted(df["Step"].unique())
    rows = []
    for step in steps:
        frame = df[df["Step"] == step].sort_values("Agent_ID")
        positions = frame[["X", "Y"]].to_numpy(dtype=float)
        thetas = frame["Theta"].to_numpy(dtype=float)
        agent_ids = frame["Agent_ID"].to_numpy()

        for gi, g in enumerate(compute_frame_groups(positions, thetas, agent_ids,
                                                    max_dist, include_singletons)):
            rows.append({
                "Step": step,
                "group_local_id": gi,
                "size": g["size"],
                "alignment": g["alignment"],
                "centroid_x": g["centroid"][0],
                "centroid_y": g["centroid"][1],
                "member_ids": _ID_SEP.join(str(int(a)) for a in g["agent_ids"]),
            })
    return pd.DataFrame(rows)


# =============================================================================
# Steady-state aggregation (label-invariant; no correspondence needed)
# =============================================================================

def _select_window(df: pd.DataFrame, last_n_frames=None) -> pd.DataFrame:
    """Keep only the trailing `last_n_frames` distinct Steps (None -> all)."""
    if last_n_frames is None or last_n_frames <= 0:
        return df
    steps = np.sort(df["Step"].unique())
    keep = set(steps[-int(last_n_frames):])
    return df[df["Step"].isin(keep)]


def compute_alignment_vs_size(groups_df: pd.DataFrame,
                              last_n_frames=None,
                              include_singletons: bool = True) -> pd.DataFrame:
    """
    A(n): mean alignment as a function of group size, pooled over the trailing
    window of `last_n_frames` frames.

    Each sample is one group in one frame ("group-frame"). NOTE: consecutive
    frames are highly autocorrelated, so `n_groupframes` is NOT an independent
    sample count and `sem_alignment` UNDERESTIMATES the true uncertainty. Use it
    only as a rough guide, and for real error bars aggregate this table across
    independent trials/seeds.

    Returns DataFrame columns:
        size, n_groupframes, n_frames, mean_alignment, std_alignment, sem_alignment
    """
    w = _select_window(groups_df, last_n_frames)
    if not include_singletons:
        w = w[w["size"] > 1]

    out = []
    for size, sub in w.groupby("size"):
        a = sub["alignment"].to_numpy(dtype=float)
        out.append({
            "size": int(size),
            "n_groupframes": int(len(a)),
            "n_frames": int(sub["Step"].nunique()),
            "mean_alignment": float(np.mean(a)),
            "std_alignment": float(np.std(a)),
            "sem_alignment": float(np.std(a) / np.sqrt(len(a))) if len(a) else float("nan"),
        })
    return pd.DataFrame(out).sort_values("size").reset_index(drop=True)


def compute_frame_summary(groups_df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-frame, label-invariant observables.

    Returns DataFrame columns:
        Step, n_groups, max_size, mean_size, align_group_weighted, align_size_weighted
    where
        align_group_weighted = mean over groups of alignment
        align_size_weighted  = sum(size*align)/sum(size)
                             = mean over robots of their own group's alignment
    """
    rows = []
    for step, sub in groups_df.groupby("Step"):
        sizes = sub["size"].to_numpy(dtype=float)
        aligns = sub["alignment"].to_numpy(dtype=float)
        tot = sizes.sum()
        rows.append({
            "Step": step,
            "n_groups": int(len(sub)),
            "max_size": int(sizes.max()) if len(sizes) else 0,
            "mean_size": float(sizes.mean()) if len(sizes) else 0.0,
            "align_group_weighted": float(aligns.mean()) if len(aligns) else float("nan"),
            "align_size_weighted": float((sizes * aligns).sum() / tot) if tot > 0 else float("nan"),
        })
    return pd.DataFrame(rows).sort_values("Step").reset_index(drop=True)


def _block_sem(x: np.ndarray, n_blocks: int = 10) -> float:
    """SEM via block averaging -- a rough correction for time autocorrelation."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    if len(x) < n_blocks or n_blocks < 2:
        return float(np.std(x) / np.sqrt(len(x)))
    means = np.array([b.mean() for b in np.array_split(x, n_blocks)])
    return float(np.std(means, ddof=1) / np.sqrt(n_blocks))


def steady_state_summary(frame_summary: pd.DataFrame,
                         last_n_frames=None,
                         n_blocks: int = 10) -> pd.DataFrame:
    """
    Time-average the per-frame observables over the trailing window, with
    block-averaged error bars (rough autocorrelation correction). One-row frame.
    """
    w = _select_window(frame_summary, last_n_frames)
    cols = ["n_groups", "max_size", "mean_size",
            "align_group_weighted", "align_size_weighted"]
    rec = {"n_frames": int(w["Step"].nunique())}
    for c in cols:
        v = w[c].to_numpy(dtype=float)
        rec[f"{c}_mean"] = float(np.nanmean(v)) if len(v) else float("nan")
        rec[f"{c}_sem"] = _block_sem(v, n_blocks)
    return pd.DataFrame([rec])


# =============================================================================
# OPTIONAL: frame-to-frame group tracking  (interface kept for later use)
# =============================================================================
# The grouping above is computed independently per frame and is fully
# label-invariant. Because every group row keeps its member Agent_IDs, a
# persistent identity can be assigned AFTER THE FACT by member overlap, without
# recomputing the grouping. This is the only place a "correspondence table"
# lives; nothing else in the analysis depends on it.

def _parse_members(s) -> frozenset:
    s = str(s).strip()
    return frozenset(int(t) for t in s.split(_ID_SEP)) if s else frozenset()


def track_groups(groups_df: pd.DataFrame,
                 jaccard_threshold: float = 0.3) -> pd.DataFrame:
    """
    OPTIONAL add-on (NOT used by the default analysis).

    Assign a persistent `group_global_id` to each per-frame group by greedy
    member-overlap (Jaccard) matching between consecutive frames:

        For each frame, candidate (prev_group -> cur_group) matches with
        Jaccard >= threshold are taken in descending-overlap order; each
        previous id and each current group is claimed at most once. Current
        groups left unclaimed receive a fresh id (a "birth").

    Returns a copy of `groups_df` with an added `group_global_id` column.

    This relies only on the `member_ids` column already present in the output --
    exactly why that column is preserved. Tracking is threshold-sensitive and
    unstable under rapid membership churn; for split/merge event logs or
    adjacency time-smoothing, extend from here.
    """
    df = (groups_df.copy()
          .sort_values(["Step", "group_local_id"])
          .reset_index(drop=True))
    members = df["member_ids"].map(_parse_members)

    gid_col = {}
    next_id = 0
    prev = []   # list of (members_set, gid) from the previous frame

    for step in sorted(df["Step"].unique()):
        cur = [(i, members[i]) for i in df.index[df["Step"] == step]]

        # Collect all admissible (Jaccard, prev_gid, cur_position) candidates.
        candidates = []
        for pset, pgid in prev:
            for k, (_, mset) in enumerate(cur):
                u = len(mset | pset)
                if u == 0:
                    continue
                j = len(mset & pset) / u
                if j >= jaccard_threshold:
                    candidates.append((j, pgid, k))
        candidates.sort(reverse=True)              # highest overlap first

        claimed, used_cur, used_gid = {}, set(), set()
        for _, pgid, k in candidates:
            if k in used_cur or pgid in used_gid:
                continue
            claimed[k] = pgid
            used_cur.add(k)
            used_gid.add(pgid)

        new_prev = []
        for k, (i, mset) in enumerate(cur):
            if k in claimed:
                gid = claimed[k]
            else:
                gid = next_id
                next_id += 1
            gid_col[i] = gid
            new_prev.append((mset, gid))
        prev = new_prev

    df["group_global_id"] = [gid_col[i] for i in df.index]
    return df


# =============================================================================
# Plot: A(n)
# =============================================================================

def plot_alignment_vs_size(an_df: pd.DataFrame,
                           out_path: str,
                           title: str = None,
                           min_groupframes: int = 1) -> str:
    """Plot mean alignment vs group size with std error bars and sample counts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = an_df[an_df["n_groupframes"] >= min_groupframes]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(d["size"], d["mean_alignment"], yerr=d["std_alignment"],
                fmt="o-", capsize=3, lw=1.5, color="#5b2c8f", ecolor="#b6a8cf")
    for _, r in d.iterrows():
        ax.annotate(f"{int(r['n_groupframes'])}",
                    (r["size"], r["mean_alignment"]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=7, color="gray")
    ax.set_xlabel("group size  n")
    ax.set_ylabel("alignment  A(n)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or "Alignment vs group size")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# =============================================================================
# Video: group overlay  (analogous to render_voronoi_video in heatmap.py)
# =============================================================================

# Distinct BGR colours for groups (OpenCV uses BGR).
_GROUP_PALETTE = [
    (66, 135, 245), (80, 220, 100), (60, 60, 255), (245, 200, 60),
    (200, 100, 255), (40, 200, 240), (255, 140, 60), (120, 255, 200),
    (180, 120, 255), (90, 180, 90), (255, 90, 160), (60, 240, 180),
]
_SINGLETON_COLOR = (150, 150, 150)



def render_group_video(exp_name: str,
                       trial_tid: int,
                       base_dir: str,
                       out_path: str,
                       max_dist: float = 150.0,
                       fps: int = 60,
                       video_stride: int = 2,
                       include_singletons: bool = True,
                       video_path: str = None):
    """
    Render a video with group overlay, analogous to render_voronoi_video() in
    heatmap.py, but coloured by connected-component GROUP instead of annotating
    each robot's own alignment.

    Per frame:
      - background = original video frame
      - intra-group adjacency edges drawn in the group's colour
      - robot centres = filled dots in the group's colour (singletons in grey)
      - each group labelled near its centroid:  "n=<size> A=<alignment>"
      - top-left summary:  number of groups, largest size, size-weighted alignment

    Group colours come from the member set, so a stable group keeps a consistent
    colour across frames even though no tracking is performed.

    Paths: exp_name/trial_tid/base_dir locate the data CSV at
        base_dir/exp_name/trial_<tid>/trial_0000_POS_ALL.csv
    The background video is matched BY TRIAL: if `video_path` is given it is used
    verbatim; otherwise it defaults to
        base_dir/../videos/exp_name/trial_<tid>.mp4
    (i.e. trial_0002 data -> trial_0002.mp4, not the old hard-coded trial_0000).
    """
    import cv2  # imported lazily, like the original

    trial_dir = os.path.join(base_dir, exp_name, f"trial_{trial_tid:04d}")
    pos_csv = os.path.join(trial_dir, "trial_0000_POS_ALL.csv")
    if video_path is None:
        video_in = os.path.join(base_dir, "..", "videos", exp_name,
                                f"trial_{trial_tid:04d}.mp4")
    else:
        video_in = video_path

    if not os.path.isfile(pos_csv):
        raise FileNotFoundError(f"POS_ALL not found: {pos_csv}")
    if not os.path.isfile(video_in):
        raise FileNotFoundError(f"Video not found: {video_in}")

    df = pd.read_csv(pos_csv)
    df.columns = df.columns.str.strip()
    steps = sorted(df["Step"].unique())

    cap = cv2.VideoCapture(video_in)
    vid_fps = cap.get(cv2.CAP_PROP_FPS) or fps
    vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, vid_fps, (vid_w, vid_h))

    print(f"Rendering group overlay: {len(steps)} data frames, "
          f"{n_vid} video frames -> {out_path}")

    def nearest_step_idx(vfi):
        return min(vfi * video_stride, len(steps) - 1)

    vfi = 0
    while True:
        ret, bgr = cap.read()
        if not ret:
            break

        step = steps[nearest_step_idx(vfi)]
        fdf = df[df["Step"] == step].sort_values("Agent_ID")
        positions = fdf[["X", "Y"]].to_numpy(dtype=float)
        thetas = fdf["Theta"].to_numpy(dtype=float)
        agent_ids = fdf["Agent_ID"].to_numpy()

        adj = voronoi_adjacency(positions, max_dist)
        groups = compute_frame_groups(positions, thetas, agent_ids,
                                      max_dist, include_singletons)

        overlay = bgr.copy()

        # Assign each group a palette colour by its rank in this frame
        # (rank-based, so same-frame groups are GUARANTEED distinct colours).
        # Singletons always get _SINGLETON_COLOR.
        # Sort by size descending so the largest group gets the first colour.
        sorted_groups = sorted(groups, key=lambda g: -g["size"])
        palette_idx = 0
        group_colors = []
        for g in sorted_groups:
            if g["size"] == 1:
                group_colors.append(_SINGLETON_COLOR)
            else:
                group_colors.append(_GROUP_PALETTE[palette_idx % len(_GROUP_PALETTE)])
                palette_idx += 1

        # local_idx -> colour, built from group membership only
        idx_color = {}
        for g, col in zip(sorted_groups, group_colors):
            for li in g["local_idx"]:
                idx_color[int(li)] = col

        # Intra-group adjacency edges: iterate over each group's own members only,
        # so cross-group edges are structurally impossible regardless of adj content.
        for g, col in zip(sorted_groups, group_colors):
            if g["size"] < 2:
                continue
            members = list(g["local_idx"])
            for a_pos, i in enumerate(members):
                for j in members[a_pos + 1:]:
                    if adj[i, j]:
                        p1 = tuple(positions[i].astype(int))
                        p2 = tuple(positions[j].astype(int))
                        cv2.line(overlay, p1, p2, col, 1, cv2.LINE_AA)

        # robot dots
        for li, col in idx_color.items():
            cx, cy = tuple(positions[li].astype(int))
            cv2.circle(overlay, (cx, cy), 5, col, -1, cv2.LINE_AA)

        # group labels at centroids
        for g, col in zip(sorted_groups, group_colors):
            cx, cy = int(g["centroid"][0]), int(g["centroid"][1])
            txt = f"n={g['size']} A={g['alignment']:.2f}"
            cv2.putText(overlay, txt, (cx + 6, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(overlay, txt, (cx + 6, cy - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1, cv2.LINE_AA)

        bgr = cv2.addWeighted(overlay, 0.85, bgr, 0.15, 0)

        # top-left summary
        sizes = np.array([g["size"] for g in groups], dtype=float)
        aligns = np.array([g["alignment"] for g in groups], dtype=float)
        tot = sizes.sum()
        sw = float((sizes * aligns).sum() / tot) if tot > 0 else float("nan")
        summary = (f"groups: {len(groups)}   "
                   f"max: {int(sizes.max()) if len(sizes) else 0}   "
                   f"A_sw: {sw:.3f}")
        cv2.putText(bgr, summary, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(bgr, summary, (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)

        writer.write(bgr)
        vfi += 1

    cap.release()
    writer.release()
    print(f"Done -> {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Group-level alignment analysis from POS_ALL data")
    ap.add_argument("--input", required=True, help="path to _POS_ALL.csv")
    ap.add_argument("--max_dist", type=float, default=150.0, help="Voronoi adjacency threshold")
    ap.add_argument("--fps", type=float, default=60.0, help="data frame rate (Step per second)")
    ap.add_argument("--last_seconds", type=float, default=10.0,
                    help="trailing window for steady-state A(n); <=0 uses all frames")
    ap.add_argument("--exclude_singletons", action="store_true",
                    help="drop size-1 groups from A(n) and frame summary")
    ap.add_argument("--out_prefix", default="grouping", help="prefix for output CSV/PNG files")
    ap.add_argument("--plot", action="store_true", help="also save the A(n) plot PNG")
    args = ap.parse_args()

    include_singletons = not args.exclude_singletons
    last_n = int(args.last_seconds * args.fps) if args.last_seconds > 0 else None

    print(f"Loading: {args.input}  (max_dist={args.max_dist}, "
          f"window={last_n if last_n else 'all'} frames, "
          f"singletons={'kept' if include_singletons else 'dropped'})")

    groups_df = process_pos_all_groups(args.input, args.max_dist, include_singletons)
    frame_df = compute_frame_summary(groups_df)
    an_df = compute_alignment_vs_size(groups_df, last_n, include_singletons)
    steady = steady_state_summary(frame_df, last_n)

    paths = {
        "groups": f"{args.out_prefix}_groups.csv",
        "frame": f"{args.out_prefix}_frame_summary.csv",
        "an": f"{args.out_prefix}_alignment_vs_size.csv",
        "steady": f"{args.out_prefix}_steady_state.csv",
    }
    groups_df.to_csv(paths["groups"], index=False)
    frame_df.to_csv(paths["frame"], index=False)
    an_df.to_csv(paths["an"], index=False)
    steady.to_csv(paths["steady"], index=False)

    print(f"\nPer-group rows : {len(groups_df)}  -> {paths['groups']}")
    print(f"  (member_ids retained -> ready for optional track_groups())")
    print(f"Frame summary  : {len(frame_df)} frames -> {paths['frame']}")
    print(f"\nA(n)  -> {paths['an']}")
    print(an_df.to_string(index=False))
    print(f"\nSteady-state (last {last_n if last_n else 'all'} frames) -> {paths['steady']}")
    print(steady.to_string(index=False))

    if args.plot:
        png = f"{args.out_prefix}_alignment_vs_size.png"
        plot_alignment_vs_size(
            an_df, png,
            title=f"A(n), last {args.last_seconds:g}s, max_dist={args.max_dist:g}")
        print(f"\nPlot -> {png}")


if __name__ == "__main__":
    main()

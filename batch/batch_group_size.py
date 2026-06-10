"""
batch_group_size.py
-------------------
For every (freq, ampli) parameter combination in the sweep tree, pool the
last-N-seconds of group data across all trials, compute the group-size
probability distribution (fraction of groups that have size n), and:

  1. Save one bar-chart PNG per parameter combination showing P(size).
  2. Save one heatmap PNG showing, for each (freq, ampli) cell, the dominant
     group size (mode of P) and its probability, with cell colour = probability.

"Probability" here is GROUP-COUNT fraction:
    P(n) = (number of groups with size n across all frames and trials)
           / (total number of groups across all frames and trials)

Run:
    python batch_group_size.py
    python batch_group_size.py --root D:/mydata --max_dist 70 --last_seconds 10
"""

import argparse
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# ── Path setup: add the analysis-scripts folder so pos_all_grouping imports
#    work regardless of where this batch file lives.
#    Edit SCRIPTS_DIR if needed, or pass --scripts_dir on the command line.
# ---------------------------------------------------------------------------
SCRIPTS_DIR = Path(__file__).parent.parent          # default: same folder as this file


# =============================================================================
# User settings
# =============================================================================

ROOT          = r"C:/Users/tianyihu/Pictures/Camera Roll/0604_unsync"
MAX_DIST      = 90.0
FPS           = 10.0
LAST_SECONDS  = 10.0

# Bar-chart appearance
BAR_COLOR     = "#5b2c8f"
BAR_EDGE      = "#3a1a6e"

# Heatmap appearance
HEATMAP_CMAP  = "plasma"
HEATMAP_VMIN  = 0.0        # set to None for auto
HEATMAP_VMAX  = 1.0        # set to None for auto

POS_ALL_NAME  = "trial_0000_POS_ALL.csv"
OUT_SUBDIR    = "group_size_analysis"   # created under ROOT


# =============================================================================
# Helpers: directory walking
# =============================================================================

def parse_exp_name(name: str) -> dict:
    """Extract freq_slot and ampli_slot integers from a folder name like
    'trial_N17_J1p0_J2p0_W1f6_W2f6_A1a2_A2a2'.  Returns None values if not found."""
    def grab(pat):
        m = re.search(pat, name)
        return int(m.group(1)) if m else None
    return {
        "exp_name":   name,
        "freq_slot":  grab(r"W1f(\d+)"),
        "ampli_slot": grab(r"A1a(\d+)"),
    }


def iter_combos(datafile_dir: Path):
    """
    Yield (meta_dict, [list of trial_dirs]) for every exp_name folder found.
    trial_dirs that exist are included regardless of how many there are.
    """
    for exp_dir in sorted(p for p in datafile_dir.iterdir() if p.is_dir()):
        meta = parse_exp_name(exp_dir.name)
        trial_dirs = sorted(
            p for p in exp_dir.iterdir()
            if p.is_dir() and re.match(r"trial_\d+", p.name)
        )
        if trial_dirs:
            yield meta, trial_dirs


# =============================================================================
# Core: build group-size distribution from grouping CSVs (or raw POS_ALL)
# =============================================================================

def load_groups_window(trial_dirs, last_n_frames: int, max_dist: float):
    """
    For each trial_dir, load (or recompute) the grouping data and return all
    group rows within the trailing last_n_frames distinct Steps.

    Strategy:
      - If grouping_groups.csv already exists (written by batch_grouping.py),
        read it directly — fast path.
      - Otherwise fall back to running process_pos_all_groups on the raw
        POS_ALL csv — slower but self-contained.

    Returns a concatenated DataFrame with at least columns [Step, size].
    """
    from pos_all_grouping import process_pos_all_groups

    dfs = []
    for td in trial_dirs:
        cached = td / "grouping_groups.csv"
        pos_csv = td / POS_ALL_NAME

        if cached.is_file():
            df = pd.read_csv(cached, usecols=["Step", "size"])
        elif pos_csv.is_file():
            full = process_pos_all_groups(str(pos_csv), max_dist,
                                          include_singletons=True)
            df = full[["Step", "size"]]
        else:
            continue

        # Keep only the trailing window
        steps = np.sort(df["Step"].unique())
        if last_n_frames and last_n_frames < len(steps):
            keep = set(steps[-last_n_frames:])
            df = df[df["Step"].isin(keep)]

        dfs.append(df)

    if not dfs:
        return pd.DataFrame(columns=["Step", "size"])
    return pd.concat(dfs, ignore_index=True)


def compute_size_distribution(groups_df: pd.DataFrame) -> pd.Series:
    """
    Group-count fraction: P(n) = count(groups with size==n) / total groups.
    Returns a Series indexed by size, sorted ascending.
    """
    if groups_df.empty:
        return pd.Series(dtype=float)
    # counts = groups_df["size"].value_counts().sort_index()
    # return counts / counts.sum()

    counts = groups_df["size"].value_counts().sort_index()
    weighted = counts * counts.index  # size × 该size的组数 = 该size贡献的机器人数
    return weighted / weighted.sum()


# =============================================================================
# Plot 1: bar chart for one parameter combination
# =============================================================================

def plot_size_bar(prob: pd.Series, meta: dict, out_path: str,
                  last_seconds: float, max_dist: float):
    """
    Bar chart of group-size probability distribution.
    prob : Series indexed by size (int), values sum to 1.
    """
    if prob.empty:
        return

    sizes = prob.index.to_numpy(dtype=int)
    probs = prob.to_numpy(dtype=float)
    mode_size = int(sizes[np.argmax(probs)])
    mode_prob = float(probs.max())

    # Ensure x-axis is contiguous from 1 to max_size
    max_size = int(sizes.max())
    x = np.arange(1, max_size + 1)
    y = np.array([prob.get(s, 0.0) for s in x])

    fig, ax = plt.subplots(figsize=(max(5, max_size * 0.55 + 1.5), 4))
    bars = ax.bar(x, y, color=BAR_COLOR, edgecolor=BAR_EDGE, linewidth=0.6)

    # Highlight the dominant bar
    dominant_idx = mode_size - 1   # x starts at 1
    bars[dominant_idx].set_edgecolor("white")
    bars[dominant_idx].set_linewidth(1.5)

    # Annotate each bar with its probability
    for xi, yi in zip(x, y):
        if yi > 0.005:
            ax.text(xi, yi + 0.005, f"{yi:.2f}", ha="center", va="bottom",
                    fontsize=7.5, color="white" if yi > 0.15 else "#333333")

    freq  = meta.get("freq_slot",  "?")
    ampli = meta.get("ampli_slot", "?")
    ax.set_title(
        f"freq={freq}  ampli={ampli}\n"
        f"dominant: n={mode_size} ({mode_prob:.1%})  "
        f"[last {last_seconds:g}s, max_dist={max_dist:g}]",
        fontsize=9)
    ax.set_xlabel("group size  n")
    ax.set_ylabel("P(size = n)  [group-count fraction]")
    ax.set_xticks(x)
    ax.set_ylim(0, min(1.0, probs.max() * 1.25 + 0.05))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# =============================================================================
# Plot 2: heatmap of dominant group size + probability
# =============================================================================

def plot_dominant_size_heatmap(records: list, out_path: str,
                               last_seconds: float, max_dist: float,
                               cmap=HEATMAP_CMAP,
                               vmin=HEATMAP_VMIN, vmax=HEATMAP_VMAX):
    """
    records : list of dicts with keys
        freq_slot, ampli_slot, mode_size, mode_prob
    Heatmap colour = mode_prob; cell annotation = "n=<mode_size>\n<prob%>".
    """
    if not records:
        return

    df = pd.DataFrame(records).dropna(subset=["freq_slot", "ampli_slot"])
    freq_vals  = sorted(df["freq_slot"].unique())
    ampli_vals = sorted(df["ampli_slot"].unique())
    n_f = len(freq_vals)
    n_a = len(ampli_vals)

    prob_mat = np.full((n_f, n_a), np.nan)
    size_mat = np.full((n_f, n_a), np.nan)

    f_idx = {v: i for i, v in enumerate(freq_vals)}
    a_idx = {v: i for i, v in enumerate(ampli_vals)}

    for r in records:
        fi = f_idx.get(r["freq_slot"])
        ai = a_idx.get(r["ampli_slot"])
        if fi is None or ai is None:
            continue
        # Average over multiple records for the same cell (shouldn't happen,
        # but be safe).
        if np.isnan(prob_mat[fi, ai]):
            prob_mat[fi, ai] = r["mode_prob"]
            size_mat[fi, ai] = r["mode_size"]
        else:
            prob_mat[fi, ai] = (prob_mat[fi, ai] + r["mode_prob"]) / 2

    # Colour limits based on size_mat (integer group sizes)
    valid_s = size_mat[np.isfinite(size_mat)]
    _vmin = float(valid_s.min()) if len(valid_s) else 1
    _vmax = float(valid_s.max()) if len(valid_s) else 1

    fig_w = max(6, n_a * 1.1 + 2)
    fig_h = max(4, n_f * 0.9 + 1.5)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    # Colour encodes dominant group SIZE; probability written as text
    im = ax.imshow(size_mat, cmap=cmap, vmin=_vmin, vmax=_vmax,
                   aspect="auto", origin="lower")

    # Cell annotations: "n=X\nYY%"
    for fi in range(n_f):
        for ai in range(n_a):
            p = prob_mat[fi, ai]
            s = size_mat[fi, ai]
            if np.isnan(s):
                txt = "N/A"
            else:
                txt = f"n={int(s)}\n{p:.0%}"
            normed = (s - _vmin) / (_vmax - _vmin + 1e-9) if not np.isnan(s) else 0.5
            txt_color = "black" if normed > 0.55 else "white"
            ax.text(ai, fi, txt, ha="center", va="center",
                    fontsize=8.5, color=txt_color, linespacing=1.4)

    ax.set_xticks(range(n_a))
    ax.set_xticklabels([f"a{v}" for v in ampli_vals])
    ax.set_yticks(range(n_f))
    ax.set_yticklabels([f"f{v}" for v in freq_vals])
    ax.set_xlabel("amplitude slot")
    ax.set_ylabel("frequency slot")
    ax.set_title(
        f"Dominant group size  (colour = size, text = probability)\n"
        f"last {last_seconds:g}s · max_dist={max_dist:g} · group-count fraction",
        fontsize=10)

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.set_label("dominant group size", fontsize=9)
    if (_vmax - _vmin) < 20:
        import matplotlib.ticker as ticker
        cbar.locator = ticker.MaxNLocator(integer=True)
        cbar.update_ticks()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Heatmap saved -> {out_path}")


# =============================================================================
# Main batch
# =============================================================================

def run(root: Path, scripts_dir: Path, max_dist: float,
        last_seconds: float, fps: float):

    # Make the analysis scripts importable.
    sys.path.insert(0, str(scripts_dir))

    datafile = root / "datafile"
    if not datafile.is_dir():
        raise FileNotFoundError(f"datafile dir not found: {datafile}")

    out_dir = root / OUT_SUBDIR
    bar_dir = out_dir / "bar_charts"
    bar_dir.mkdir(parents=True, exist_ok=True)

    last_n_frames = int(last_seconds * fps) if last_seconds > 0 else None
    combos = list(iter_combos(datafile))
    print(f"Found {len(combos)} parameter combinations under {datafile}")
    print(f"Settings: max_dist={max_dist}  last={last_seconds:g}s "
          f"({last_n_frames} frames)  out={out_dir}\n")

    heatmap_records = []

    for k, (meta, trial_dirs) in enumerate(combos, 1):
        freq  = meta.get("freq_slot",  "?")
        ampli = meta.get("ampli_slot", "?")
        tag   = f"[{k}/{len(combos)}] f={freq} a={ampli}"

        try:
            groups_df = load_groups_window(trial_dirs, last_n_frames, max_dist)
        except Exception as e:
            print(f"{tag}  ERROR loading: {e}")
            continue

        if groups_df.empty:
            print(f"{tag}  SKIP (no data)")
            continue

        prob = compute_size_distribution(groups_df[groups_df["size"]>1])
        mode_size = int(prob.idxmax())
        mode_prob = float(prob.max())

        print(f"{tag}  n_groups={len(groups_df)}  "
              f"dominant: size={mode_size}  P={mode_prob:.3f}")

        # Bar chart
        bar_path = bar_dir / f"size_dist_f{freq}_a{ampli}.png"
        plot_size_bar(prob, meta, str(bar_path), last_seconds, max_dist)

        heatmap_records.append({
            "freq_slot":  freq,
            "ampli_slot": ampli,
            "mode_size":  mode_size,
            "mode_prob":  mode_prob,
        })

    # Heatmap
    heatmap_path = out_dir / "dominant_size_heatmap.png"
    plot_dominant_size_heatmap(
        heatmap_records, str(heatmap_path), last_seconds, max_dist)

    # Also save the heatmap data as CSV
    if heatmap_records:
        pd.DataFrame(heatmap_records).sort_values(
            ["freq_slot", "ampli_slot"]
        ).to_csv(out_dir / "dominant_size_table.csv", index=False)
        print(f"Table  saved -> {out_dir / 'dominant_size_table.csv'}")


def main():
    ap = argparse.ArgumentParser(
        description="Group-size distribution bar charts + dominant-size heatmap")
    ap.add_argument("--root",         default=ROOT)
    ap.add_argument("--scripts_dir",  default=str(SCRIPTS_DIR),
                    help="folder containing pos_all_grouping.py and pos_all_alignment.py")
    ap.add_argument("--max_dist",     type=float, default=MAX_DIST)
    ap.add_argument("--last_seconds", type=float, default=LAST_SECONDS)
    ap.add_argument("--fps",          type=float, default=FPS)
    args = ap.parse_args()

    run(Path(args.root), Path(args.scripts_dir),
        args.max_dist, args.last_seconds, args.fps)


if __name__ == "__main__":
    main()

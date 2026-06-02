# -*- coding: utf-8 -*-
"""
Unified pipeline:
  Phase 1  — LoadingData.py          : loop all trials, compute trend / t0 signals
  Phase 2  — GetThresholds4t0...py   : MinCovDet + 1D-GMM → inside / outside labels
  Phase 3  — Statistical analysis    : 6-panel comparison figure

Output files (all written to base_dir):
  inside_outside_scatter.png          ← Phase 2 ellipse scatter
  inside/<trial>_t0.png               ← per-trial dual-axis plots
  outside/<trial>_t0.png
  inside_outside_analysis.png         ← Phase 3 6-panel figure
"""

# ─────────────────────────────────────────────────────────────────────────────
# Imports
# ─────────────────────────────────────────────────────────────────────────────
import os, glob, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
from scipy.signal import savgol_filter
from scipy import stats as scipy_stats
from sklearn.decomposition import PCA
from sklearn.covariance import MinCovDet
from sklearn.mixture import GaussianMixture
from tqdm import tqdm

# ─────────────────────────────────────────────────────────────────────────────
# Configuration  ← edit here
# ─────────────────────────────────────────────────────────────────────────────
#base_dir = r"G:\Smarticles\Simulation Video\results_N10_250videos\results_N10_250results_parallel"
base_dir = r"G:\Smarticles\Simulation Video\results_N7_2_200videos\results_N7_2_200results_parallel"
N_SMARTICLES = 7
cols        = ["s" + str(i + 1) for i in range(N_SMARTICLES)]
k           = 4         # top-k impact columns to sum
tail_window = 1000       # steps used for tail-window statistics
MAX_LAG     = 900        # max lag for trend-alignment search

# ─────────────────────────────────────────────────────────────────────────────
# Geometry constants  (LoadingData – unchanged)
# ─────────────────────────────────────────────────────────────────────────────
W, H            = 900, 760
SCREEN_MARGIN   = 26
BASE_N_REF      = 5
BASE_INNER_R_REF = 140
BASE_WALL_THICK  = 30

INNER_R_UNSCALED    = int(math.sqrt(max(1, N_SMARTICLES) / BASE_N_REF) * BASE_INNER_R_REF)
outer_need_unscaled = INNER_R_UNSCALED + BASE_WALL_THICK
outer_allow         = (min(W, H) / 2.0) - SCREEN_MARGIN
SCALE               = min(1.0, outer_allow / max(1.0, outer_need_unscaled))
INNER_R             = max(20, int(INNER_R_UNSCALED * SCALE))

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 helpers  (LoadingData – unchanged)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pca_features_from_flat(row):
    centers  = row.reshape(-1, 2)
    pca      = PCA(n_components=2)
    pca.fit(centers)
    lam1     = pca.explained_variance_[0]
    lam2     = pca.explained_variance_[1]
    linearity = lam2 / (lam1 + 1e-8)
    return lam1, lam2, linearity


def expand_array(A):
    parsed = np.array([
        np.fromstring(s.strip("[]"), sep=" ")
        for s in A.flatten()
    ])
    return parsed.reshape(A.shape[0], 2 * A.shape[1])


def trend_alignment_metrics(x, trend, t0,
                             win_trend=61, win_t0=61, poly=3,
                             eps_quantile=60, max_lag=80):
    x      = np.asarray(x,     dtype=float)
    trend  = np.asarray(trend,  dtype=float)
    t0     = np.asarray(t0,    dtype=float)

    trend_s = savgol_filter(trend, window_length=win_trend, polyorder=poly)
    t0_s    = savgol_filter(t0,    window_length=win_t0,    polyorder=poly)
    dtrend  = np.gradient(trend_s, x)
    dt0     = np.gradient(t0_s, x)

    mag   = np.maximum(np.abs(dtrend), np.abs(dt0))
    eps   = np.percentile(mag, eps_quantile)
    valid = mag > eps

    best_sign, best_corr = -np.inf, -np.inf
    best_lag_sign = best_lag_corr = None

    for lag in range(-max_lag, max_lag + 1):
        if lag < 0:
            a, b = dtrend[:lag], dt0[-lag:]
            m    = valid[:lag] & valid[-lag:]
        elif lag > 0:
            a, b = dtrend[lag:], dt0[:-lag]
            m    = valid[lag:] & valid[:-lag]
        else:
            a, b, m = dtrend, dt0, valid

        if m.sum() < 5:
            continue
        s = np.mean(np.sign(a[m]) == np.sign(b[m]))
        c = np.corrcoef(a[m], b[m])[0, 1]
        if s > best_sign:  best_sign = s;  best_lag_sign = lag
        if c > best_corr:  best_corr = c;  best_lag_corr = lag

    return best_sign, best_lag_sign, best_corr, best_lag_corr


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 helpers  (GetThresholds – adapted to accept arrays, not .npy paths)
# ─────────────────────────────────────────────────────────────────────────────

def plot_ellipse_boundary(mu, Sigma, r_star, ax):
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    order   = eigvals.argsort()[::-1]
    eigvals = eigvals[order];  eigvecs = eigvecs[:, order]
    angle   = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
    width   = 2 * r_star * np.sqrt(eigvals[0])
    height  = 2 * r_star * np.sqrt(eigvals[1])
    ax.add_patch(Ellipse(xy=mu, width=width, height=height,
                         angle=angle, fill=False, linewidth=2, edgecolor="k"))


def classify_inside_outside(trend_tail_arr: np.ndarray,
                             t0_tail_arr:    np.ndarray,
                             scatter_save_path: str = None):
    """
    MinCovDet robust covariance → Mahalanobis radius →
    1-D 2-component GMM → r_star boundary.

    Returns
    -------
    labels  : int array, shape (n_trials,)   0 = inside, 1 = outside
    r_star  : float, the GMM boundary radius
    mu, Sigma : robust location / covariance (for plotting)
    """
    X        = np.column_stack((trend_tail_arr, t0_tail_arr))
    mcd      = MinCovDet().fit(X)
    mu       = mcd.location_
    Sigma    = mcd.covariance_
    Sigma_inv = np.linalg.inv(Sigma)

    # Mahalanobis radius for every trial
    d  = X - mu
    r2 = np.einsum("ij,jk,ik->i", d, Sigma_inv, d)
    r  = np.sqrt(r2)

    # 1-D GMM on radii
    gmm = GaussianMixture(n_components=2, random_state=0)
    gmm.fit(r.reshape(-1, 1))

    order   = gmm.means_.flatten().argsort()          # component 0 = inner
    means   = gmm.means_.flatten()[order]
    vars_   = gmm.covariances_.flatten()[order]
    weights = gmm.weights_.flatten()[order]
    m1, m2  = means
    s1, s2  = np.sqrt(vars_)
    w1, w2  = weights

    # Intersection of the two 1-D Gaussians
    a = 1 / (2 * vars_[0]) - 1 / (2 * vars_[1])
    b = m2 / vars_[1] - m1 / vars_[0]
    c = (m1**2 / (2 * vars_[0]) - m2**2 / (2 * vars_[1])
         + np.log((w2 * s1) / (w1 * s2)))

    roots      = np.roots([a, b, c])
    roots      = np.real(roots[np.isreal(roots)])
    candidates = roots[(roots > m1) & (roots < m2)]
    r_star     = candidates[0] if len(candidates) > 0 else 0.5 * (m1 + m2)

    labels = (r >= r_star).astype(int)   # 0 = inside, 1 = outside

    # Scatter + ellipse plot
    fig, ax = plt.subplots(figsize=(7, 5))
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels,
                         cmap="bwr", s=20, alpha=0.8)
    plot_ellipse_boundary(mu, Sigma, r_star, ax)
    ax.set_xlabel("Mean Trend  (top-k sum)")
    ax.set_ylabel("Mean t0")
    ax.set_title(f"Inside / Outside classification\n"
                 f"(r* = {r_star:.3f},  inside={( labels==0).sum()}, "
                 f"outside={(labels==1).sum()})")
    ax.set_xlim(5.0, 13.0)
    ax.set_ylim(0.0, 1.0)
    plt.colorbar(scatter, ax=ax, label="0=inside  1=outside")
    plt.tight_layout()
    if scatter_save_path:
        plt.savefig(scatter_save_path, dpi=120)
        print(f"Saved scatter → {scatter_save_path}")
    plt.close()

    return labels, r_star, mu, Sigma


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 helpers  (statistical analysis)
# ─────────────────────────────────────────────────────────────────────────────

def per_trial_stats(records: list) -> dict:
    n   = len(records)
    keys = ["t0_mean", "t0_std",
            "trend_mean", "trend_std",
            "trend_min",  "trend_max", "trend_range",
            "trend_early_std", "trend_late_std",
            "trend_slope", "t0_trend_corr"]
    out = {k: np.full(n, np.nan) for k in keys}

    for i, rec in enumerate(records):
        t0    = rec["t0_series"]
        trend = rec["trend_series"]
        T     = len(t0)
        q     = T // 4

        out["t0_mean"][i]        = t0.mean()
        out["t0_std"][i]         = t0.std()
        out["trend_mean"][i]     = trend.mean()
        out["trend_std"][i]      = trend.std()
        out["trend_min"][i]      = trend.min()
        out["trend_max"][i]      = trend.max()
        out["trend_range"][i]    = trend.max() - trend.min()
        out["trend_early_std"][i] = trend[:q].std()
        out["trend_late_std"][i]  = trend[-q:].std()
        out["trend_slope"][i]    = np.polyfit(np.linspace(0, 1, T), trend, 1)[0]
        if t0.std() > 0 and trend.std() > 0:
            out["t0_trend_corr"][i] = np.corrcoef(t0, trend)[0, 1]

    return out


def align_stack(records: list, key: str) -> np.ndarray:
    arrays = [r[key] for r in records]
    T = min(len(a) for a in arrays)
    return np.vstack([a[:T] for a in arrays])


def smooth(x: np.ndarray, w: int = 50) -> np.ndarray:
    return np.convolve(x, np.ones(w) / w, mode="same")


def ttest_annot(a: np.ndarray, b: np.ndarray):
    p   = scipy_stats.ttest_ind(a[~np.isnan(a)], b[~np.isnan(b)]).pvalue
    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
    return p, sig


def _style(ax, C_AX="#2d2d54"):
    ax.set_facecolor(C_AX)
    for sp in ["top",  "right"]:  ax.spines[sp].set_visible(False)
    for sp in ["bottom","left"]:  ax.spines[sp].set_color("#555577")
    ax.tick_params(colors="#aaaacc", labelsize=8)
    ax.xaxis.label.set_color("#aaaacc")
    ax.yaxis.label.set_color("#aaaacc")
    ax.title.set_color("#ddddff")
    ax.grid(color="#3a3a60", linestyle="--", alpha=0.5)


def run_analysis(in_records: list, out_records: list, save_path: str):
    C_IN  = "#5b9cf6"
    C_OUT = "#ff7f50"
    C_BG  = "#1a1a2e"
    C_AX  = "#2d2d54"

    print(f"\n── Statistical analysis  "
          f"(inside={len(in_records)}, outside={len(out_records)}) ──")

    in_s  = per_trial_stats(in_records)
    out_s = per_trial_stats(out_records)

    in_mat  = align_stack(in_records,  "trend_series")
    out_mat = align_stack(out_records, "trend_series")
    T       = in_mat.shape[1]
    x_norm  = np.linspace(0, 1, T)

    in_avg  = in_mat.mean(0);   in_std  = in_mat.std(0)
    out_avg = out_mat.mean(0);  out_std = out_mat.std(0)
    in_ts   = in_mat.std(0)     # cross-trial std at each step
    out_ts  = out_mat.std(0)

    # Print table
    print(f"\n{'Metric':<22}  {'Inside':>10}  {'Outside':>10}  "
          f"{'Δ':>8}  {'p':>8}  Sig")
    print("─" * 70)
    for key in in_s:
        iv, ov = in_s[key], out_s[key]
        p, sig = ttest_annot(iv, ov)
        print(f"{key:<22}  {np.nanmean(iv):>10.4f}  {np.nanmean(ov):>10.4f}"
              f"  {np.nanmean(ov)-np.nanmean(iv):>+8.4f}  {p:>8.4f}  {sig}")

    # ── Figure ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 13))
    fig.patch.set_facecolor(C_BG)
    gs  = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38,
                            left=0.07, right=0.97, top=0.93, bottom=0.06)

    # ① Average trend trajectory
    ax = fig.add_subplot(gs[0, :2]);  _style(ax, C_AX)
    ax.plot(x_norm, in_avg,  color=C_IN,  lw=2,
            label=f"Inside  (n={len(in_records)})")
    ax.plot(x_norm, out_avg, color=C_OUT, lw=2, linestyle="--",
            label=f"Outside (n={len(out_records)})")
    ax.fill_between(x_norm, in_avg  - in_std  * 0.3,
                             in_avg  + in_std  * 0.3, alpha=0.2, color=C_IN)
    ax.fill_between(x_norm, out_avg - out_std * 0.3,
                             out_avg + out_std * 0.3, alpha=0.2, color=C_OUT)
    ax.set_title("① Average Trend Trajectory  (top-k sum, Savitzky-Golay smoothed)",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalised Time Step");  ax.set_ylabel("Trend")
    ax.legend(fontsize=8, facecolor=C_AX, edgecolor="#555577", labelcolor="white")

    # ② Late-stage variability
    ax = fig.add_subplot(gs[0, 2]);  _style(ax, C_AX)
    p_l, sig_l = ttest_annot(in_s["trend_late_std"], out_s["trend_late_std"])
    bp = ax.boxplot([in_s["trend_late_std"], out_s["trend_late_std"]],
                    patch_artist=True,
                    medianprops=dict(color="white", linewidth=2),
                    whiskerprops=dict(color="#8888aa"),
                    capprops=dict(color="#8888aa"),
                    flierprops=dict(marker=".", markersize=3, alpha=0.4))
    bp["boxes"][0].set(facecolor=C_IN,  alpha=0.7)
    bp["boxes"][1].set(facecolor=C_OUT, alpha=0.7)
    ax.set_xticklabels(["Inside", "Outside"])
    ax.set_title(f"② Late-Stage Trend Variability\n(last 25% of steps)  {sig_l}",
                 fontsize=10, fontweight="bold")
    ax.set_ylabel("Std of Trend (last 25%)")
    ax.text(0.5, 0.97, f"p = {p_l:.2e}", ha="center", va="top",
            transform=ax.transAxes, color="#ffdd44", fontsize=9)

    # ③ Cross-trial variability over time
    ax = fig.add_subplot(gs[1, :2]);  _style(ax, C_AX)
    ax.plot(x_norm, smooth(in_ts),  color=C_IN,  lw=2, label="Inside")
    ax.plot(x_norm, smooth(out_ts), color=C_OUT, lw=2, label="Outside",
            linestyle="--")
    ax.set_title("③ Cross-Trial Variability of Trend Over Time  (smoothed)\n"
                 "Outside stays noisier — Inside converges",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Normalised Time Step");  ax.set_ylabel("Std Across Trials")
    ax.legend(fontsize=8, facecolor=C_AX, edgecolor="#555577", labelcolor="white")

    # ④ t0 ↔ trend correlation
    ax = fig.add_subplot(gs[1, 2]);  _style(ax, C_AX)
    p_c, sig_c = ttest_annot(in_s["t0_trend_corr"], out_s["t0_trend_corr"])
    ax.hist(in_s["t0_trend_corr"],  bins=25, alpha=0.7,
            color=C_IN,  density=True, label="Inside")
    ax.hist(out_s["t0_trend_corr"], bins=25, alpha=0.7,
            color=C_OUT, density=True, label="Outside")
    ax.axvline(np.nanmean(in_s["t0_trend_corr"]),  color=C_IN,  linestyle="--", lw=2)
    ax.axvline(np.nanmean(out_s["t0_trend_corr"]), color=C_OUT, linestyle="--", lw=2)
    ax.set_title(f"④ t0 ↔ Trend Correlation\n(p = {p_c:.2e}  {sig_c})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Pearson r  (t0 vs trend)");  ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=C_AX, edgecolor="#555577", labelcolor="white")
    ax.text(0.04, 0.96,
            f"Inside:  r = {np.nanmean(in_s['t0_trend_corr']):.3f}\n"
            f"Outside: r = {np.nanmean(out_s['t0_trend_corr']):.3f}",
            ha="left", va="top", transform=ax.transAxes,
            color="#ddddff", fontsize=8,
            bbox=dict(boxstyle="round", facecolor=C_BG, alpha=0.7))

    # ⑤ Trend minimum
    ax = fig.add_subplot(gs[2, 0]);  _style(ax, C_AX)
    p_m, sig_m = ttest_annot(in_s["trend_min"], out_s["trend_min"])
    ax.hist(in_s["trend_min"],  bins=25, alpha=0.7,
            color=C_IN,  density=True, label="Inside")
    ax.hist(out_s["trend_min"], bins=25, alpha=0.7,
            color=C_OUT, density=True, label="Outside")
    ax.axvline(np.nanmean(in_s["trend_min"]),  color=C_IN,  linestyle="--", lw=2)
    ax.axvline(np.nanmean(out_s["trend_min"]), color=C_OUT, linestyle="--", lw=2)
    ax.set_title(f"⑤ Trend Minimum Value\n(p = {p_m:.4f}  {sig_m})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Min Trend per Trial");  ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=C_AX, edgecolor="#555577", labelcolor="white")

    # ⑥ Trend total range
    ax = fig.add_subplot(gs[2, 1]);  _style(ax, C_AX)
    p_r, sig_r = ttest_annot(in_s["trend_range"], out_s["trend_range"])
    ax.hist(in_s["trend_range"],  bins=25, alpha=0.7,
            color=C_IN,  density=True, label="Inside")
    ax.hist(out_s["trend_range"], bins=25, alpha=0.7,
            color=C_OUT, density=True, label="Outside")
    ax.axvline(np.nanmean(in_s["trend_range"]),  color=C_IN,  linestyle="--", lw=2)
    ax.axvline(np.nanmean(out_s["trend_range"]), color=C_OUT, linestyle="--", lw=2)
    ax.set_title(f"⑥ Trend Total Range  (max − min)\n(p = {p_r:.4f}  {sig_r})",
                 fontsize=10, fontweight="bold")
    ax.set_xlabel("Range of Trend per Trial");  ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=C_AX, edgecolor="#555577", labelcolor="white")

    # ⑦ Summary table
    ax = fig.add_subplot(gs[2, 2])
    ax.set_facecolor(C_AX);  ax.axis("off")
    ax.set_title("Key Findings Summary", fontsize=10, fontweight="bold",
                 color="#ddddff")

    rows = []
    for key, label in [("trend_late_std",  "Trend std (late 25%)"),
                        ("t0_trend_corr",   "t0 ↔ Trend corr"),
                        ("trend_min",       "Trend min"),
                        ("trend_range",     "Trend range"),
                        ("trend_slope",     "Trend slope")]:
        p, sig = ttest_annot(in_s[key], out_s[key])
        rows.append([label,
                     f"{np.nanmean(in_s[key]):.3f}",
                     f"{np.nanmean(out_s[key]):.3f}",
                     sig])

    tbl = ax.table(cellText=rows,
                   colLabels=["Metric", "Inside", "Outside", "Sig"],
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False);  tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#2d2d54" if r > 0 else "#3a3a70")
        cell.set_edgecolor("#555577")
        if   r == 0: cell.set_text_props(color="#ddddff", fontweight="bold")
        elif c == 3: cell.set_text_props(color="#ffdd44", fontweight="bold")
        elif c == 1: cell.set_text_props(color=C_IN)
        elif c == 2: cell.set_text_props(color=C_OUT)
        else:        cell.set_text_props(color="#ccccee")

    fig.text(0.5, 0.97,
             "Inside vs Outside — Statistical Analysis of Trend & t0",
             ha="center", va="top",
             fontsize=14, fontweight="bold", color="#eeeeff")

    plt.savefig(save_path, dpi=130, facecolor=C_BG, bbox_inches="tight")
    plt.close()
    print(f"  Saved analysis → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Main loop  (LoadingData logic, with time-series storage added)
# ─────────────────────────────────────────────────────────────────────────────

trial_dirs = sorted(glob.glob(os.path.join(base_dir, "trial_*")))

best_sign_all  = []
best_lag_all   = []
best_corr_all  = []
trend_tail_all = []
t0_tail_all    = []
Mdist_tail_all = []
linearity_all, lam1_all, lam2_all = [], [], []

all_records = []   # ← stores per-trial dicts used in Phase 2 & 3

for trial_path in tqdm(trial_dirs, desc="Phase 1  loading trials"):

    impact_file = glob.glob(os.path.join(trial_path, "*_impacts.csv"))
    gr_file     = glob.glob(os.path.join(trial_path, "*_GRs_ORDs_Mdist.csv"))
    pos_file    = glob.glob(os.path.join(trial_path, "*_POS_ALL.csv"))

    if not impact_file or not gr_file or not pos_file:
        continue

    # POS → PCA
    rows    = []
    counter = 0
    with open(pos_file[0], "r") as f:
        for line in f:
            line = line.strip()
            if not line or counter < 1:
                counter += 1
                continue
            parts = line.split("],")
            parts = [p.strip() + "]" if not p.strip().endswith("]")
                     else p.strip() for p in parts]
            rows.append(parts)

    df_pos = expand_array(np.array(rows))

    pca_feats = []
    start     = max(0, df_pos.shape[0] - tail_window)
    for i in range(start, df_pos.shape[0]):
        pca_feats.append(compute_pca_features_from_flat(df_pos[i]))
    pca_feats     = np.array(pca_feats)
    lam1_all.append(pca_feats[:, 0].mean())
    lam2_all.append(pca_feats[:, 1].mean())
    linearity_all.append(pca_feats[:, 2].mean())

    # Signals
    df_imp = pd.read_csv(impact_file[0])
    df_gr  = pd.read_csv(gr_file[0])

    df_imp["top3_sum"] = np.sort(df_imp[cols].values, axis=1)[:, -k:].sum(axis=1)
    df_imp["trend"]    = savgol_filter(df_imp["top3_sum"],
                                       window_length=61, polyorder=3)
    df_gr["t0"]        = savgol_filter(df_gr["t0"],
                                       window_length=61, polyorder=3)
    t0    = df_gr["t0"].values
    x     = df_imp["time"].values
    #mdist = df_gr["t5"].values
    mdist = df_gr["t4"].values

    if len(x) < 200:
        continue

    # Trend alignment
    best_sign, best_lag, best_corr, _ = trend_alignment_metrics(
        x, df_imp["trend"].values, t0, max_lag=MAX_LAG
    )
    if np.isnan(best_sign):
        continue

    # Tail-window statistics
    start_tail = max(0, len(x) - tail_window)
    trend_vals = df_imp["trend"].values[start_tail:]
    t0_vals    = t0[start_tail:]

    trend_tail = trend_vals.mean()
    t0_tail    = t0_vals.mean()
    mdist_tail = mdist[start_tail:].mean()

    trend_tail_all.append(trend_tail)
    t0_tail_all.append(t0_tail)
    Mdist_tail_all.append(mdist_tail)
    best_sign_all.append(best_sign)
    best_lag_all.append(best_lag)
    best_corr_all.append(best_corr)

    # Store full tail time series for Phase 3
    all_records.append({
        "trial_path":   trial_path,
        "trend_series": trend_vals,   # shape (tail_window,)
        "t0_series":    t0_vals,      # shape (tail_window,)
        "trend_tail":   trend_tail,
        "t0_tail":      t0_tail,
        "trend_full":   df_imp["trend"].values,   # full series (for plot)
        "t0_full":      t0,                        # full series (for plot)
    })

# Convert to arrays
best_sign_all  = np.array(best_sign_all)
best_lag_all   = np.array(best_lag_all)
best_corr_all  = np.array(best_corr_all)
trend_tail_all = np.array(trend_tail_all)
t0_tail_all    = np.array(t0_tail_all)
Mdist_tail_all = np.array(Mdist_tail_all)

# ── original results printout (LoadingData – unchanged) ──────────────────────
print("\n===== TREND ALIGNMENT RESULT =====")
print("Mean sign agreement =",   np.nanmean(best_sign_all))
print("Median sign agreement =", np.nanmedian(best_sign_all))
print("Mean correlation =",      np.nanmean(best_corr_all))
print("Median correlation =",    np.nanmedian(best_corr_all))
print("Mean lag =",              np.nanmean(best_lag_all))
print("Median lag =",            np.nanmedian(best_lag_all))

# ── original scatter / histogram plots (LoadingData – unchanged) ─────────────
plt.figure(); plt.scatter(linearity_all, t0_tail_all)
plt.xlabel("linearity (λ2/λ1)"); plt.ylabel("t0"); plt.grid()
plt.savefig(os.path.join(base_dir, "linearity_vs_t0.png"), dpi=100); plt.close()

plt.figure(); plt.scatter(lam1_all, t0_tail_all)
plt.xlabel("λ1 (structure length)"); plt.ylabel("t0")
plt.savefig(os.path.join(base_dir, "lam1_vs_t0.png"), dpi=100); plt.close()

for data, title, fname in [
    (best_sign_all, "Trend alignment (sign agreement)", "hist_sign.png"),
    (best_lag_all,  "Lag distribution",                  "hist_lag.png"),
    (best_corr_all, "Derivative correlation",            "hist_corr.png"),
]:
    plt.figure(); plt.hist(data, bins=30)
    plt.title(title); plt.grid()
    plt.savefig(os.path.join(base_dir, fname), dpi=100); plt.close()

q1, q2 = np.percentile(trend_tail_all, [33, 66])
labels_3 = np.zeros_like(trend_tail_all, dtype=int)
labels_3[(trend_tail_all >= q1) & (trend_tail_all < q2)] = 1
labels_3[trend_tail_all >= q2] = 2
plt.figure()
for i, name in enumerate(["Low", "Mid", "High"]):
    mask = labels_3 == i
    plt.scatter(trend_tail_all[mask], t0_tail_all[mask], label=name, alpha=0.7)
plt.legend(); plt.xlabel("Mean Trend"); plt.ylabel("Mean t0")
plt.title("Impact-binned scatter"); plt.grid()
plt.savefig(os.path.join(base_dir, "impact_binned_scatter.png"), dpi=100); plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — MinCovDet + GMM classification  (GetThresholds logic)
# ─────────────────────────────────────────────────────────────────────────────
print("\n── Phase 2  GMM classification ──")

labels_io, r_star, mu, Sigma = classify_inside_outside(
    trend_tail_all,
    t0_tail_all,
    scatter_save_path=os.path.join(base_dir, "inside_outside_scatter.png")
)

# Split records by label
#   labels_io[i] corresponds to all_records[i]  (same sorted order)
inside_records  = [r for r, lab in zip(all_records, labels_io) if lab == 0]
outside_records = [r for r, lab in zip(all_records, labels_io) if lab == 1]

print(f"  inside  = {len(inside_records)},  outside = {len(outside_records)}")

# Save per-trial dual-axis plots to inside/ and outside/ folders
# (GetThresholds logic – unchanged except plt.show() → plt.savefig())
outputdir_in  = os.path.join(base_dir, "inside")
outputdir_out = os.path.join(base_dir, "outside")
os.makedirs(outputdir_in,  exist_ok=True)
os.makedirs(outputdir_out, exist_ok=True)

outside_paths = {r["trial_path"] for r in outside_records}

print("── Saving per-trial plots ──")
for rec in tqdm(all_records, desc="Phase 2  saving plots"):
    fig, ax1 = plt.subplots()
    ax1.plot(rec["t0_full"])
    ax1.set_xlabel("Step");  ax1.set_ylabel("t0")
    ax2 = ax1.twinx()
    ax2.plot(rec["trend_full"], "r-");  ax2.set_ylabel("trend")

    save_name = os.path.basename(rec["trial_path"]) + "_t0.png"
    save_dir  = outputdir_out if rec["trial_path"] in outside_paths else outputdir_in
    plt.savefig(os.path.join(save_dir, save_name))
    plt.close()

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────
run_analysis(
    inside_records,
    outside_records,
    save_path=os.path.join(base_dir, "inside_outside_analysis.png")
)

print("\n✓ All done.")
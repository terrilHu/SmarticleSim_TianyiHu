"""
catalyst_radial_figs.py
-----------------------
Figures for the catalyst-position hypotheses, from catalyst_radial_build.py's cache
and catalyst_radial_stats.py's csvs. Writes fig*.png/.svg, raw_groups_per_trial.csv,
ccdf_raw.npz and catalyst_role_profile.csv into <out>.

H3 is computed on the raw (unsmoothed) grouping, the same one a862_trial_stats.csv
and the mean_composition plots use; the spatial figures use the smoothed grouping
the tracks are built on.

Usage:
    python catalyst_radial_figs.py <out> <baseline_dir> NAME=DIR [NAME=DIR ...] [--results DIR]

<out> holds cache/; the metrics csvs are read from, and figures written to, --results
(default <out>). One or more strategies; NAME picks its fixed colour and marker.
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import catalyst_radial_stats as crs

INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
# a condition keeps its colour and marker across every figure set
COL = {"baseline": INK2, "farthest": "#eb6834", "nearest": "#2a78d6", "major_end": "#1baf7a", "x_end": "#6250d6", "major_end_36": "#eda100", "xend_36": "#6250d6",
       "fixed_36": "#d6336c"}
NAME = {c: c for c in COL}
MARK = {"baseline": "o", "farthest": "^", "nearest": "s", "major_end": "D", "x_end": "v", "major_end_36": "X", "xend_36": "v", "fixed_36": "P"}
STRATS = ["farthest", "nearest"]
CONDS = ["baseline"] + STRATS
EDGE = crs.rho_edge(crs.OUTER_K)
T_EV = 100.0

plt.rcParams.update({"font.family": ["Microsoft YaHei", "DejaVu Sans"], "axes.unicode_minus": False, "font.size": 10,
                     "axes.edgecolor": INK3, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlecolor": INK,
                     "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
                     "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})


ALSO_BARE = False    # save() also writes <name>_bare: same panels, no text at all


def strip_text(fig):
    """Remove every piece of text, keeping marks, axis lines, tick marks, grid and panel geometry."""
    fig.canvas.draw()
    fig.set_layout_engine("none")       # freeze: the panels must not grow into the freed space
    for ax in fig.axes:
        for t in (ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label):
            t.set_text("")
        ax.tick_params(which="both", labelbottom=False, labeltop=False, labelleft=False, labelright=False)
        for t in list(ax.texts):
            t.remove()
        if ax.get_legend():
            ax.get_legend().remove()
    fig.legends.clear()
    if fig._suptitle:
        fig._suptitle.set_text("")


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".png"), dpi=170, bbox_inches="tight")
    fig.savefig(os.path.join(out, name + ".svg"), bbox_inches="tight")
    if ALSO_BARE:
        strip_text(fig)
        fig.savefig(os.path.join(out, name + "_bare.png"), dpi=170, bbox_inches="tight")
        fig.savefig(os.path.join(out, name + "_bare.svg"), bbox_inches="tight")
    plt.close(fig)


def cache(out, exp, tn):
    return os.path.join(out, "cache", f"{exp}__trial_{tn:04d}.npz")


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def legend_all(ax, **kw):
    ax.legend(frameon=False, **kw)


# ---------------------------------------------------------------- raw grouping for H3
def raw_h3(dirs, out, fps=60, N=500):
    path = os.path.join(out, "raw_groups_per_trial.csv")
    cc_path = os.path.join(out, "ccdf_raw.npz")
    ts_path = os.path.join(out, "raw_timeseries.csv")
    if all(os.path.exists(p) for p in (path, cc_path, ts_path)):
        z = np.load(cc_path)
        return pd.read_csv(path), {k: z[k] for k in z.files}, pd.read_csv(ts_path)
    rows, cc, ts = [], {c: [] for c in CONDS}, []
    for c in CONDS:
        for tn in range(5):
            t = f"trial_{tn:04d}"
            df = pd.read_csv(os.path.join(dirs[c], t, f"{t}_groups_d65.csv"), usecols=["Step", "size"],
                             dtype={"Step": np.int32, "size": np.int16})
            st, sz = df.Step.to_numpy(), df["size"].to_numpy().astype(np.int64)
            T = st.max() + 1
            largest = np.zeros(T, np.int64)
            np.maximum.at(largest, st, sz)
            per = lambda k: np.bincount(st[sz >= k], weights=sz[sz >= k].astype(float), minlength=T) / N
            s3, s6, s10, s20 = per(3), per(6), per(10), per(20)
            for w, t0 in crs.WINDOWS.items():
                a = int(np.ceil(t0 * fps))
                rows.append(dict(cond=c, trial=t, window=w, share_ge3=s3[a:].mean(), share_ge6=s6[a:].mean(),
                                 share_ge10=s10[a:].mean(), share_ge20=s20[a:].mean(), largest_mean=largest[a:].mean(),
                                 largest_peak=largest[a:].max(),
                                 mean_size=(np.bincount(st, weights=sz.astype(float) ** 2, minlength=T) / N)[a:].mean()))
            a = int(np.ceil(T_EV * fps))
            k = st >= a
            h = np.bincount(np.minimum(sz[k], 80), weights=sz[k].astype(float), minlength=81)[1:] / (N * (T - a))
            cc[c].append(h[::-1].cumsum()[::-1])
            bin_w = 25 * fps
            for b in range(0, T - bin_w + 1, bin_w):
                ts.append(dict(cond=c, trial=t, t=(b + bin_w / 2) / fps, share_ge6=s6[b:b + bin_w].mean(),
                               share_ge3=s3[b:b + bin_w].mean(), largest=largest[b:b + bin_w].mean()))
            print("raw", c, t, flush=True)
    R, TS = pd.DataFrame(rows), pd.DataFrame(ts)
    R.to_csv(path, index=False)
    TS.to_csv(ts_path, index=False)
    cc = {c: np.array(v) for c, v in cc.items()}
    np.savez(cc_path, **cc)
    return R, cc, TS


# ---------------------------------------------------------------- spatial reductions
def cond_caches(out, exps):
    """Every cached trial of one condition; exps is one experiment name or a list to pool."""
    names = [exps] if isinstance(exps, str) else exps
    paths = [p for e in names for p in sorted(glob.glob(os.path.join(out, "cache", f"{e}__trial_*.npz")))]
    if not paths:
        raise SystemExit(f"no cached trials for {names} in {out}/cache")
    return paths


def spatial_reduction(exps, out, t0=T_EV):
    """Per-trial radial profiles, 2D large-group maps, time series and KM inputs."""
    res = {c: dict(tot=[], large=[], cat=[], ung=[], cat_ung=[], ts_o=[], ts_i=[], map=[], km=[]) for c in CONDS}
    ed = np.linspace(-1.1, 1.1, 45)
    for c in CONDS:
        for tn, path in enumerate(cond_caches(out, exps[c])):
            d, f, fps = crs.load_trial(path)
            T, R = int(d["T"]), float(d["R"])
            a = int(np.ceil(t0 * fps))
            hh = d["hist_swarm"]
            h = hh[a:].sum(axis=0).astype(float) / (T - a)
            res[c]["tot"].append(h.sum(axis=(1, 2)))
            res[c]["large"].append(h[:, 2:].sum(axis=(1, 2)))
            res[c]["cat"].append(h[:, :, 1].sum(axis=1))
            res[c]["ung"].append(h[:, 0].sum(axis=1))
            res[c]["cat_ung"].append(h[:, 0, 1])
            bw = int(25 * fps)
            nb = T // bw
            lg = hh[:nb * bw, :, 2:].sum(axis=(2, 3)).astype(float)
            res[c]["ts_o"].append(lg[:, crs.OUTER_K:].sum(1).reshape(nb, bw).mean(1) / d["N"])
            res[c]["ts_i"].append(lg[:, :crs.OUTER_K].sum(1).reshape(nb, bw).mean(1) / d["N"])
            w = f[(f.t >= a) & (f["size"] >= crs.LARGE)]
            sc = d["swarm_c"][w.t.to_numpy()]
            xy = (d["frames_xy"][(f.t >= a) & (f["size"] >= crs.LARGE)] - sc) / R
            res[c]["map"].append(np.histogram2d(xy[:, 0], xy[:, 1], bins=[ed, ed])[0] / (T - a))
            lt = crs.large_track_table(f, fps)
            ign = lt[lt.formed & (lt.t_ign >= a)]
            res[c]["km"].append(pd.DataFrame(dict(dur=(ign.t_end - ign.t_ign + 1) / fps, obs=ign.t_end < T - 1,
                                                  outer=ign.rho_ign >= EDGE, trial=tn)))
            print("spatial", c, tn, flush=True)
    return res, ed


# ---------------------------------------------------------------- figures
def fig_forest(C, out):
    """Relative change vs baseline with 95% CI: the whole argument on one page."""
    rows = [("H1 / H2  外圈", None),
            ("外圈大 group 形成速率", ("ignitions_outer_per_min", "t>=100")),
            ("外圈大 group 数量", ("n_large_outer", "t>=100")),
            ("外圈中处于大 group 的机器人比例", ("large_frac_in_outer", "t>=100")),
            ("大 group 机器人落在外圈的占比", ("outer_share_of_large_robots", "t>=100")),
            ("外圈大 group 衰减率（越高越不稳定）", ("decay_per_groupmin_outer", "t>=100")),
            ("外圈形成的大 group 的中位寿命", ("life_after_ign_outer_s", "t>=100")),
            ("对照  内圈", None),
            ("内圈大 group 数量", ("n_large_inner", "t>=100")),
            ("内圈大 group 衰减率", ("decay_per_groupmin_inner", "t>=100"))]
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True, constrained_layout=True)
    for ax, win, title in ((axs[0], "t>=100", "t ≥ 100 s（含过渡期，事件数多）"), (axs[1], "t>=400", "t ≥ 400 s（接近稳态，事件数少）")):
        y, yt, yl = 0, [], []
        for lab, key in rows:
            if key is None:
                if ax is axs[0]:
                    ax.text(-0.01, y, lab, transform=ax.get_yaxis_transform(), ha="right", va="center",
                            fontweight="bold", color=INK)
                y -= 1
                continue
            yt.append(y)
            # a median that is never reached drops that trial, so say when a row has fewer pairs
            nmin = C[(C.window == key[1]) & (C.metric == key[0]) & C.strategy.isin(STRATS)].n.min()
            yl.append(lab if nmin >= 5 else f"{lab}（n = {int(nmin)}）")
            # a 100 s window censors every lifetime at <= 100 s, so that row is only read at t >= 100
            if not (win == "t>=400" and key[0].startswith("life_")):
                n = len(STRATS)
                for off, c in zip(0.17 * (n - 1 - 2 * np.arange(n)) / max(n - 1, 1), STRATS):
                    r = C[(C.window == win) & (C.strategy == c) & (C.metric == key[0])].iloc[0]
                    lo, hi, mid = (100 * v / r.baseline for v in (r.ci_lo, r.ci_hi, r["diff"]))
                    ax.plot([np.clip(lo, -140, 300), np.clip(hi, -140, 300)], [y + off] * 2, color=COL[c], lw=2,
                            solid_capstyle="round", zorder=2)
                    ax.plot(mid, y + off, MARK[c], color=COL[c], ms=8, mec=SURFACE, mew=1.5, zorder=3,
                            label=NAME[c] if key[0] == "ignitions_outer_per_min" else None)
            else:
                ax.text(0, y, "窗口太短，寿命被截断", ha="center", va="center", color=INK2, fontsize=9,
                        bbox=dict(fc=SURFACE, ec="none", pad=2))
            y -= 1
        ax.axvline(0, color=INK3, lw=1)
        ax.set_yticks(yt, yl)
        ax.set_title(title)
        ax.set_xlabel("相对 baseline 的变化（%），配对差均值 ± 95% CI，除标注外 n = 5 对")
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        grid(ax, "x")
        ax.set_xlim(-140, 300)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="outside lower center", ncols=len(STRATS), frameon=False)
    save(fig, out, "fig1_effects_overview")


def fig_maps(res, ed, out):
    maps = {c: np.mean(res[c]["map"], axis=0) for c in CONDS}
    vmax = np.percentile(np.concatenate([m.ravel() for m in maps.values()]), 99.7)
    fig, axs = plt.subplots(1, len(CONDS), figsize=(3.5 * len(CONDS), 3.9), constrained_layout=True)
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#c6dcf5", "#6da7e8", "#2a78d6", "#134a8e"])
    for ax, c in zip(axs, CONDS):
        im = ax.pcolormesh(ed, ed, maps[c].T, cmap=cmap, vmin=0, vmax=vmax, rasterized=True)
        th = np.linspace(0, 2 * np.pi, 200)
        ax.plot(np.cos(th), np.sin(th), color=INK, lw=1.2)
        ax.plot(EDGE * np.cos(th), EDGE * np.sin(th), color=INK, lw=1, ls="--")
        ax.set_aspect("equal")
        ax.set_title(NAME[c])
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    axs[0].text(0, -1.03, f"虚线内 / 外 = 内圈 / 外圈（r = {EDGE:.2f} R）", ha="center", va="top", color=INK2, fontsize=9)
    cb = fig.colorbar(im, ax=axs, shrink=0.8, pad=0.01)
    cb.set_label("平均每格 size ≥ 6 的 group 数", color=INK2)
    cb.outline.set_visible(False)
    save(fig, out, "fig2_large_group_maps")


def profile(ax, res, key, den=None, scale=1.0, ylabel=""):
    rho = np.sqrt((np.arange(20) + 0.5) * 1.21 / 20)
    for c in CONDS:
        a = np.array(res[c][key], float)
        if den:
            a = a / np.array(res[c][den], float)
        a = a[:, :18] * scale
        m, s = np.nanmean(a, 0), np.nanstd(a, 0, ddof=1) / np.sqrt(len(a))
        ax.fill_between(rho[:18], m - s, m + s, color=COL[c], alpha=0.16, lw=0)
        ax.plot(rho[:18], m, color=COL[c], lw=2, marker=MARK[c], ms=4, label=NAME[c])
    ax.axvspan(EDGE, 1.02, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("r / R（等面积分格）")
    ax.set_xlim(0, 1.02)
    grid(ax)


def fig_radial(res, out):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    profile(axs[0], res, "tot", ylabel="机器人数（每等面积格）")
    axs[0].set_title("a  机器人密度")
    profile(axs[1], res, "large", den="tot", scale=100, ylabel="该位置机器人处于 ≥6 group 的比例（%）")
    axs[1].set_title("b  局部大 group 占比")
    profile(axs[2], res, "large", ylabel="处于 ≥6 group 的机器人数（每等面积格）")
    axs[2].set_title("c  大 group 机器人的径向分布")
    legend_all(axs[0], loc="lower left")
    for ax in axs:
        ax.text(EDGE + 0.012, ax.get_ylim()[1], "外圈", color=INK2, va="top", fontsize=9)
    save(fig, out, "fig3_radial_profiles")


def dots(ax, M, metric, window, title, ylabel, scale=1.0):
    t = M[M.window == window]
    pv = {c: t[t.cond == c].sort_values("trial")[metric].to_numpy(float) * scale for c in CONDS}
    for i in range(5):
        ax.plot(range(len(CONDS)), [pv[c][i] for c in CONDS], color=GRID, lw=1, zorder=1)
    for x, c in enumerate(CONDS):
        ax.plot([x] * 5, pv[c], MARK[c], color=COL[c], ms=7, mec=SURFACE, mew=1.2, zorder=3, alpha=0.9)
        ax.plot([x - 0.24, x + 0.24], [np.nanmean(pv[c])] * 2, color=INK, lw=2, zorder=4, solid_capstyle="round")
    ax.set_xticks(range(len(CONDS)), [NAME[c] for c in CONDS])
    ax.set_xlim(-0.45, len(CONDS) - 0.55)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    grid(ax)


def fig_form_stab(M, res, out):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    w = "t>=100"
    dots(axs[0, 0], M, "ignitions_outer_per_min", w, "a  外圈形成速率", "次 / min")
    dots(axs[0, 1], M, "ignitions_inner_per_min", w, "b  内圈形成速率", "次 / min")
    dots(axs[0, 2], M, "outer_share_of_ignitions", w, "c  形成事件落在外圈的占比", "占比（%）", 100)
    axs[0, 2].axhline(39.5, color=INK3, ls="--", lw=1)
    axs[0, 2].text(len(CONDS) - 0.58, 40.5, "外圈面积占比 39.5%", ha="right", va="bottom", fontsize=8.5, color=INK2)
    axs[0, 2].set_ylim(top=max(axs[0, 2].get_ylim()[1], 46))    # keep the reference line in view
    dots(axs[1, 0], M, "decay_per_groupmin_outer", w, "d  外圈衰减率", "次 / (group·min)")
    dots(axs[1, 1], M, "decay_per_groupmin_inner", w, "e  内圈衰减率", "次 / (group·min)")
    ax = axs[1, 2]
    for c in CONDS:
        km = pd.concat(res[c]["km"])
        o = km[km.outer]
        x, y = crs.km_curve(o.dur.to_numpy(), o.obs.to_numpy())
        ax.step(x, y, where="post", color=COL[c], lw=2, label=f"{NAME[c]} (n={len(o)})")
    ax.axhline(0.5, color=INK3, ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_xlim(1, 400)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("形成后存活时间（s）")
    ax.set_ylabel("存活比例（Kaplan–Meier）")
    ax.set_title("f  外圈形成的大 group 的寿命（5 trial 合并）")
    grid(ax)
    legend_all(ax, loc="lower left", fontsize=9)
    save(fig, out, "fig4_formation_stability")


def fig_h3(R, cc, M, out):
    fig, axs = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    ax = axs[0, 0]
    s = np.arange(1, 81)
    for c in CONDS:
        a = cc[c] * 100
        m = a.mean(0)
        se = a.std(0, ddof=1) / np.sqrt(5)
        ax.fill_between(s, np.maximum(m - se, 1e-3), m + se, color=COL[c], alpha=0.16, lw=0)
        ax.plot(s, m, color=COL[c], lw=2, label=NAME[c])
    ax.set_yscale("log")
    ax.set_xlim(3, 50)
    ax.set_ylim(0.05, 100)
    ax.set_xlabel("group 大小 s")
    ax.set_ylabel("处于 ≥ s group 的机器人比例（%）")
    ax.set_title("a  机器人加权的 group 大小分布")
    grid(ax)
    legend_all(ax, loc="upper right")
    w = "t>=100"
    dots(axs[0, 1], R, "share_ge6", w, "b  处于 ≥6 group 的机器人", "比例（%）", 100)
    dots(axs[0, 2], R, "share_ge20", w, "c  处于 ≥20 group 的机器人", "比例（%）", 100)
    dots(axs[1, 0], R, "largest_mean", w, "d  最大 group（时间平均）", "机器人数")
    dots(axs[1, 1], M, "large_km_median_s", w, "e  大 group 中位寿命", "s")
    t = M[M.window == w]
    ax = axs[1, 2]
    for k, (metric, lab) in enumerate([("splits_per_min", "分裂"), ("merges_per_min", "合并")]):
        for j, c in enumerate(CONDS):
            v = t[t.cond == c][metric].to_numpy()
            x = k + (j - (len(CONDS) - 1) / 2) * 0.72 / len(CONDS)
            ax.plot([x] * 5, v, MARK[c], color=COL[c], ms=7, mec=SURFACE, mew=1.2, alpha=0.9,
                    label=NAME[c] if k == 0 else None)
            ax.plot([x - 0.27 / len(CONDS), x + 0.27 / len(CONDS)], [v.mean()] * 2, color=INK, lw=2, solid_capstyle="round")
    ax.set_xticks([0, 1], ["分裂", "合并"])
    ax.set_ylabel("次 / min")
    ax.set_ylim(bottom=0)
    ax.set_title("f  分裂与合并速率")
    grid(ax)
    legend_all(ax, loc="lower center", ncols=len(CONDS), fontsize=8.5, columnspacing=0.8, handletextpad=0.2)
    save(fig, out, "fig5_h3_group_size")


def fig_mech(res, TS, out):
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    rho = np.sqrt((np.arange(20) + 0.5) * 1.21 / 20)
    ax = axs[0]
    for c in STRATS:
        a = 100 * np.array(res[c]["cat_ung"], float) / np.array(res[c]["ung"], float)
        m = np.nanmean(a, 0)[:18]
        ax.plot(rho[:18], m, color=COL[c], lw=2, marker=MARK[c], ms=4, label=NAME[c])
    ax.axvspan(EDGE, 1.02, color=GRID, alpha=0.45, lw=0, zorder=0)
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("r / R（等面积分格）")
    ax.set_ylabel("该位置 size ≤ 2 的机器人中 catalyst 占比（%）")
    ax.set_title("a  catalyst 的实际位置")
    grid(ax)
    ax = axs[1]
    t = (np.arange(len(res["baseline"]["ts_o"][0])) + 0.5) * 25
    for c in CONDS:
        o = np.array(res[c]["ts_o"]) * 100
        ax.plot(t, o.mean(0), color=COL[c], lw=2, label=NAME[c])
        ax.fill_between(t, o.mean(0) - o.std(0, ddof=1) / 5 ** .5, o.mean(0) + o.std(0, ddof=1) / 5 ** .5,
                        color=COL[c], alpha=0.16, lw=0)
    ax.axvline(T_EV, color=INK3, ls="--", lw=1)
    ax.set_xlabel("时间（s）")
    ax.set_ylabel("外圈 ≥6 group 中的机器人（占总数 %）")
    ax.set_title("b  外圈大 group 占比随时间")
    ax.set_ylim(bottom=0)
    grid(ax)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncols=len(CONDS), frameon=False)
    ax = axs[2]
    for c in CONDS:
        i = np.array(res[c]["ts_i"]) * 100
        ax.plot(t, i.mean(0), color=COL[c], lw=2, label=NAME[c])
        ax.fill_between(t, i.mean(0) - i.std(0, ddof=1) / 5 ** .5, i.mean(0) + i.std(0, ddof=1) / 5 ** .5,
                        color=COL[c], alpha=0.16, lw=0)
    ax.axvline(T_EV, color=INK3, ls="--", lw=1)
    ax.set_xlabel("时间（s）")
    ax.set_ylabel("内圈 ≥6 group 中的机器人（占总数 %）")
    ax.set_title("c  内圈大 group 占比随时间")
    ax.set_ylim(bottom=0)
    grid(ax)
    save(fig, out, "fig6_mechanism_and_time")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("baseline")
    ap.add_argument("strategies", nargs="+", metavar="NAME=DIR")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    global STRATS, CONDS
    strat = dict(x.split("=", 1) for x in a.strategies)
    STRATS[:] = list(strat)       # in place: other modules import these lists
    CONDS[:] = ["baseline"] + STRATS
    dirs = dict(baseline=a.baseline, **strat)
    results = a.results or a.out
    os.makedirs(results, exist_ok=True)
    exps = {c: os.path.basename(os.path.normpath(p)) for c, p in dirs.items()}
    M = pd.read_csv(os.path.join(results, "metrics_per_trial.csv"))
    C = pd.read_csv(os.path.join(results, "comparisons.csv"))
    R, cc, TS = raw_h3(dirs, results)
    res, ed = spatial_reduction(exps, a.out)

    # numbers quoted in the write-up
    rho = np.sqrt((np.arange(20) + 0.5) * 1.21 / 20)
    rows = []
    for c in STRATS:
        cat, ung, cu, tot = (np.array(res[c][k]) for k in ("cat", "ung", "cat_ung", "tot"))
        rows.append(dict(cond=c, catalysts=cat.sum(1).mean(), ungrouped=ung.sum(1).mean(),
                         catalyst_frac_of_ungrouped=(cu.sum(1) / ung.sum(1)).mean(),
                         cat_in_groups_ge3=(cat.sum(1) - cu.sum(1)).mean(),
                         catalysts_in_outer_frac=(cat[:, crs.OUTER_K:].sum(1) / cat.sum(1)).mean(),
                         catalyst_mean_rho=(cat * rho).sum(1).mean() / cat.sum(1).mean(),
                         ung_catfrac_outer=(cu[:, crs.OUTER_K:].sum(1) / ung[:, crs.OUTER_K:].sum(1)).mean(),
                         ung_catfrac_inner=(cu[:, :crs.OUTER_K].sum(1) / ung[:, :crs.OUTER_K].sum(1)).mean()))
    P = pd.DataFrame(rows)
    P.to_csv(os.path.join(results, "catalyst_role_profile.csv"), index=False)
    print(P.T.to_string())

    fig_forest(C, results)
    fig_maps(res, ed, results)
    fig_radial(res, results)
    fig_form_stab(M, res, results)
    fig_h3(R, cc, M, results)
    fig_mech(res, TS, results)

    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format)
    print(R.groupby(["window", "cond"]).mean(numeric_only=True).to_string())
    h3 = []
    for w in crs.WINDOWS:
        base = R[(R.window == w) & (R.cond == "baseline")].sort_values("trial")
        for s in STRATS:
            st = R[(R.window == w) & (R.cond == s)].sort_values("trial")
            for me in ("share_ge3", "share_ge6", "share_ge10", "share_ge20", "largest_mean", "largest_peak", "mean_size"):
                h3.append(dict(window=w, **crs.compare(base, st, s, me, predicted=1)))
    H = pd.DataFrame(h3)
    H.to_csv(os.path.join(results, "h3_raw_comparisons.csv"), index=False)
    print(H[["window", "strategy", "metric", "baseline", "strat", "diff", "ci_lo", "ci_hi", "pairs_in_dir", "signflip_p",
             "welch_p"]].to_string())
    print(R[(R.window == "t>=100")].pivot(index="trial", columns="cond", values="largest_mean").to_string())
    print(R[(R.window == "t>=100")].pivot(index="trial", columns="cond", values="share_ge20").to_string())


if __name__ == "__main__":
    main()

"""
strategy_hypotheses.py
----------------------
Test two hypotheses about a strategy against a baseline run from the same
seeds / initial conditions (trials paired by name):

  H1  the strategy makes large groups (size >= --large, default 6) easier to form
  H2  the strategy does not change the share of robots that are in a group at
      all (size >= --grouped, default 3)

Per-frame quantities come from the grouping cache written by
batch_group_plots.py (trial_XXXX_groups_d<max_dist>.csv, singletons included),
with the same robot-weighted definitions as the mean_composition plots:

    share_ge<k>      fraction of robots whose group has >= k members
    large_given_grp  share_ge<large> / share_ge<grouped>: of the grouped robots,
                     how many sit in a large group
    n_large          number of groups with >= <large> members in the frame
    mean_size        sum(s^2) / N, the group size seen by a random robot
    P(S >= s)        robot-weighted complementary CDF of group size

Each is time-averaged over a window, one value per trial: t >= --t0 (default
50 s), or with --last S the final S seconds of the run.

If group_lifetimes.py has been run on both experiments, its tracks
(<exp>_tracks_d<d>_m<grouped>_w<win>.csv) add
    large_form_per_min      tracks born in the window that start below <large> and
                            grow to it, per minute -- how often a large group forms
                            (fragments split off an already-large group excluded)
    frac_time_grouped_hyst  share of robot-time in hysteresis-smoothed groups
                            >= <grouped> within the window (each track's mean size
                            x its overlap with the window, so approximate)
    large_km_median_s       Kaplan-Meier median lifetime of groups >= 6, over the
                            whole run (group_lifetimes' own summary, not windowed)

Statistics, per metric, on the paired differences d = strategy - baseline:
mean, 95% t-interval, relative change, paired t-test, exact sign-flip
permutation test (all 2^n flips; with n=5 the smallest two-sided p is 0.0625,
one-sided 0.031), and Welch's t-test as an unpaired check.
For H2 "no change" is tested as equivalence (TOST): the 90% CI of d must lie
inside +-margin. Instead of picking a margin, the table reports the smallest
margin TOST would accept at alpha = 0.05, i.e. max(|90% CI bound|).

Usage:
    python strategy_hypotheses.py <baseline_exp> <strategy_exp> [--t0 50 | --last 100] [--large 6] [--grouped 3]
                                  [--labels baseline strategy] [--out-dir DIR]
"""

import argparse
import warnings
import glob
import itertools
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", "All-NaN slice|Mean of empty slice", RuntimeWarning)

INK, INK2, INK3, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834"]
S_MAX = 60


def trial_metrics(groups_csv, n, fps, t0, large, grouped, last=None):
    df = pd.read_csv(groups_csv, usecols=["Step", "size"])
    T = df.Step.max() + 1
    if last:
        t0 = T / fps - last
    keep = np.arange(T) / fps >= t0
    sz, st = df["size"].to_numpy(), df.Step.to_numpy()

    def per_frame(mask, w):
        return np.bincount(st[mask], weights=w[mask], minlength=T)

    ge_g = per_frame(sz >= grouped, sz.astype(float)) / n
    ge_l = per_frame(sz >= large, sz.astype(float)) / n
    n_large = per_frame(sz >= large, np.ones(len(sz)))
    mean_size = per_frame(np.ones(len(sz), bool), sz.astype(float) ** 2) / n
    # robot-weighted CCDF, averaged over kept frames
    k = keep[st]
    hist = np.bincount(np.minimum(sz[k], S_MAX), weights=sz[k], minlength=S_MAX + 1)[1:] / (n * keep.sum())
    ccdf = hist[::-1].cumsum()[::-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        cond = np.where(ge_g > 0, ge_l / ge_g, np.nan)
    m = {f"share_ge{grouped}": ge_g[keep].mean(), f"share_ge{large}": ge_l[keep].mean(),
         "large_given_grp": np.nanmean(cond[keep]), "n_large": n_large[keep].mean(),
         "mean_size": mean_size[keep].mean()}
    series = pd.DataFrame({"t": np.arange(T) / fps, "ge_g": ge_g, "ge_l": ge_l})
    return m, ccdf, series, T, t0


def load(exp_dir, t0, large, grouped, max_dist, win_s, last=None):
    cfg = json.load(open(os.path.join(exp_dir, "config_snapshot.json"), encoding="utf-8"))
    n, fps = cfg["N_SMARTICLES"], cfg.get("RENDER_FPS_HEADLESS", 60)
    name = os.path.basename(exp_dir)
    rows, ccdfs, series = [], [], []
    for csv in sorted(glob.glob(os.path.join(exp_dir, "trial_*", f"*_groups_d{max_dist}.csv"))):
        trial = os.path.basename(os.path.dirname(csv))
        m, c, s, T, t0 = trial_metrics(csv, n, fps, t0, large, grouped, last)
        rows.append(dict(trial=trial, **m))
        ccdfs.append(c)
        series.append(s.assign(trial=trial))
        print(f"  {name}/{trial} done")
    if not rows:
        raise SystemExit(f"{exp_dir}: no trial_*/*_groups_d{max_dist}.csv -- run batch_group_plots.py first")
    t = pd.DataFrame(rows)

    tracks_csv = os.path.join(exp_dir, f"{name}_tracks_d{max_dist}_m{grouped}_w{win_s:g}.csv")
    if os.path.exists(tracks_csv):
        tr = pd.read_csv(tracks_csv, usecols=["trial", "t_start", "t_end", "size_start", "max_size", "mean_size"])
        minutes = (T / fps - t0) / 60
        # a formation is a track that starts below `large` and grows to it; fragments that split
        # off an already-large group start at >= large and are fragmentation, not formation
        born = tr[(tr.t_start / fps >= t0) & (tr.size_start < large) & (tr.max_size >= large)]
        t["large_form_per_min"] = t.trial.map(born.groupby("trial").size()).fillna(0) / minutes
        life = pd.read_csv(tracks_csv.replace("_tracks_", "_lifetime_trials_")).set_index("trial")
        overlap = (tr.t_end + 1).clip(upper=T) - tr.t_start.clip(lower=int(np.ceil(t0 * fps)))
        robot_frames = (tr.mean_size * overlap.clip(lower=0)).groupby(tr.trial).sum()
        t["frac_time_grouped_hyst"] = t.trial.map(robot_frames) / (n * (T - np.ceil(t0 * fps)))
        if large == 6:      # group_lifetimes.LARGE
            t["large_km_median_s"] = t.trial.map(life.large_km_median_s)
    else:
        print(f"  ({tracks_csv} missing -- run group_lifetimes.py for the formation rate)")
    return name, n, t, np.array(ccdfs), pd.concat(series), t0


def signflip_p(d, alternative="two-sided"):
    flips = np.array(list(itertools.product([1, -1], repeat=len(d))))
    null = (flips * d).mean(axis=1)
    obs = d.mean()
    if alternative == "greater":
        return np.mean(null >= obs - 1e-12)
    return np.mean(np.abs(null) >= abs(obs) - 1e-12)


def compare(ref, alt, metric):
    x, y = alt[metric].to_numpy(), ref[metric].to_numpy()
    d = x - y
    n = len(d)
    se = d.std(ddof=1) / np.sqrt(n)
    q95, q90 = stats.t.ppf(0.975, n - 1), stats.t.ppf(0.95, n - 1)
    return dict(metric=metric, baseline_mean=y.mean(), baseline_sd=y.std(ddof=1),
                strategy_mean=x.mean(), strategy_sd=x.std(ddof=1),
                diff=d.mean(), ci95_lo=d.mean() - q95 * se, ci95_hi=d.mean() + q95 * se,
                **{k: v / y.mean() if y.mean() else np.nan for k, v in
                   (('rel_change', d.mean()), ('rel_ci95_lo', d.mean() - q95 * se), ('rel_ci95_hi', d.mean() + q95 * se))},
                n_pairs_up=int((d > 0).sum()), cohen_dz=d.mean() / d.std(ddof=1),
                paired_t_p=stats.ttest_rel(x, y).pvalue,
                paired_t_p_greater=stats.ttest_rel(x, y, alternative="greater").pvalue,
                signflip_p=signflip_p(d), signflip_p_greater=signflip_p(d, "greater"),
                welch_p=stats.ttest_ind(x, y, equal_var=False).pvalue,
                tost_min_margin=max(abs(d.mean() - q90 * se), abs(d.mean() + q90 * se)))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("baseline")
    ap.add_argument("strategy")
    ap.add_argument("--t0", type=float, default=50.0)
    ap.add_argument("--last", type=float, default=None, help="use the final LAST seconds instead of t >= t0")
    ap.add_argument("--large", type=int, default=6)
    ap.add_argument("--grouped", type=int, default=3)
    ap.add_argument("--max-dist", default="65")
    ap.add_argument("--win-s", type=float, default=1.0, help="hysteresis tag of the group_lifetimes tracks to read")
    ap.add_argument("--labels", nargs=2, default=["baseline", "strategy"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    L, G = args.large, args.grouped

    ref_name, n, ref, ref_cc, ref_ts, t0a = load(os.path.abspath(args.baseline), args.t0, L, G, args.max_dist,
                                                 args.win_s, args.last)
    alt_name, n2, alt, alt_cc, alt_ts, t0b = load(os.path.abspath(args.strategy), args.t0, L, G, args.max_dist,
                                                  args.win_s, args.last)
    if abs(t0a - t0b) > 1:
        raise SystemExit(f"runs differ in length (window starts {t0a:.1f} s vs {t0b:.1f} s)")
    args.t0 = t0a
    args.window = f"last {args.last:g} s" if args.last else f"t ≥ {t0a:g} s"
    if n != n2 or list(ref.trial) != list(alt.trial):
        raise SystemExit("experiments must have the same N and the same trial names (seed-paired)")

    both = [m for m in ("large_form_per_min", "frac_time_grouped_hyst", "large_km_median_s")
            if m in ref and m in alt]
    for m in [m for m in both if ref[m].isna().any() or alt[m].isna().any()]:
        # e.g. large_km_median_s when over half the large groups outlive the run
        print(f"  {m}: undefined in some trial, left out\n"
              f"    {ref_name}: {ref[m].round(1).tolist()}\n    {alt_name}: {alt[m].round(1).tolist()}")
        both.remove(m)
    h1 = [f"share_ge{L}", "large_given_grp", "n_large"] + [m for m in both if m == "large_form_per_min"]
    h2 = [f"share_ge{G}"] + [m for m in both if m == "frac_time_grouped_hyst"]
    ctx = ["mean_size"] + [m for m in both if m == "large_km_median_s"]
    table = pd.DataFrame([dict(hypothesis="H1", **compare(ref, alt, m)) for m in h1] +
                         [dict(hypothesis="H2", **compare(ref, alt, m)) for m in h2] +
                         [dict(hypothesis="context", **compare(ref, alt, m)) for m in ctx])

    out_dir = args.out_dir or os.path.commonpath([os.path.abspath(args.baseline), os.path.abspath(args.strategy)])
    out = os.path.join(out_dir, f"hypotheses_{ref_name}_vs_{alt_name}_" +
                       (f"last{args.last:g}" if args.last else f"t{args.t0:g}"))
    table.to_csv(out + ".csv", index=False)
    pd.concat([ref.assign(exp=ref_name), alt.assign(exp=alt_name)]).to_csv(out + "_trials.csv", index=False)
    pd.set_option("display.width", 250)
    print(table.round(4).T.to_string())

    plot(args, ref, alt, ref_cc, alt_cc, ref_ts, alt_ts, table, out, ref_name, alt_name)


# =============================================================================
# Figures
# =============================================================================

def plot(args, ref, alt, ref_cc, alt_cc, ref_ts, alt_ts, table, out, ref_name, alt_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    L, G = args.large, args.grouped
    lab = args.labels
    row = table.set_index("metric")

    def style(ax, grid="y"):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis=grid, color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def pfmt(p):
        return f"{p:.3f}" if p >= 0.001 else "<0.001"

    fig = plt.figure(figsize=(15, 10), facecolor=SURFACE, layout="constrained")
    top, bot = fig.subfigures(2, 1, height_ratios=[1.05, 1])
    for sf in (top, bot):
        sf.set_facecolor(SURFACE)
    axA, axB = top.subplots(1, 2, gridspec_kw={"width_ratios": [1.15, 1]})

    # ---- A: robot-weighted CCDF --------------------------------------------
    style(axA, "both")
    s = np.arange(1, 41)
    for cc, c, name in ((ref_cc, SERIES[0], lab[0]), (alt_cc, SERIES[1], lab[1])):
        cc = np.where(cc > 0, cc, np.nan)[:, :len(s)]
        axA.fill_between(s, np.nanmin(cc, 0), np.nanmax(cc, 0), color=c, alpha=0.18, lw=0)
        axA.plot(s, np.nanmean(cc, 0), color=c, lw=2, label=name)
    axA.set_yscale("log")
    axA.set_xlim(1, s[-1])
    axA.set_ylim(1e-3, 1.6)
    for k, h in ((G, "H2"), (L, "H1")):
        a, b = ref_cc[:, k - 1].mean(), alt_cc[:, k - 1].mean()
        axA.axvline(k, color=INK3, lw=1, ls="--", zorder=1)
        axA.scatter([k, k], [a, b], s=40, color=SERIES, edgecolor=SURFACE, lw=1.5, zorder=4)
        axA.annotate(f"{h}  s ≥ {k}:  {a*100:.1f}% → {b*100:.1f}%", (k, min(a, b)),
                     xytext=(40, -26 if h == "H2" else -34), textcoords="offset points", fontsize=9, color=INK,
                     arrowprops=dict(arrowstyle="-", color=INK3, lw=0.8))
    axA.set_xlabel("group size s")
    axA.set_ylabel("P(robot's group size ≥ s)   (log)")
    axA.set_title(f"A   Robot-weighted group-size distribution, {args.window} (t ≥ {args.t0:.0f} s)", loc="left", fontsize=11)
    axA.legend(frameon=False, loc="upper right", fontsize=9, title="mean; band = range of 5 trials",
               title_fontsize=8.5)

    # ---- B: forest plot of relative change ---------------------------------
    style(axB, "x")
    names = {f"share_ge{G}": f"share of robots in groups ≥{G}", f"share_ge{L}": f"share of robots in groups ≥{L}",
             "large_given_grp": f"share in ≥{L} among grouped (≥{G})", "n_large": f"# groups ≥{L} per frame",
             "large_form_per_min": f"groups growing to ≥{L} / min", "mean_size": "mean group size (robot view)",
             "frac_time_grouped_hyst": f"robot-time in groups ≥{G} (1 s hyst.)",
             "large_km_median_s": f"median lifetime of groups ≥{L} (whole run)"}
    order = [m for m in [f"share_ge{G}", "frac_time_grouped_hyst", f"share_ge{L}", "large_given_grp", "n_large",
                         "large_form_per_min", "mean_size", "large_km_median_s"] if m in row.index]
    y = np.arange(len(order))[::-1]
    for yi, m in zip(y, order):
        r = row.loc[m]
        col = INK3 if r.hypothesis == "context" else (SERIES[1] if r.hypothesis == "H1" else INK)
        if r.baseline_mean == 0:
            axB.text(0, yi, f"  baseline = 0 → {r.strategy_mean:.2g} (no relative change)", va="center",
                     fontsize=9, color=col)
            axB.text(1.02, yi, f"p={pfmt(r.paired_t_p)}  {r.n_pairs_up}/{len(ref)}↑",
                     transform=axB.get_yaxis_transform(), va="center", fontsize=9, color=INK2)
            continue
        axB.plot([r.rel_ci95_lo * 100, r.rel_ci95_hi * 100], [yi, yi], color=col, lw=2, solid_capstyle="round")
        axB.plot(r.rel_change * 100, yi, "o", ms=8, color=col, mec=SURFACE, mew=2)
        axB.text(1.02, yi, f"{r.rel_change*100:+.0f}%  p={pfmt(r.paired_t_p)}  {r.n_pairs_up}/{len(ref)}↑",
                 transform=axB.get_yaxis_transform(), va="center", fontsize=9, color=INK2)
    axB.axvline(0, color=INK2, lw=1)
    axB.set_yticks(y, [f"{row.loc[m].hypothesis}   {names[m]}" for m in order])
    axB.set_ylim(y.min() - 0.6, y.max() + 0.6)
    axB.set_xlabel(f"{lab[1]} vs {lab[0]}: relative change (%), 95% CI")
    axB.set_title("B   Paired effects  (p = paired t; k/n↑ = pairs that rose)", loc="left", fontsize=11)

    # ---- C: where the change comes from: Δ share per size bin ---------------
    axes = bot.subplots(1, 4, gridspec_kw={"width_ratios": [1.15, 1, 1, 1]})
    style(axes[0])
    parts = {f"1-{G - 1}": lambda t: 1 - t[f"share_ge{G}"], f"{G}-{L - 1}": lambda t: t[f"share_ge{G}"] - t[f"share_ge{L}"],
             f"{L}+": lambda t: t[f"share_ge{L}"]}
    n_tr = len(ref)
    q = stats.t.ppf(0.975, n_tr - 1)
    for i, (bl, f) in enumerate(parts.items()):
        d = (f(alt) - f(ref)).to_numpy() * 100
        ci = q * d.std(ddof=1) / np.sqrt(n_tr)
        axes[0].bar(i, d.mean(), width=0.6, color=SERIES[1] if d.mean() > 0 else SERIES[0], alpha=0.85)
        axes[0].errorbar(i, d.mean(), yerr=ci, color=INK, capsize=4, lw=1.2)
        axes[0].scatter(np.full(n_tr, i) + np.linspace(-0.12, 0.12, n_tr), d, s=12, color=INK, zorder=3)
        axes[0].text(i + 0.32, d.mean(), f"{d.mean():+.1f}", ha="left", va="center", fontsize=9)
    axes[0].axhline(0, color=INK2, lw=1)
    axes[0].set_xticks(range(len(parts)), [f"size {k}" for k in parts])
    axes[0].set_ylabel("Δ share of robots (pp)")
    axes[0].set_title("C   Δ composition (strategy − baseline)", loc="left", fontsize=11, pad=48)
    axes[0].text(0.5, 1.0, "bar = mean, whisker = 95% CI, dots = 5 pairs\nthe three bars sum to 0: robots gained by\n"
                 "one size class were lost by the others", transform=axes[0].transAxes,
                 ha="center", va="bottom", fontsize=8.5, color=INK2)

    # ---- D-F: per-trial paired values --------------------------------------
    panels = [(f"share_ge{G}", 100, f"D   H2: % robots in groups ≥{G}"),
              (f"share_ge{L}", 100, f"E   H1: % robots in groups ≥{L}")]
    if "large_form_per_min" in row.index:
        panels.append(("large_form_per_min", 1, f"F   H1: groups reaching ≥{L} / min"))
    else:
        panels.append(("n_large", 1, f"F   H1: # groups ≥{L} per frame"))
    for ax, (m, scale, title) in zip(axes[1:], panels):
        style(ax)
        a, b = ref[m].to_numpy() * scale, alt[m].to_numpy() * scale
        for u, v in zip(a, b):
            ax.plot([0, 1], [u, v], color=GRID, lw=1.5, zorder=1)
        ax.scatter(np.zeros_like(a), a, s=55, color=SERIES[0], edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.scatter(np.ones_like(b), b, s=55, color=SERIES[1], edgecolor=SURFACE, lw=1.5, zorder=3)
        for xi, v in ((0, a), (1, b)):
            ax.plot([xi - 0.12, xi + 0.12], [v.mean()] * 2, color=INK, lw=2)
        r = row.loc[m]
        unit = " pp" if scale == 100 else ""
        txt = (f"Δ = {r['diff']*scale:+.2f}{unit}  [95% CI {r.ci95_lo*scale:+.2f}, {r.ci95_hi*scale:+.2f}]\n"
               f"{r.n_pairs_up}/{len(a)} pairs up · paired t p = {pfmt(r.paired_t_p)}")
        if r.hypothesis == "H2":
            txt += f"\n“no change” holds (TOST) only if margin ≥ {r.tost_min_margin*scale:.1f} pp"
        ax.text(0.5, 1.0, txt, transform=ax.transAxes, ha="center", va="bottom", fontsize=8.5, color=INK2)
        ax.set_xticks([0, 1], lab)
        ax.set_xlim(-0.4, 1.4)
        ax.set_title(title, loc="left", fontsize=11, pad=48)
    bot.supxlabel("D–F: one dot per trial, lines join trials with the same initial condition, black tick = mean",
                  fontsize=9, color=INK2)

    fig.suptitle(f"{lab[0]} ({ref_name})  vs  {lab[1]} ({alt_name}),  N = {len(ref)} paired trials,  window: {args.window}",
                 fontsize=12)
    fig.savefig(out + ".png", dpi=150, facecolor=SURFACE)
    fig.savefig(out + ".svg", facecolor=SURFACE)
    plt.close(fig)

    # ---- time series of the two shares -------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), facecolor=SURFACE, layout="constrained", sharex=True)
    for ax, col, title in ((axes[0], "ge_g", f"share of robots in groups ≥{G} (H2)"),
                           (axes[1], "ge_l", f"share of robots in groups ≥{L} (H1)")):
        style(ax)
        for ts, c, name in ((ref_ts, SERIES[0], lab[0]), (alt_ts, SERIES[1], lab[1])):
            w = ts.pivot(index="t", columns="trial", values=col)
            w = w.rolling(300, center=True, min_periods=1).mean() * 100   # 5 s at 60 fps
            mu, sd = w.mean(axis=1), w.std(axis=1)
            ax.fill_between(w.index, mu - sd, mu + sd, color=c, alpha=0.18, lw=0)
            ax.plot(w.index, mu, color=c, lw=2, label=f"{name}")
        ax.axvspan(0, args.t0, color=GRID, alpha=0.6, lw=0)
        ax.text(args.t0 / 2, 0.02, "excluded", transform=ax.get_xaxis_transform(), ha="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("%")
        ax.set_xlim(0, ref_ts.t.max())
    axes[0].legend(frameon=False, loc="lower right")
    fig.suptitle(f"mean ± SD across {len(ref)} trials, 5 s moving average; shaded = outside the analysis window", fontsize=9, color=INK2)
    fig.savefig(out + "_timeseries.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nwrote {out}.csv, _trials.csv, .png/.svg, _timeseries.png")


if __name__ == "__main__":
    main()

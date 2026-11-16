"""
catalyst_radial_stats.py
------------------------
Where in the arena do large groups form and persist, compared with a baseline?
Reads the per-trial reductions of catalyst_radial_build.py and writes, into <out>:

    metrics_per_trial.csv     one row per experiment x trial x window x metric-set
    comparisons.csv           strategy - baseline, per metric
    sensitivity_outer.csv     outer-ring metrics at other ring boundaries / centres, per window
    exploratory_bands.csv     post-hoc: large-group share in bands picked after seeing the profiles

Definitions (fixed before looking at any comparison)
    rho            distance to the reference centre / INNER_R. Centre = the frame's swarm
                   centroid, the point the selectors rank by; fixed arena centre as a check.
    outer ring     rho^2 >= 0.605, i.e. rho >= 0.778: the outer 295 px = 2.8 body lengths,
                   39.5% of the arena's area.
    large group    size >= 6 (peak size >= 6 for a track), on the hysteresis-smoothed groups.
    ignition       the first frame a track reaches size >= 6, if it began below 6 (a track
                   that splits off at >= 6 is a fragment, not a new formation).
    decay          a large group dropping below 6 or vanishing. Merging into another group
                   is not decay: the robots are still in a group that is at least as large.
    decay hazard   decays per large-group-minute, by the region the group is in.

Statistics: one value per trial, n = 5 pairs sharing initial conditions. Paired mean
difference with a 95% t-interval, the exact one-sided sign-flip p (floor 1/32 = 0.031),
and how many of the 5 pairs moved in the hypothesised direction.

Usage:
    python catalyst_radial_stats.py <out> [--baseline a862] [--strategy NAME=EXP ...] [--results DIR]

<out> holds cache/; csvs go to --results (default <out>). Default strategies are the
first pair, farthest=a862cis_farthest36 and nearest=a862cis_nearest36.
"""
import argparse
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

OUTER_K = 10                  # outer ring = equal-area bins k.. of 20 over rho^2 in [0, 1.21]
LARGE = 6
WINDOWS = {"t>=400": 400.0, "t>=100": 100.0}


def rho_edge(k):
    return np.sqrt(1.21 * k / 20)


def load_trial(path, center="swarm"):
    z = np.load(path, allow_pickle=True)      # tracks/events are record arrays with text columns
    d = {k: z[k] for k in z.files}
    fps, R = float(d["fps"]), float(d["R"])
    t = d["frames_t"]
    c = d["swarm_c"][t] if center == "swarm" else d["arena_c"][None]
    rho = np.hypot(*(d["frames_xy"] - c).T) / R
    f = pd.DataFrame({"t": t, "id": d["frames_id"], "size": d["frames_size"], "rho": rho})
    return d, f, fps


def large_track_table(f, fps):
    """One row per track that was ever >= LARGE, with what happened at and after it first was."""
    big = f[f["size"] >= LARGE]
    ids = big["id"].unique()
    g = f[f["id"].isin(ids)].sort_values(["id", "t"], kind="mergesort")
    rows = []
    for tid, s in g.groupby("id", sort=False):
        sz, t, rho = s["size"].to_numpy(), s["t"].to_numpy(), s["rho"].to_numpy()
        i = int(np.argmax(sz >= LARGE))
        L = sz >= LARGE
        rows.append(dict(id=tid, t_ign=t[i], rho_ign=rho[i], formed=sz[0] < LARGE, t_end=t[-1], t_start=t[0],
                         rho_mean_large=rho[L].mean(), peak=sz.max(), frames_large=int(L.sum()),
                         frac_large_outer=(rho[L] >= rho_edge(OUTER_K)).mean()))
    return pd.DataFrame(rows)


def km_median(dur, obs):
    """Kaplan-Meier median; nan if survival never reaches 1/2."""
    if len(dur) == 0:
        return np.nan
    o = np.argsort(dur)
    dur, obs = np.asarray(dur)[o], np.asarray(obs)[o]
    at_risk, s = len(dur), 1.0
    for d, e in zip(dur, obs):
        if e:
            s *= 1 - 1 / at_risk
            if s <= 0.5:
                return float(d)
        at_risk -= 1
    return np.nan


def km_curve(dur, obs):
    o = np.argsort(dur)
    dur, obs = np.asarray(dur, float)[o], np.asarray(obs)[o]
    at_risk, s, xs, ys = len(dur), 1.0, [0.0], [1.0]
    for d, e in zip(dur, obs):
        if e:
            s *= 1 - 1 / at_risk
            xs.append(d)
            ys.append(s)
        at_risk -= 1
    return np.array(xs), np.array(ys)


def decay_events(f, tracks_rec):
    """Frames at which a track goes from >= LARGE to < LARGE, or ends while >= LARGE by vanishing.

    Returns (t, rho) of each decay. A track that ends by merging is not counted.
    """
    death = {int(r["track_id"]): str(r["death"]) for r in tracks_rec}
    g = f.sort_values(["id", "t"], kind="mergesort")
    idn, sz, t, rho = g["id"].to_numpy(), g["size"].to_numpy(), g["t"].to_numpy(), g["rho"].to_numpy()
    same = idn[1:] == idn[:-1]
    drop = same & (sz[:-1] >= LARGE) & (sz[1:] < LARGE)
    ts, rs = [t[:-1][drop]], [rho[:-1][drop]]
    last = np.append(~same, True) & (sz >= LARGE)
    van = np.array([death.get(int(i)) == "vanish" for i in idn[last]], bool)
    ts.append(t[last][van])
    rs.append(rho[last][van])
    return np.concatenate(ts), np.concatenate(rs)


def spatial_metrics(d, f, fps, t0, k=OUTER_K, hist_key="hist_swarm"):
    T = int(d["T"])
    N = int(d["N"])
    a = int(np.ceil(t0 * fps))
    edge = rho_edge(k)
    minutes = (T - a) / fps / 60
    frames = T - a

    h = d[hist_key][a:].sum(axis=0).astype(np.int64)      # bins, size class, role
    tot = h.sum(axis=(1, 2)).astype(float)
    large = h[:, 2:].sum(axis=(1, 2)).astype(float)
    grouped = h[:, 1:].sum(axis=(1, 2)).astype(float)
    m = {}
    m["robots_outer_frac"] = tot[k:].sum() / tot.sum()
    m["large_robots_outer_of_N"] = large[k:].sum() / tot.sum()
    m["large_robots_inner_of_N"] = large[:k].sum() / tot.sum()
    m["large_frac_in_outer"] = large[k:].sum() / tot[k:].sum()
    m["large_frac_in_inner"] = large[:k].sum() / tot[:k].sum()
    m["outer_share_of_large_robots"] = large[k:].sum() / large.sum()
    m["grouped_frac_in_outer"] = grouped[k:].sum() / tot[k:].sum()

    w = f[f["t"] >= a]
    wl = w[w["size"] >= LARGE]
    m["n_large_outer"] = (wl["rho"] >= edge).sum() / frames
    m["n_large_inner"] = (wl["rho"] < edge).sum() / frames

    lt = large_track_table(f, fps)
    ign = lt[lt["formed"] & (lt["t_ign"] >= a)]
    out, inn = ign[ign["rho_ign"] >= edge], ign[ign["rho_ign"] < edge]
    m["ignitions_outer_per_min"] = len(out) / minutes
    m["ignitions_inner_per_min"] = len(inn) / minutes
    m["outer_share_of_ignitions"] = len(out) / len(ign) if len(ign) else np.nan

    def life(s):
        dur = (s["t_end"] - s["t_ign"] + 1) / fps
        return km_median(dur.to_numpy(), (s["t_end"] < T - 1).to_numpy())

    m["life_after_ign_outer_s"] = life(out)
    m["life_after_ign_inner_s"] = life(inn)
    m["frac_large_time_outer_after_outer_ign"] = (
        np.average(out["frac_large_outer"], weights=out["frames_large"]) if len(out) else np.nan)

    dt, drho = decay_events(f, d["tracks"])
    keep = dt >= a
    dt, drho = dt[keep], drho[keep]
    gm_out = (wl["rho"] >= edge).sum() / fps / 60
    gm_inn = (wl["rho"] < edge).sum() / fps / 60
    m["decay_per_groupmin_outer"] = (drho >= edge).sum() / gm_out if gm_out else np.nan
    m["decay_per_groupmin_inner"] = (drho < edge).sum() / gm_inn if gm_inn else np.nan
    return m, lt


def global_metrics(d, f, fps, t0):
    T, N = int(d["T"]), int(d["N"])
    a = int(np.ceil(t0 * fps))
    frames, minutes = T - a, (T - a) / fps / 60
    w = f[f["t"] >= a]
    s = w["size"].to_numpy()
    m = {}
    for k in (6, 10, 20):
        m[f"share_ge{k}"] = s[s >= k].sum() / (N * frames)
    m["share_ge3"] = s.sum() / (N * frames)
    largest = w.groupby("t")["size"].max()
    m["largest_mean"] = largest.mean()
    m["largest_peak"] = largest.max()
    m["n_large"] = (s >= LARGE).sum() / frames

    tr = pd.DataFrame.from_records(d["tracks"])
    tr["life_s"] = (tr["t_end"] - tr["t_start"] + 1) / fps
    big = tr[(tr["max_size"] >= LARGE) & ~tr["censored_left"]]
    m["large_km_median_s"] = km_median(big["life_s"].to_numpy(), (~big["censored_right"]).to_numpy())
    ev = pd.DataFrame.from_records(d["events"])
    ew = ev[ev["step"] >= a]
    m["splits_per_min"] = (ew["event"] == "split").sum() / minutes
    m["merges_per_min"] = (ew["event"] == "merge").sum() / minutes
    m["n_catalyst_mean"] = d["n_catalyst"][a:].mean()
    return m


def band_metrics(d, fps, t0=100.0):
    """Post-hoc bands read off the radial profiles; not part of the hypothesis tests."""
    T, N = int(d["T"]), int(d["N"])
    a = int(np.ceil(t0 * fps))
    h = d["hist_swarm"]
    tot = h[a:].sum(axis=(0, 2, 3)).astype(float)
    large = h[a:, :, 2:].sum(axis=(0, 2, 3)).astype(float)
    m = {}
    # k -> rho edges sqrt(1.21 k / 20): 0-4 is r < 0.55, 6-9 is 0.60-0.78, 10-11 is 0.78-0.85, 12-14 is 0.85-0.95
    for name, lo, hi in (("core_r<0.55", 0, 5), ("mid_0.60-0.78", 6, 10), ("gap_0.78-0.85", 10, 12), ("wall_0.85-0.95", 12, 15)):
        m[f"large_frac {name}"] = large[lo:hi].sum() / tot[lo:hi].sum()
        m[f"large_of_N {name}"] = large[lo:hi].sum() / (N * (T - a))
        m[f"robots_of_N {name}"] = tot[lo:hi].sum() / (N * (T - a))
    # the early transient, where the time series separate most
    b, e = int(50 * fps), int(200 * fps)
    for name, lo, hi in (("outer", OUTER_K, 20), ("inner", 0, OUTER_K)):
        m[f"large_of_N {name} 50-200s"] = h[b:e, lo:hi, 2:].sum() / (N * (e - b))
    return m


def signflip_p(x, greater=True):
    flips = np.array(list(itertools.product([1, -1], repeat=len(x))))
    null = (flips * x).mean(axis=1)
    return np.mean(null >= x.mean() - 1e-12) if greater else np.mean(null <= x.mean() + 1e-12)


def compare(base, strat, label, metric, predicted=None):
    """predicted: +1 / -1 for a directional hypothesis, None to report the direction seen."""
    a, b = strat[metric].to_numpy(float), base[metric].to_numpy(float)
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    d = a - b
    n = len(d)
    if n < 2:
        return dict(strategy=label, metric=metric, n=n)
    se = d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    sign = predicted if predicted else (1 if d.mean() >= 0 else -1)
    return dict(strategy=label, metric=metric, n=n, baseline=b.mean(), baseline_sd=b.std(ddof=1), strat=a.mean(),
                strat_sd=a.std(ddof=1), diff=d.mean(), ci_lo=d.mean() - q * se, ci_hi=d.mean() + q * se,
                rel=d.mean() / b.mean() if b.mean() else np.nan, direction=sign, pairs_in_dir=int((sign * d > 0).sum()),
                signflip_p=signflip_p(sign * d), welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--baseline", default="a862")
    ap.add_argument("--strategy", action="append", default=None, metavar="NAME=EXP")
    ap.add_argument("--results", default=None)
    a = ap.parse_args()
    strategies = dict(x.split("=", 1) for x in (a.strategy or ["farthest=a862cis_farthest36", "nearest=a862cis_nearest36"]))
    exps = {"baseline": a.baseline, **strategies}
    results = a.results or a.out
    os.makedirs(results, exist_ok=True)

    rows, sens, bands = [], [], []
    for label, exp in exps.items():
        for tn in range(5):
            trial = f"trial_{tn:04d}"
            path = os.path.join(a.out, "cache", f"{exp}__{trial}.npz")
            d, f, fps = load_trial(path)
            for wname, t0 in WINDOWS.items():
                sm, _ = spatial_metrics(d, f, fps, t0)
                rows.append(dict(cond=label, trial=trial, window=wname, **sm, **global_metrics(d, f, fps, t0)))
            for wname, t0 in WINDOWS.items():
                for k in (8, 10, 12):
                    sm, _ = spatial_metrics(d, f, fps, t0, k=k)
                    sens.append(dict(cond=label, trial=trial, window=wname, center="swarm", outer_rho=rho_edge(k), **sm))
            d2, f2, _ = load_trial(path, center="arena")
            for wname, t0 in WINDOWS.items():
                sm, _ = spatial_metrics(d2, f2, fps, t0, hist_key="hist_arena")
                sens.append(dict(cond=label, trial=trial, window=wname, center="arena", outer_rho=rho_edge(OUTER_K), **sm))
            bands.append(dict(cond=label, trial=trial, **band_metrics(d, fps)))
            print(label, trial, "done", flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(results, "metrics_per_trial.csv"), index=False)
    S = pd.DataFrame(sens)
    S.to_csv(os.path.join(results, "sensitivity_per_trial.csv"), index=False)
    keys = ["ignitions_outer_per_min", "n_large_outer", "large_frac_in_outer", "outer_share_of_large_robots",
            "decay_per_groupmin_outer"]
    out_rows = []
    for (win, cen, edge), g in S.groupby(["window", "center", "outer_rho"]):
        base = g[g.cond == "baseline"].sort_values("trial")
        for label in strategies:
            st = g[g.cond == label].sort_values("trial")
            for me in keys:
                out_rows.append(dict(window=win, center=cen, outer_rho=edge, **compare(base, st, label, me)))
    pd.DataFrame(out_rows).to_csv(os.path.join(results, "sensitivity_outer.csv"), index=False)
    B = pd.DataFrame(bands)
    B.to_csv(os.path.join(results, "exploratory_bands_per_trial.csv"), index=False)
    base = B[B.cond == "baseline"].sort_values("trial")
    ex = [compare(base, B[B.cond == l].sort_values("trial"), l, me) for l in strategies
          for me in B.columns if me not in ("cond", "trial")]
    pd.DataFrame(ex).to_csv(os.path.join(results, "exploratory_bands.csv"), index=False)

    metrics = [c for c in M.columns if c not in ("cond", "trial", "window")]
    cmp_rows = []
    for wname in WINDOWS:
        base = M[(M.cond == "baseline") & (M.window == wname)].sort_values("trial")
        for label in strategies:
            st = M[(M.cond == label) & (M.window == wname)].sort_values("trial")
            for me in metrics:
                cmp_rows.append(dict(window=wname, **compare(base, st, label, me)))
    C = pd.DataFrame(cmp_rows)
    C.to_csv(os.path.join(results, "comparisons.csv"), index=False)

    pd.set_option("display.width", 220, "display.max_rows", 200, "display.float_format", "{:.3f}".format)
    print(C[["window", "strategy", "metric", "baseline", "baseline_sd", "strat", "strat_sd", "diff", "ci_lo", "ci_hi",
             "pairs_in_dir", "signflip_p"]].to_string())


if __name__ == "__main__":
    main()

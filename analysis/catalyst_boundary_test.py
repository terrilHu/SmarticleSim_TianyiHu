"""
catalyst_boundary_test.py
-------------------------
Inner vs outer large-group metrics at any ring boundary, each strategy against baseline,
from catalyst_radial_build.py's cache. Redraws the map and radial-profile figures with
that boundary. Nothing is written outside <results>.

A group is placed by its centroid, so the boundary need not sit on a histogram bin edge.
That makes large_robots_* group-size-weighted by centroid, a little different from
catalyst_radial_stats.py, which counts each robot where it stands. Those robot-position
metrics exist only at bin edges, so they are reported at the two edges around the
boundary (robot_large_* with edge_rho).

Tests use one value per trial: paired t (two-sided), exact sign-flip (two-sided; its floor
is 2/32 = 0.0625 at n = 5) and Welch. Holm adjusts the paired t p within CLAIMS, per window.

Usage:
    python catalyst_boundary_test.py <out> <results> --edge 0.75 [--bare] NAME=EXP [NAME=EXP ...]

The first NAME=EXP is the baseline; NAME picks the condition's colour and marker.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

import catalyst_radial_figs as crf
import catalyst_radial_stats as crs

METRICS = ["large_robots_inner_of_N", "large_robots_outer_of_N", "n_large_inner", "n_large_outer",
           "ignitions_inner_per_min", "ignitions_outer_per_min", "decay_per_groupmin_inner",
           "decay_per_groupmin_outer", "outer_share_of_large_robots", "outer_share_of_ignitions"]
# what "farthest promotes inner, nearest promotes outer" means: occupancy and formation
CLAIMS = [("farthest", "large_robots_inner_of_N"), ("farthest", "ignitions_inner_per_min"),
          ("nearest", "large_robots_outer_of_N"), ("nearest", "ignitions_outer_per_min")]


def region_metrics(d, f, fps, t0, edge):
    T, N = int(d["T"]), int(d["N"])
    a = int(np.ceil(t0 * fps))
    frames, minutes = T - a, (T - a) / fps / 60
    wl = f[(f["t"] >= a) & (f["size"] >= crs.LARGE)]
    out, s = wl["rho"].to_numpy() >= edge, wl["size"].to_numpy(float)
    lt = crs.large_track_table(f, fps)
    ign = lt[lt["formed"] & (lt["t_ign"] >= a)]
    io = ign["rho_ign"].to_numpy() >= edge
    dt, drho = crs.decay_events(f, d["tracks"])
    do = drho[dt >= a] >= edge
    return {"large_robots_inner_of_N": s[~out].sum() / (N * frames),
            "large_robots_outer_of_N": s[out].sum() / (N * frames),
            "n_large_inner": (~out).sum() / frames,
            "n_large_outer": out.sum() / frames,
            "ignitions_inner_per_min": (~io).sum() / minutes,
            "ignitions_outer_per_min": io.sum() / minutes,
            "decay_per_groupmin_inner": (~do).sum() / ((~out).sum() / fps / 60),
            "decay_per_groupmin_outer": do.sum() / (out.sum() / fps / 60),
            "outer_share_of_large_robots": s[out].sum() / s.sum(),
            "outer_share_of_ignitions": io.mean() if len(io) else np.nan}


def robot_position_metrics(d, fps, t0, k):
    a = int(np.ceil(t0 * fps))
    h = d["hist_swarm"][a:].sum(axis=0).astype(float)
    tot, large = h.sum(axis=(1, 2)), h[:, 2:].sum(axis=(1, 2))
    return {"robot_large_frac_inner": large[:k].sum() / tot[:k].sum(),
            "robot_large_frac_outer": large[k:].sum() / tot[k:].sum(),
            "robot_large_inner_of_N": large[:k].sum() / tot.sum(),
            "robot_large_outer_of_N": large[k:].sum() / tot.sum(),
            "robot_density_outer_frac": tot[k:].sum() / tot.sum()}


def test(a, b):
    """a - b per trial, paired by position."""
    d = a - b
    q = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    sign = 1 if d.mean() >= 0 else -1
    return dict(baseline=b.mean(), strat=a.mean(), diff=d.mean(), ci_lo=d.mean() - q, ci_hi=d.mean() + q,
                pairs_up=int((d > 0).sum()), paired_t_p=stats.ttest_rel(a, b).pvalue,
                signflip_p=min(1.0, 2 * crs.signflip_p(sign * d)), welch_p=stats.ttest_ind(a, b, equal_var=False).pvalue)


def holm(p):
    p = np.asarray(p, float)
    order, adj, run = np.argsort(p), np.empty(len(p)), 0.0
    for i, j in enumerate(order):
        run = max(run, (len(p) - i) * p[j])
        adj[j] = min(run, 1.0)
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("results")
    ap.add_argument("conds", nargs="+", metavar="NAME=EXP")
    ap.add_argument("--edge", type=float, default=0.75)
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--bare", action="store_true", help="also write each figure as <name>_bare, with no text at all")
    a = ap.parse_args()
    os.makedirs(a.results, exist_ok=True)
    exps = dict(x.split("=", 1) for x in a.conds)
    names = list(exps)
    base, strats = names[0], names[1:]
    if base != "baseline":
        raise SystemExit("the first NAME=EXP must be baseline=...")

    k_lo = int(np.floor(a.edge ** 2 / 1.21 * 20))
    ks = [k_lo] if np.isclose(crs.rho_edge(k_lo), a.edge) else [k_lo, k_lo + 1]
    rows, robot = [], []
    for c, e in exps.items():
        for tn in range(a.trials):
            d, f, fps = crs.load_trial(os.path.join(a.out, "cache", f"{e}__trial_{tn:04d}.npz"))
            for w, t0 in crs.WINDOWS.items():
                rows.append(dict(cond=c, trial=tn, window=w, **region_metrics(d, f, fps, t0, a.edge)))
                for k in ks:
                    robot.append(dict(cond=c, trial=tn, window=w, edge_rho=crs.rho_edge(k),
                                      **robot_position_metrics(d, fps, t0, k)))
    M, Rb = pd.DataFrame(rows), pd.DataFrame(robot)
    M.to_csv(os.path.join(a.results, "region_metrics_per_trial.csv"), index=False)
    Rb.to_csv(os.path.join(a.results, "robot_position_per_trial.csv"), index=False)

    def get(t, c, me):
        return t[t.cond == c].sort_values("trial")[me].to_numpy(float)

    out = []
    for w in crs.WINDOWS:
        t = M[M.window == w]
        for c in strats:
            for me in METRICS:
                out.append(dict(window=w, strategy=c, metric=me, **test(get(t, c, me), get(t, base, me))))
    T = pd.DataFrame(out)
    T["is_claim"] = [(s, m) in CLAIMS for s, m in zip(T.strategy, T.metric)]
    T["holm_p"] = np.nan
    for w in crs.WINDOWS:
        q = T.is_claim & (T.window == w)
        T.loc[q, "holm_p"] = holm(T.loc[q, "paired_t_p"])
    T.to_csv(os.path.join(a.results, "region_tests.csv"), index=False)

    rt = []
    for (w, er), t in Rb.groupby(["window", "edge_rho"]):
        for c in strats:
            for me in ("robot_large_frac_inner", "robot_large_frac_outer", "robot_large_inner_of_N",
                       "robot_large_outer_of_N", "robot_density_outer_frac"):
                rt.append(dict(window=w, edge_rho=er, strategy=c, metric=me, **test(get(t, c, me), get(t, base, me))))
    RT = pd.DataFrame(rt)
    RT.to_csv(os.path.join(a.results, "robot_position_tests.csv"), index=False)

    pd.set_option("display.width", 220, "display.float_format", "{:.4f}".format, "display.max_rows", 300)
    print(T.drop(columns="is_claim").to_string())
    print(RT.to_string())

    crf.EDGE = a.edge
    crf.ALSO_BARE = a.bare
    crf.STRATS[:] = strats
    crf.CONDS[:] = names
    res, ed = crf.spatial_reduction(exps, a.out)
    crf.fig_maps(res, ed, a.results)
    crf.fig_radial(res, a.results)
    w = "t>=100"
    fig, axs = crf.plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for j, (region, lab) in enumerate((("inner", f"内圈（r < {a.edge:g} R）"), ("outer", f"外圈（r ≥ {a.edge:g} R）"))):
        crf.dots(axs[j, 0], M, f"large_robots_{region}_of_N", w, f"{lab}：≥6 group 中的机器人", "占总数（%）", 100)
        crf.dots(axs[j, 1], M, f"n_large_{region}", w, f"{lab}：≥6 group 数量", "个")
        crf.dots(axs[j, 2], M, f"ignitions_{region}_per_min", w, f"{lab}：形成速率", "次 / min")
    crf.save(fig, a.results, "fig_inner_outer_paired")


if __name__ == "__main__":
    main()

"""
catalyst_wall_stability.py
--------------------------
Are large groups more stable near the wall than in the interior? A within-trial
comparison, so each condition is its own control; the baseline is the cleanest read,
the two strategies are replications with catalysts in different places.

Uses catalyst_radial_build.py's cache and writes into <out>/wall_stability/.

Regions, by the group centroid's r / INNER_R (reference point: swarm centroid):
    inner  r < 0.778
    gap    0.778 <= r < 0.852   the inner part of the outer ring
    wall   r >= 0.852           the layer against the wall
    outer  gap + wall

Measures (large = size >= 6). Each one guards against a different artefact:
    decay_hazard       decays per large-group-minute, by current region. Plain rate; can
                       be flattered by larger groups or by long-lived groups drifting out.
    decay_hazard_6_7   same, counting only time at size 6-7: size matched, since only a
                       group at the threshold can decay in one frame.
    split_hazard       fragments shed per large-group-minute, directly standardised over
                       size classes 6-9 / 10-19 / 20+ so a size difference cannot explain it.
    surv60             Kaplan-Meier survival 60 s after formation, by where the group FORMED.
                       Every group starts at size ~6 and location is fixed before the
                       outcome, so "survivors drift to the wall" cannot produce a difference.
    median_life_s      Kaplan-Meier median of the same.
    retention30        of a large group's members now, largest share still together 30 s
                       later, standardised over size classes. Independent of the track-ID
                       rules. Also still_large30: that subset is still in a group >= 6.
    mean_size          mean size of large groups in the region (context, not stability).

Statistics: per-trial values, within-trial ratio or difference between regions,
95% t-interval (on the log ratio for hazards), exact one-sided sign-flip p
(floor 1/32 per condition; 2^-15 over all 15 trials, which share ICs by condition).

Usage:
    python catalyst_wall_stability.py <out> <baseline_dir> <farthest_dir> <nearest_dir> [--jobs 3]
"""
import argparse
import itertools
import multiprocessing as mp
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import catalyst_radial_stats as crs

GAP_EDGE, WALL_EDGE = crs.rho_edge(10), crs.rho_edge(12)
PROFILE_EDGES = [0, crs.rho_edge(5), crs.rho_edge(8), GAP_EDGE, WALL_EDGE, 9]
PROFILE_LABELS = ["<0.55", "0.55-0.70", "0.70-0.78", "0.78-0.85", ">=0.85"]
SIZE_CLASSES = [(6, 9), (10, 19), (20, 10**6)]
TAUS = (10, 30, 60)
WINDOWS = {"t>=100": 100.0, "t>=400": 400.0}
CONDS = ["baseline", "farthest", "nearest"]


def region(rho):
    return np.where(rho < GAP_EDGE, "inner", np.where(rho < WALL_EDGE, "gap", "wall"))


# ------------------------------------------------------------------ robot-level retention
def retention_job(job):
    """Sample large groups every 5 s; follow their members tau seconds on."""
    exp_dir, trial_dir, out_csv = job
    if os.path.exists(out_csv):
        return f"{out_csv}: cached"
    from catalyst_radial_build import load_xy
    from group_lifetimes import load_labels, smooth_labels
    trial = os.path.basename(trial_dir)
    prefix = os.path.join(trial_dir, trial)
    L = load_labels(prefix + "_groups_d65.csv")
    T, N = L.shape
    fps, R = 60, 1328.0
    S = smooth_labels(L, 30)
    del L
    xy = load_xy(prefix + "_POS_ALL.csv", T, N)
    swarm = np.nanmean(xy, axis=1)
    rows = []
    for t in range(50 * fps, T, 5 * fps):
        ug, cnt = np.unique(S[t], return_counts=True)
        for g in ug[cnt >= 6]:
            idx = np.flatnonzero(S[t] == g)
            p = xy[t, idx] - swarm[t]
            rm = np.hypot(p[:, 0], p[:, 1]) / R
            row = dict(t=t / fps, size=len(idx), rho=np.hypot(*p.mean(axis=0)) / R, wall_member_frac=(rm >= WALL_EDGE).mean())
            for tau in TAUS:
                t2 = t + tau * fps
                if t2 >= T:
                    row[f"kept{tau}"], row[f"large{tau}"] = np.nan, np.nan
                    continue
                lab = S[t2, idx]
                u, c = np.unique(lab, return_counts=True)
                dom = u[np.argmax(c)]
                row[f"kept{tau}"] = c.max() / len(idx)
                row[f"large{tau}"] = float(c.max() * 2 >= len(idx) and (S[t2] == dom).sum() >= 6)
            rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return f"{out_csv}: done"


# ------------------------------------------------------------------ per-trial metrics
def standardise(values, weights):
    """Weighted mean over the strata both regions have; values: {class: value or nan}."""
    v = np.array([values[c] for c in range(len(SIZE_CLASSES))], float)
    w = np.array(weights, float)
    ok = np.isfinite(v)
    return (v[ok] * w[ok]).sum() / w[ok].sum() if ok.any() else np.nan


def km_at(dur, obs, at):
    if len(dur) == 0:
        return np.nan
    x, y = crs.km_curve(dur, obs)
    return y[np.searchsorted(x, at, side="right") - 1]


def trial_metrics(path, ret, t0, weights):
    d, f, fps = crs.load_trial(path)
    T = int(d["T"])
    a = int(np.ceil(t0 * fps))
    big = f[f["size"] >= crs.LARGE].copy()
    big["reg"] = region(big["rho"].to_numpy())
    w = big[big.t >= a]

    dt, drho = crs.decay_events(f, d["tracks"])
    k = dt >= a
    dreg = region(drho[k])

    ev = pd.DataFrame.from_records(d["events"])
    sp = ev[(ev.event == "split") & (ev.step >= a)][["step", "other_id"]].copy()
    sp["t"] = sp.step - 1
    sp = sp.merge(big[["t", "id", "size", "reg"]], left_on=["t", "other_id"], right_on=["t", "id"], how="inner")

    lt = crs.large_track_table(f, fps)
    ign = lt[lt.formed & (lt.t_ign >= a)].copy()
    ign["reg"] = region(ign.rho_ign.to_numpy())
    ign["dur"] = (ign.t_end - ign.t_ign + 1) / fps
    ign["obs"] = ign.t_end < T - 1

    r = ret[ret.t >= t0].copy()
    r["reg"] = region(r.rho.to_numpy())

    regions = {"inner": ["inner"], "outer": ["gap", "wall"], "gap": ["gap"], "wall": ["wall"]}
    out = []
    for name, regs in regions.items():
        g = w[w.reg.isin(regs)]
        minutes = len(g) / fps / 60
        m = dict(region=name, group_minutes=minutes, mean_size=g["size"].mean() if len(g) else np.nan)
        m["decay_hazard"] = np.isin(dreg, regs).sum() / minutes if minutes else np.nan
        # a decay always leaves from size 6, so all of them count against time at 6-7
        min67 = g["size"].between(6, 7).sum() / fps / 60
        m["decay_hazard_6_7"] = np.isin(dreg, regs).sum() / min67 if min67 else np.nan
        hz, kept, stay = {}, {}, {}
        rg = r[r.reg.isin(regs)]
        sg = sp[sp.reg.isin(regs)]
        for c, (lo, hi) in enumerate(SIZE_CLASSES):
            mins = g["size"].between(lo, hi).sum() / fps / 60
            hz[c] = sg["size"].between(lo, hi).sum() / mins if mins > 0.5 else np.nan
            x = rg[rg["size"].between(lo, hi)]
            kept[c] = x.kept30.mean() if x.kept30.notna().sum() >= 3 else np.nan
            stay[c] = x.large30.mean() if x.large30.notna().sum() >= 3 else np.nan
        m["split_hazard"] = standardise(hz, weights)
        m["retention30"] = standardise(kept, weights)
        m["still_large30"] = standardise(stay, weights)
        for tau in (10, 60):
            m[f"retention{tau}_6_9"] = rg[rg["size"].between(6, 9)][f"kept{tau}"].mean()
        m["retention30_6_9"] = kept[0]
        s = ign[ign.reg.isin(regs)]
        m["n_formed"] = len(s)
        m["surv60"] = km_at(s.dur.to_numpy(), s.obs.to_numpy(), 60.0)
        m["median_life_s"] = crs.km_median(s.dur.to_numpy(), s.obs.to_numpy())
        out.append(m)
    # membership-based wall definition, as a check on using the centroid
    rm = r[(r.wall_member_frac >= 0.5)]
    rn = r[(r.wall_member_frac < 0.5) & (r.rho < GAP_EDGE)]
    prof = []
    for lo, hi, lab in zip(PROFILE_EDGES[:-1], PROFILE_EDGES[1:], PROFILE_LABELS):
        g = w[(w.rho >= lo) & (w.rho < hi)]
        minutes = len(g) / fps / 60
        x = r[(r.rho >= lo) & (r.rho < hi)]
        prof.append(dict(bin=lab, decay_hazard=((drho[k] >= lo) & (drho[k] < hi)).sum() / minutes if minutes else np.nan,
                         retention30_6_9=x[x["size"].between(6, 9)].kept30.mean(), mean_size=g["size"].mean()))
    check = dict(ret_member_wall=rm[rm["size"].between(6, 9)].kept30.mean(),
                 ret_member_inner=rn[rn["size"].between(6, 9)].kept30.mean())
    return pd.DataFrame(out), pd.DataFrame(prof), check, ign[["dur", "obs", "reg"]]


def signflip_p(x):
    flips = np.array(list(itertools.product([1, -1], repeat=len(x))))
    return np.mean((flips * x).mean(axis=1) >= x.mean() - 1e-12)


def contrast(a, b, better_high, log):
    """a vs b per trial; positive 'favour' means a is more stable than b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0) if log else np.isfinite(a) & np.isfinite(b)
    a, b = a[ok], b[ok]
    n = len(a)
    if n < 3:
        return dict(n=n)
    d = np.log(a / b) if log else a - b
    se = d.std(ddof=1) / np.sqrt(n)
    q = stats.t.ppf(0.975, n - 1)
    lo, hi = d.mean() - q * se, d.mean() + q * se
    fav = d if better_high else -d
    res = dict(n=n, a=a.mean(), b=b.mean(), pairs_favour=int((fav > 0).sum()), p=signflip_p(fav))
    if n <= 15:
        res["effect"] = np.exp(d.mean()) if log else d.mean()
        res["lo"], res["hi"] = (np.exp(lo), np.exp(hi)) if log else (lo, hi)
    return res


METRICS = [("decay_hazard", False, True), ("decay_hazard_6_7", False, True), ("split_hazard", False, True),
           ("surv60", True, False), ("median_life_s", True, True), ("retention30", True, False),
           ("still_large30", True, False)]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("baseline")
    ap.add_argument("farthest")
    ap.add_argument("nearest")
    ap.add_argument("--jobs", type=int, default=3)
    a = ap.parse_args()
    dirs = dict(baseline=a.baseline, farthest=a.farthest, nearest=a.nearest)
    exps = {c: os.path.basename(os.path.normpath(p)) for c, p in dirs.items()}
    wd = os.path.join(a.out, "wall_stability")
    os.makedirs(wd, exist_ok=True)

    jobs = [(dirs[c], os.path.join(dirs[c], f"trial_{tn:04d}"), os.path.join(wd, f"retention_{exps[c]}_{tn}.csv"))
            for c in CONDS for tn in range(5)]
    with mp.Pool(a.jobs) as pool:
        for msg in pool.imap_unordered(retention_job, jobs):
            print(msg, flush=True)

    # fixed weights for the size standardisation: pooled large-group time over everything
    tot = np.zeros(len(SIZE_CLASSES))
    for c in CONDS:
        for tn in range(5):
            z = np.load(os.path.join(a.out, "cache", f"{exps[c]}__trial_{tn:04d}.npz"), allow_pickle=True)
            s = z["frames_size"]
            for i, (lo, hi) in enumerate(SIZE_CLASSES):
                tot[i] += ((s >= lo) & (s <= hi)).sum()
    weights = tot / tot.sum()
    print("size-class weights", weights.round(3), flush=True)

    rows, profs, checks, igns = [], [], [], []
    for c in CONDS:
        for tn in range(5):
            ret = pd.read_csv(os.path.join(wd, f"retention_{exps[c]}_{tn}.csv"))
            path = os.path.join(a.out, "cache", f"{exps[c]}__trial_{tn:04d}.npz")
            for win, t0 in WINDOWS.items():
                m, p, ck, ig = trial_metrics(path, ret, t0, weights)
                rows.append(m.assign(cond=c, trial=tn, window=win))
                profs.append(p.assign(cond=c, trial=tn, window=win))
                checks.append(dict(cond=c, trial=tn, window=win, **ck))
                igns.append(ig.assign(cond=c, trial=tn, window=win))
            print(c, tn, flush=True)
    M = pd.concat(rows)
    P = pd.concat(profs)
    K = pd.DataFrame(checks)
    I = pd.concat(igns)
    M.to_csv(os.path.join(wd, "metrics_per_trial_region.csv"), index=False)
    P.to_csv(os.path.join(wd, "radial_profile_per_trial.csv"), index=False)
    I.to_csv(os.path.join(wd, "formation_lifetimes.csv"), index=False)

    C = []
    pairs = [("outer", "inner"), ("wall", "inner"), ("gap", "inner"), ("wall", "gap")]
    for win in WINDOWS:
        for x, y in pairs:
            for me, high, log in METRICS:
                if win == "t>=400" and me in ("surv60", "median_life_s"):
                    continue    # a 100 s window censors lifetimes at <= 100 s
                vals = {}
                for c in CONDS:
                    t = M[(M.window == win) & (M.cond == c)].pivot(index="trial", columns="region", values=me)
                    vals[c] = (t[x].to_numpy(), t[y].to_numpy())
                    C.append(dict(window=win, contrast=f"{x} vs {y}", metric=me, cond=c, **contrast(*vals[c], high, log)))
                A = np.concatenate([vals[c][0] for c in CONDS])
                B = np.concatenate([vals[c][1] for c in CONDS])
                A, B = np.asarray(A, float), np.asarray(B, float)
                ok = np.isfinite(A) & np.isfinite(B)
                d = (A - B)[ok] * (1 if high else -1)
                C.append(dict(window=win, contrast=f"{x} vs {y}", metric=me, cond="all", n=int(ok.sum()), a=A[ok].mean(), b=B[ok].mean(),
                              pairs_favour=int((d > 0).sum()), p=stats.binomtest(int((d > 0).sum()), int(ok.sum()), 0.5, alternative="greater").pvalue))
    C = pd.DataFrame(C)
    C.to_csv(os.path.join(wd, "contrasts.csv"), index=False)
    pd.set_option("display.width", 220, "display.max_rows", 400, "display.float_format", "{:.3f}".format)
    print(C.to_string())
    print(M[M.window == "t>=100"].groupby(["cond", "region"])[["group_minutes", "mean_size", "n_formed"]].mean().to_string())
    print(P[P.window == "t>=100"].groupby(["cond", "bin"], sort=False)[["decay_hazard", "retention30_6_9", "mean_size"]].mean().to_string())
    print(K[K.window == "t>=100"].groupby("cond")[["ret_member_wall", "ret_member_inner"]].mean().to_string())


if __name__ == "__main__":
    main()

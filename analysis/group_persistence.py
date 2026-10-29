"""
group_persistence.py  -  identity-free stability of the grouping (1014 sweep).

Kaplan-Meier on track lifetimes needs every group to carry an identity, and
that identity is fragile: one robot leaving can count as a "split", the
largest-predecessor rule decides who survives a merge, and the large groups
that matter most are the ones most often still alive at t=500s (censored).
Pooled per condition, those groups are also not independent samples.

This measures stability without tracking groups at all. It asks each robot:
"of the robots I am grouped with now, what fraction am I still grouped with
tau seconds later?"

    q_i(t, tau) = |G_i(t) & G_i(t+tau)| / |G_i(t) | G_i(t+tau)|

G_i(t) is the set of *other* robots in i's group (groups of size >= MIN_SIZE;
anything smaller counts as the empty set). q is averaged over every robot that
is grouped at t and every origin t in the window. q(tau) = 1 means the
partition is frozen; q(tau) -> chance level means memberships are fully
reshuffled. One curve, hence one number, per trial: the sample is trials and
trials are paired by initial condition (GROUP_STATS_METHOD.md).

Headline numbers per trial:
    q_<tau>s     overlap at fixed lags
    tau_half_s   lag at which q drops to 0.5 (NaN if it does not within MAX_LAG_S)
    tau_int_s    integral of q over [0, MAX_LAG_S] -- never censored

Flicker: Voronoi contacts blink on and off at sub-second scale. Positions are
sampled at SAMPLE_HZ and a pair counts as bonded in a BLOCK_S block only if
it was co-grouped in more than half of that block's samples (the same majority
vote as group_lifetimes.py's hysteresis).

    python group_persistence.py <sweep_root> [--jobs 10]

Writes <sweep_root>/_persistence_per_trial.csv and _persistence_curves.npz.
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

# The simulation modules (config, pos_all_grouping, strategy, gait, ...)
# live one level up. Running this as `python analysis/<name>.py` puts only
# analysis/ on the path, so the project root has to be added explicitly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pos_all_grouping import compute_frame_groups

FPS, MAXD = 60.0, 65.0
SAMPLE_HZ = 6.0
BLOCK_S = 2.0
MIN_SIZE = 3
T0_S = 200.0             # origins t >= T0_S
MAX_LAG_S = 120.0
FIXED_LAGS_S = (10, 30, 60, 120)


def sampled_labels(trial_dir):
    """-> (sample times s, (S, N) group label per robot) at SAMPLE_HZ.

    Reads the groups_d<MAXD> cache written by batch_group_plots.py when there
    is one, and groups POS_ALL directly otherwise.
    """
    stride = int(round(FPS / SAMPLE_HZ))
    cache = glob.glob(os.path.join(trial_dir, f"*_groups_d{MAXD:g}.csv"))
    if cache:
        from group_lifetimes import load_labels
        L = load_labels(cache[0], stride)
        return np.arange(len(L)) * stride / FPS, L
    return labels_from_pos_all(glob.glob(os.path.join(trial_dir, "*_POS_ALL.csv"))[0], stride)


def labels_from_pos_all(path, stride):
    df = pd.read_csv(path, usecols=["Step", "Agent_ID", "X", "Y", "Theta"])
    df = df[df["Step"] % stride == 0].sort_values(["Step", "Agent_ID"], kind="mergesort")
    steps = np.unique(df["Step"].to_numpy())
    aid_all = df["Agent_ID"].to_numpy()
    N = int(aid_all.max())
    xy = df[["X", "Y"]].to_numpy(float).reshape(len(steps), -1, 2)
    th = df["Theta"].to_numpy(float).reshape(len(steps), -1)
    aid = aid_all.reshape(len(steps), -1)

    # per-sample group label of every robot (singletons get their own label)
    lab = np.empty((len(steps), N), np.int32)
    for k in range(len(steps)):
        for g_i, g in enumerate(compute_frame_groups(xy[k], th[k], aid[k], MAXD, True)):
            lab[k, g["agent_ids"] - 1] = g_i
    return steps / FPS, lab


def block_comembership(t, lab):
    """-> (block start times s, (B, N, N) bool co-membership, groups >= MIN_SIZE)."""
    N = lab.shape[1]
    per = int(round(BLOCK_S * SAMPLE_HZ))
    nb = len(t) // per
    out = np.zeros((nb, N, N), bool)
    for b in range(nb):
        L = lab[b * per:(b + 1) * per]
        co = (L[:, :, None] == L[:, None, :]).sum(0) * 2 > per
        _, c = connected_components(csr_matrix(co), directed=False)
        sizes = np.bincount(c)
        big = sizes[c] >= MIN_SIZE
        m = (c[:, None] == c[None, :]) & big[:, None] & big[None, :]
        np.fill_diagonal(m, False)
        out[b] = m
    return t[::per][:nb], out


def overlap_curve(t, C, t0=T0_S, max_lag=MAX_LAG_S):
    """Memory curves for tau = 0, BLOCK_S, ..., max_lag, origins t >= t0.

    q          robot-level Jaccard overlap (intermittent)
    pair       P(pair grouped at t+tau | grouped at t)       -- intermittent
    pair_cont  P(pair grouped throughout [t, t+tau] | grouped at t)
               -- a survival function of partnerships, with no group
               identity and no censoring: only origins with t+tau inside the
               run are used, so every partnership is followed for the full tau
    chance     P(a random pair is grouped), the level `pair` decays to once
               memberships are fully reshuffled
    """
    nlag = int(round(max_lag / BLOCK_S)) + 1
    o0 = np.searchsorted(t, t0)
    q, pair, cont = (np.full(nlag, np.nan) for _ in range(3))
    alive = C[o0:].copy()
    for k in range(nlag):
        a = C[o0:len(C) - k]
        b = C[o0 + k:]
        if len(a) == 0:
            break
        alive = alive[:len(a)] & b
        inter = (a & b).sum(2)
        union = (a | b).sum(2)
        sz = a.sum(2)
        grouped = sz > 0
        q[k] = (inter[grouped] / union[grouped]).mean()
        pair[k] = inter.sum() / sz.sum()       # pair persistence, size-weighted
        cont[k] = alive.sum() / sz.sum()
    N = C.shape[1]
    chance = C[o0:].sum() / (len(C[o0:]) * N * (N - 1))
    return np.arange(nlag) * BLOCK_S, q, pair, cont, chance


def summarize_curve(lags, q):
    row = {f"q_{L}s": q[int(round(L / BLOCK_S))] for L in FIXED_LAGS_S}
    below = np.flatnonzero(q < 0.5)
    if len(below):
        k = below[0]
        row["tau_half_s"] = lags[k - 1] + (q[k - 1] - 0.5) / (q[k - 1] - q[k]) * BLOCK_S
    else:
        row["tau_half_s"] = np.nan
    row["tau_int_s"] = np.trapezoid(q, lags) if hasattr(np, "trapezoid") else np.trapz(q, lags)
    return row


def run_trial(job):
    pattern, trial_dir = job
    t, C = block_comembership(*sampled_labels(trial_dir))
    lags, q, pair, _, _ = overlap_curve(t, C)
    o0 = np.searchsorted(t, T0_S)
    row = dict(pattern=pattern, trial_id=int(os.path.basename(trial_dir).split("_")[1]),
               frac_grouped=C[o0:].any(2).mean(), **summarize_curve(lags, q),
               pair_q_30s=pair[int(round(30 / BLOCK_S))])
    return row, lags, q, pair


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root")
    ap.add_argument("--jobs", type=int, default=10)
    args = ap.parse_args()

    jobs = []
    for exp in sorted(glob.glob(os.path.join(args.root, "a[0-9]*"))):
        if os.path.isdir(exp):
            jobs += [(os.path.basename(exp), d) for d in sorted(glob.glob(os.path.join(exp, "trial_*")))]
    with Pool(args.jobs) as p:
        res = p.map(run_trial, jobs, chunksize=1)

    rows = pd.DataFrame([r[0] for r in res])
    ic = pd.concat([pd.read_csv(os.path.join(args.root, f"{p}_summary.csv"), usecols=["trial_id", "ic_idx"])
                    .assign(pattern=p) for p in rows.pattern.unique()])
    rows = rows.merge(ic, on=["pattern", "trial_id"], how="left")
    rows.to_csv(os.path.join(args.root, "_persistence_per_trial.csv"), index=False)
    np.savez(os.path.join(args.root, "_persistence_curves.npz"), lags=res[0][1],
             q=np.array([r[2] for r in res]), pair=np.array([r[3] for r in res]),
             pattern=rows.pattern.to_numpy(), trial_id=rows.trial_id.to_numpy())
    print(rows.groupby("pattern").agg(["mean", "std"]).round(3).to_string())


if __name__ == "__main__":
    main()

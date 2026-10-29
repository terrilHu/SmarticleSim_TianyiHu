"""
group_lifetimes.py
------------------
Give every group a persistent identity across frames, and log when it is born,
merges, splits and dies -- the raw material for group lifetime / stability.

Input is the per-trial Voronoi grouping cache written by batch_group_plots.py:

    trial_XXXX/trial_XXXX_groups_d<max_dist>.csv   one row per group per frame

Its `group_local_id` is only an index within the frame, so identity has to be
rebuilt here. Three steps:

1. Hysteresis. At 60 fps, two robots sitting next to each other drop in and out
   of the same group every few frames (median contact run ~0.5 s, ~85% of
   on/off runs < 1 s, while the persistent ones last tens to hundreds of
   seconds). Tracked raw, every one of those flickers is a merge followed by a
   split. So we first smooth **co-membership**: robots i and j count as
   grouped at frame t only if they shared a group in more than half of the
   frames in a centred window of --win-s seconds. Groups are then the connected
   components of that smoothed graph. --win-s 0 disables smoothing and
   reproduces the raw grouping exactly.

2. Size cut. Groups smaller than --min-size (default 3) are not groups; a robot
   in one is treated like a singleton.

3. Frame-to-frame matching. Group A at t-1 and group B at t are linked when
   they share at least half of the smaller one's members. Then, per link
   pattern:
       one-to-one          B continues A's track
       many-to-one (merge) B takes the track of the **largest** predecessor;
                           the others die by "merge"
       one-to-many (split) the **largest** fragment keeps A's track; the
                           others are born by "split"
       no predecessor      born "new"  (e.g. a pair picking up a third robot)
       no successor        dies "vanish" (falls below --min-size or dissolves)
   Ties on size are broken by the number of shared members.

Output, written next to the input:

    <trial>_tracks_d<d>_m<min>_w<win>.csv   one row per track
    <trial>_events_d<d>_m<min>_w<win>.csv   one row per birth/merge/split/death

Censoring: a track alive in the first frame has an unknown birth
(censored_left), one alive in the last frame an unknown death
(censored_right). Plain means of `lifetime_s` are biased by both; the summary
printed at the end drops left-censored tracks and uses Kaplan-Meier for the
right-censored ones.

Given an experiment directory instead, every trial_* in it is processed and
the tracks are pooled into (written into the experiment directory):

    <exp>_tracks_<tag>.csv           all tracks, with a `trial` column
    <exp>_lifetime_trials_<tag>.csv  one row of lifetime / event-rate stats per trial
    <exp>_lifetime_<tag>.png         survival curves (pooled, per trial, by size) + histogram

Usage:
    python group_lifetimes.py <trial_dir> [--max-dist 65] [--min-size 3] [--win-s 1.0]
    python group_lifetimes.py <exp_dir>   [--jobs 4]
"""

import argparse
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def load_labels(groups_csv, stride=1):
    """(T, N) array: frame-local group id of each robot (robot ids 1..N -> columns 0..N-1).

    stride > 1 keeps only frames with Step % stride == 0 (same as [::stride] on
    the full array). The csv is parsed in chunks: at N = 1000 it is ~0.8 GB, and
    splitting every member list at once takes several GB.
    """
    parts = []
    for ch in pd.read_csv(groups_csv, usecols=["Step", "group_local_id", "member_ids"], chunksize=500_000):
        ch = ch[ch["Step"] % stride == 0]
        members = ch["member_ids"].astype(str).str.split()
        lens = members.str.len().to_numpy()
        parts.append((np.repeat(ch["Step"].to_numpy(dtype=np.int64) // stride, lens),
                      np.repeat(ch["group_local_id"].to_numpy(dtype=np.int32), lens),
                      np.concatenate(members.to_numpy()).astype(np.int32) if len(ch) else np.empty(0, np.int32)))
    steps, gids, ids = (np.concatenate(x) for x in zip(*parts))
    T, N = steps.max() + 1, ids.max()
    L = np.full((T, N), -1, dtype=np.int32)
    L[steps, ids - 1] = gids
    if (L < 0).any():
        raise ValueError(f"{groups_csv}: some robot is missing from some frame")
    return L


def co_grouped_pairs(L):
    """(i, j), i < j, of every robot pair sharing a group in at least one row of L."""
    N = L.shape[1]
    o = np.argsort(L, axis=1, kind="stable")
    s = np.take_along_axis(L, o, axis=1)
    codes = []
    # members of one group sit next to each other after the sort, so a pair
    # k apart in sorted order is co-grouped iff its labels match
    for k in range(1, N):
        m = s[:, k:] == s[:, :-k]
        if not m.any():
            break
        a, b = o[:, :-k][m], o[:, k:][m]
        codes.append(np.unique(np.minimum(a, b) * N + np.maximum(a, b)))
    codes = np.unique(np.concatenate(codes)) if codes else np.zeros(0, np.int64)
    return codes // N, codes % N


def smooth_labels(L, half, block=2000):
    """Majority-vote co-membership over [t-half, t+half], then connected components.

    Done in blocks of frames, and per block only over pairs that are grouped
    at least once in the block (plus its half-window margins) -- every other
    pair is apart throughout and can never win the vote. That keeps memory
    flat in N: at N=500 there are 125k pairs but only ~2k ever co-grouped
    within 2000 frames.
    """
    T, N = L.shape
    if half == 0:
        return L
    out = np.empty_like(L)
    for b in range(0, T, block):
        e = min(b + block, T)
        w0, w1 = max(b - half, 0), min(e + half, T)
        W = L[w0:w1]
        pi, pj = co_grouped_pairs(W)
        co = (W[:, pi] == W[:, pj]).astype(np.int32)
        cs = np.vstack([np.zeros((1, len(pi)), np.int32), np.cumsum(co, axis=0)])
        t = np.arange(b, e)
        lo = np.maximum(t - half, 0) - w0
        hi = np.minimum(t + half + 1, T) - w0
        keep = 2 * (cs[hi] - cs[lo]) > (hi - lo)[:, None]
        for k in range(e - b):
            sel = np.flatnonzero(keep[k])
            g = coo_matrix((np.ones(len(sel)), (pi[sel], pj[sel])), shape=(N, N))
            out[b + k] = connected_components(g, directed=False)[1]
    return out


def frame_groups(labels, min_size):
    """Groups of one frame as (bitmask, size, member indices), size >= min_size."""
    out = []
    ug, counts = np.unique(labels, return_counts=True)
    for g in ug[counts >= min_size]:
        idx = np.flatnonzero(labels == g)
        out.append((sum(1 << int(i) for i in idx), len(idx), idx))
    return out


def track(labels, min_size):
    T, N = labels.shape
    tracks = {}       # id -> dict
    events = []
    next_id = 0
    prev = []         # [(track_id, mask, size, member indices)]

    def new_track(t, mask, size, birth, parent):
        nonlocal next_id
        tid = next_id
        next_id += 1
        tracks[tid] = dict(track_id=tid, t_start=t, t_end=t, birth=birth,
                           birth_parent=parent, death="end", death_into=-1,
                           first_mask=mask, last_mask=mask, size_sum=0,
                           max_size=0, n_absorbed=0, n_shed=0)
        return tid

    for t in range(T):
        cur = frame_groups(labels[t], min_size)
        # robot -> index into prev, for overlap counting
        owner = np.full(N, -1, dtype=np.int64)
        for i, p in enumerate(prev):
            owner[p[3]] = i
        links = []     # (i_prev, j_cur, shared)
        for j, (m, s, idx) in enumerate(cur):
            o = owner[idx]
            cnt = np.bincount(o[o >= 0], minlength=len(prev))
            for i in np.flatnonzero(cnt):
                if 2 * cnt[i] >= min(prev[i][2], s):
                    links.append((i, j, int(cnt[i])))
        pred = {}
        succ = {}
        for i, j, sh in links:
            if j not in pred or (prev[i][2], sh) > (prev[pred[j][0]][2], pred[j][1]):
                pred[j] = (i, sh)
            if i not in succ or (cur[j][1], sh) > (cur[succ[i][0]][1], succ[i][1]):
                succ[i] = (j, sh)

        new_prev = []
        for j, (m, s, idx) in enumerate(cur):
            if j in pred and succ[pred[j][0]][0] == j:
                tid = prev[pred[j][0]][0]
            elif t == 0:
                tid = new_track(t, m, s, "initial", -1)
            elif j in pred:
                parent = prev[pred[j][0]][0]
                tid = new_track(t, m, s, "split", parent)
                tracks[parent]["n_shed"] += 1
                events.append(dict(step=t, event="split", track_id=tid, other_id=parent, size=s))
            else:
                tid = new_track(t, m, s, "new", -1)
                events.append(dict(step=t, event="birth", track_id=tid, other_id=-1, size=s))
            tr = tracks[tid]
            tr["t_end"] = t
            tr["last_mask"] = m
            tr["size_sum"] += s
            tr["max_size"] = max(tr["max_size"], s)
            new_prev.append((tid, m, s, idx))
        cur_tid = [x[0] for x in new_prev]

        for i, (tid, m, s, _) in enumerate(prev):
            if i in succ and cur_tid[succ[i][0]] == tid:
                continue
            tr = tracks[tid]
            if i in succ:
                into = cur_tid[succ[i][0]]
                tr["death"], tr["death_into"] = "merge", into
                tracks[into]["n_absorbed"] += 1
                events.append(dict(step=t, event="merge", track_id=tid, other_id=into, size=s))
            else:
                tr["death"] = "vanish"
                events.append(dict(step=t, event="death", track_id=tid, other_id=-1, size=s))
        prev = new_prev

    rows = []
    for tr in tracks.values():
        n = tr["t_end"] - tr["t_start"] + 1
        first, last = tr.pop("first_mask"), tr.pop("last_mask")
        tr["n_frames"] = n
        tr["mean_size"] = tr.pop("size_sum") / n
        tr["size_start"] = first.bit_count()
        tr["size_end"] = last.bit_count()
        # share of the founding members still present in the final frame
        tr["retention"] = (first & last).bit_count() / first.bit_count()
        tr["censored_left"] = tr["t_start"] == 0
        tr["censored_right"] = tr["t_end"] == T - 1
        tr["members_start"] = " ".join(str(b + 1) for b in range(N) if first >> b & 1)
        tr["members_end"] = " ".join(str(b + 1) for b in range(N) if last >> b & 1)
        rows.append(tr)
    return pd.DataFrame(rows), pd.DataFrame(events, columns=["step", "event", "track_id", "other_id", "size"])


def kaplan_meier(durations, observed):
    """Survival curve S(t) at each distinct observed event time."""
    order = np.argsort(durations)
    d, o = np.asarray(durations)[order], np.asarray(observed)[order]
    times, surv, s, at_risk = [], [], 1.0, len(d)
    for u in np.unique(d):
        here = d == u
        k = o[here].sum()
        if k:
            s *= 1 - k / at_risk
            times.append(u)
            surv.append(s)
        at_risk -= here.sum()
    return np.array(times), np.array(surv)


def summarize(tr, ev, fps):
    print(f"  tracks: {len(tr)}   (left-censored {tr.censored_left.sum()}, "
          f"right-censored {tr.censored_right.sum()}, both {(tr.censored_left & tr.censored_right).sum()})")
    print("  births:", tr.birth.value_counts().to_dict(), "  deaths:", tr.death.value_counts().to_dict())
    full = tr[~tr.censored_left]
    life = full.lifetime_s.to_numpy()
    print(f"  lifetime (s), tracks with observed birth (n={len(full)}): "
          + "  ".join(f"q{int(q*100)}={np.quantile(life, q):.2f}" for q in (.1, .25, .5, .75, .9)))
    times, surv = kaplan_meier(life, ~full.censored_right.to_numpy())
    for p in (0.5, 0.25, 0.1):
        hit = times[surv <= p]
        print(f"  KM: {int(p*100)}% of groups survive past {hit[0]:.2f} s" if len(hit)
              else f"  KM: survival never drops to {int(p*100)}%")
    for name, lo, hi in SIZE_CLASSES:
        sub = full[(full.max_size >= lo) & (full.max_size <= hi)]
        if len(sub):
            print(f"  max_size {name}: n={len(sub)}, "
                  f"median lifetime {sub.lifetime_s.median():.2f} s, "
                  f"median retention {sub.retention.median():.2f}")
    print("  long-lived (>= 10 s):", (tr.lifetime_s >= 10).sum(),
          "  transient (< 1 s):", (tr.lifetime_s < 1).sum())


def km_quantile(times, surv, p):
    """First time at which KM survival drops to <= p (NaN if it never does)."""
    hit = times[surv <= p]
    return hit[0] if len(hit) else np.nan


def process_trial(trial_dir, max_dist, min_size, win_s, fps):
    prefix = os.path.join(trial_dir, os.path.basename(trial_dir))
    groups_csv = f"{prefix}_groups_d{max_dist}.csv"
    if not os.path.exists(groups_csv):
        raise SystemExit(f"missing {groups_csv}; found: {glob.glob(prefix + '_groups_d*.csv')}")
    half = int(round(win_s * fps / 2))

    L = load_labels(groups_csv)
    S = smooth_labels(L, half)
    tr, ev = track(S, min_size)
    tr["lifetime_s"] = tr.n_frames / fps
    ev.insert(1, "time_s", ev.step / fps)
    for c in ("t_start", "t_end"):
        tr[c.replace("t_", "time_") + "_s"] = tr[c] / fps
    tr.attrs["duration_s"] = L.shape[0] / fps
    tr.attrs["n_robots"] = L.shape[1]

    tag = f"d{max_dist}_m{min_size}_w{win_s:g}"
    tr.to_csv(f"{prefix}_tracks_{tag}.csv", index=False)
    ev.to_csv(f"{prefix}_events_{tag}.csv", index=False)
    print(f"{os.path.basename(trial_dir)}  [{tag}, fps={fps:g}, window={2*half+1} frames]")
    summarize(tr, ev, fps)
    print(f"  wrote {prefix}_tracks_{tag}.csv / _events_{tag}.csv", flush=True)
    return tr, ev


# --------------------------------------------------------------------------
# Experiment level: pool every trial's tracks
# --------------------------------------------------------------------------
LARGE = 6          # a "large" group peaks at >= LARGE robots
SIZE_CLASSES = [(f"3-{LARGE - 1}", 3, LARGE - 1), (f"{LARGE}+", LARGE, 10**6)]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]


def trial_row(trial, tr, ev):
    """Scalar lifetime stats of one trial."""
    full = tr[~tr.censored_left]
    times, surv = kaplan_meier(full.lifetime_s.to_numpy(), ~full.censored_right.to_numpy())
    minutes = tr.attrs["duration_s"] / 60
    robot_time = tr.lifetime_s * tr.mean_size
    total = tr.attrs["duration_s"] * tr.attrs["n_robots"]
    large = full[full.max_size >= LARGE]
    lt, ls = kaplan_meier(large.lifetime_s.to_numpy(), ~large.censored_right.to_numpy())
    return dict(trial=trial, n_tracks=len(tr), n_observed_birth=len(full),
                n_right_censored=int(full.censored_right.sum()),
                km_median_s=km_quantile(times, surv, 0.5),
                km_q75_s=km_quantile(times, surv, 0.25),
                km_q90_s=km_quantile(times, surv, 0.1),
                n_long_10s=int((tr.lifetime_s >= 10).sum()),
                n_large=len(large),
                large_km_median_s=km_quantile(lt, ls, 0.5),
                # share of all robot-time spent in any group / in groups living >= 10 s
                frac_time_grouped=robot_time.sum() / total,
                frac_time_long=robot_time[tr.lifetime_s >= 10].sum() / total,
                mean_size=(tr.mean_size * tr.n_frames).sum() / tr.n_frames.sum(),
                births_per_min=(ev.event == "birth").sum() / minutes,
                splits_per_min=(ev.event == "split").sum() / minutes,
                merges_per_min=(ev.event == "merge").sum() / minutes,
                deaths_per_min=(ev.event == "death").sum() / minutes)


def plot_pooled(pool, per_trial, out_png, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE)
    full = pool[~pool.censored_left]
    xmin = full.lifetime_s.min()
    xmax = pool.lifetime_s.max() * 1.1

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.set_xlim(xmin, xmax)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def step(ax, d, **kw):
        t, s = kaplan_meier(d.lifetime_s.to_numpy(), ~d.censored_right.to_numpy())
        ax.step(np.r_[xmin, t], np.r_[1.0, s], where="post", **kw)
        return t, s

    # A: pooled survival with each trial behind it
    ax = axes[0]
    style(ax)
    for trial, d in full.groupby("trial"):
        step(ax, d, color="#b5b4ae", lw=1)
    t, s = step(ax, full, color=SERIES[0], lw=2)
    med = km_quantile(t, s, 0.5)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.annotate(f"median {med:.1f} s", (med, 0.5), xytext=(6, 6), textcoords="offset points", color=INK)
    ax.plot([], [], color="#b5b4ae", lw=1, label="single trial")
    ax.plot([], [], color=SERIES[0], lw=2, label=f"pooled ({full.trial.nunique()} trials, n={len(full)})")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lifetime t (s)")
    ax.set_ylabel("fraction of groups surviving past t")
    ax.set_title("Group survival (Kaplan-Meier)", loc="left", color=INK)

    # B: survival by peak size
    ax = axes[1]
    style(ax)
    for (name, lo, hi), c in zip(SIZE_CLASSES, SERIES):
        d = full[(full.max_size >= lo) & (full.max_size <= hi)]
        if len(d):
            t, s = step(ax, d, color=c, lw=2, label=f"peak size {name} (n={len(d)})")
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.legend(frameon=False, loc="lower left")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lifetime t (s)")
    ax.set_title("Survival by peak group size", loc="left", color=INK)

    # C: lifetime histogram, censored tracks stacked on top
    ax = axes[2]
    style(ax)
    bins = np.logspace(np.log10(xmin), np.log10(xmax), 30)
    done = full[~full.censored_right].lifetime_s
    cens = full[full.censored_right].lifetime_s
    ax.hist([done, cens], bins=bins, stacked=True, color=[SERIES[0], "#9ec5f4"],
            edgecolor=SURFACE, linewidth=1,
            label=[f"ended in run (n={len(done)})", f"still alive at end (n={len(cens)})"])
    ax.set_ylim(0, ax.get_ylim()[1] * 1.2)    # headroom for the legend
    ax.legend(frameon=False, loc="upper left")
    ax.set_xlabel("lifetime (s), log bins")
    ax.set_ylabel("number of groups")
    ax.set_title("Lifetime distribution", loc="left", color=INK)

    fig.suptitle(title, x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def compare_experiments(exps, out_prefix, title):
    """Compare two experiments on the three numbers that matter here:
    median lifetime of large groups (peak size >= LARGE), splits/min, merges/min.

    exps: [(name, pooled tracks, per-trial table)]. Writes <out_prefix>.csv/.png.
    The large-group median is the pooled Kaplan-Meier median (log-rank test);
    the per-trial median is also listed but is NaN wherever a trial's large
    groups mostly outlive the run. Rates are compared trial by trial: paired
    Wilcoxon when both experiments have the same trial names (same seeds),
    Mann-Whitney otherwise.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    (na, pa, ta), (nb, pb, tb) = exps
    larges = [p[~p.censored_left & (p.max_size >= LARGE)] for _, p, _ in exps]
    kms = [kaplan_meier(d.lifetime_s.to_numpy(), ~d.censored_right.to_numpy()) for d in larges]
    meds = [km_quantile(t, s, 0.5) for t, s in kms]

    def censored(d):
        return stats.CensoredData(uncensored=d.lifetime_s[~d.censored_right].to_numpy(),
                                  right=d.lifetime_s[d.censored_right].to_numpy())

    p_life = stats.logrank(censored(larges[0]), censored(larges[1])).pvalue
    paired = list(ta.trial) == list(tb.trial)

    def trial_test(col):
        x, y = ta[col].to_numpy(float), tb[col].to_numpy(float)
        if paired:
            ok = ~(np.isnan(x) | np.isnan(y))
            if ok.sum() < 2 or np.all(x[ok] == y[ok]):
                return np.nan, "Wilcoxon paired"
            return stats.wilcoxon(x[ok], y[ok]).pvalue, "Wilcoxon paired"
        return stats.mannwhitneyu(x[~np.isnan(x)], y[~np.isnan(y)]).pvalue, "Mann-Whitney"

    rows = [dict(metric=f"large-group (peak >= {LARGE}) KM median lifetime_s, pooled",
                 **{na: meds[0], nb: meds[1]}, test="log-rank", p=p_life),
            dict(metric="number of large groups, pooled", **{na: len(larges[0]), nb: len(larges[1])})]
    for col in ("large_km_median_s", "splits_per_min", "merges_per_min"):
        p, test = trial_test(col)
        rows.append(dict(metric=f"{col}, per trial", **{na: np.nanmean(ta[col]), nb: np.nanmean(tb[col])},
                         **{f"{na}_std": np.nanstd(ta[col], ddof=1), f"{nb}_std": np.nanstd(tb[col], ddof=1)},
                         test=test, p=p,
                         n_nan=f"{int(ta[col].isna().sum())}/{int(tb[col].isna().sum())}"))
    table = pd.DataFrame(rows)
    table.to_csv(f"{out_prefix}.csv", index=False)

    # ---- figure -----------------------------------------------------------
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [2, 1, 1]})
    tints = ["#9ec5f4", "#f5b89c"]

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    def fmt_p(p):
        return "p = n/a" if np.isnan(p) else f"p = {p:.3f}" if p >= 0.001 else "p < 0.001"

    # A: survival of large groups, pooled over trials -- the conventional KM
    # plot: linear time, curve carried on to the last follow-up, censored
    # groups ticked, Greenwood 95% band
    ax = axes[0]
    style(ax)
    xmax = max(d.lifetime_s.max() for d in larges) * 1.03
    ax.set_xlim(0, xmax)
    # the smaller median is labelled on the left of its line, the larger on the right
    order = np.argsort([m if not np.isnan(m) else np.inf for m in meds])
    sides = {order[0]: -1, order[1]: 1}
    for i, ((n, _, _), d, med, c, tint) in enumerate(zip(exps, larges, meds, SERIES, tints)):
        dur = d.lifetime_s.to_numpy()
        obs = ~d.censored_right.to_numpy()
        xs, ys, lo, hi, s, var_sum = [0.0], [1.0], [1.0], [1.0], 1.0, 0.0
        for u in np.unique(dur[obs]):
            n_risk = (dur >= u).sum()
            k = (obs & (dur == u)).sum()
            s *= 1 - k / n_risk
            if n_risk > k:
                var_sum += k / (n_risk * (n_risk - k))
            half_w = 1.96 * s * np.sqrt(var_sum)
            xs.append(u); ys.append(s); lo.append(max(s - half_w, 0)); hi.append(min(s + half_w, 1))
        xs.append(dur.max()); ys.append(ys[-1]); lo.append(lo[-1]); hi.append(hi[-1])
        ax.fill_between(xs, lo, hi, step="post", color=tint, alpha=0.25, lw=0)
        med_txt = f"{med:.0f} s" if not np.isnan(med) else f"> {dur.max():.0f} s"
        ax.step(xs, ys, where="post", color=c, lw=2, label=f"{n}: median {med_txt} (n={len(d)})")
        # censored groups: a tick where they left observation
        cx = dur[~obs]
        cy = np.array(ys)[np.searchsorted(xs, cx, side="right") - 1]
        ax.plot(cx, cy, "|", ms=8, mew=1.5, color=c)
        if not np.isnan(med):
            ax.plot([med, med], [0, 0.5], color=c, lw=1, ls="--")
            ax.annotate(f"{med:.0f} s", (med, 0.02), xytext=(4 * sides[i], 0), textcoords="offset points",
                        ha="left" if sides[i] > 0 else "right", color=INK, fontsize=9)
    ax.axhline(0.5, color=INK2, lw=0.8, ls=":")
    ax.plot([], [], "|", ms=8, mew=1.5, color=INK2, label="still alive at end of run")
    ax.fill_between([], [], [], color=INK2, alpha=0.15, lw=0, label="95% CI (Greenwood)")
    ax.legend(frameon=False, loc="upper right")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("lifetime t (s)")
    ax.set_ylabel("fraction of large groups surviving past t")
    ax.set_title(f"Large groups (peak >= {LARGE}): survival   log-rank {fmt_p(p_life)}",
                 loc="left", color=INK)

    # B, C: event rates, one dot per trial, paired trials joined, bar = mean
    for ax, col, label in ((axes[1], "splits_per_min", "Splits"), (axes[2], "merges_per_min", "Merges")):
        style(ax)
        p, test = trial_test(col)
        if paired:
            for ya, yb in zip(ta[col], tb[col]):
                ax.plot([0, 1], [ya, yb], color=GRID, lw=1, zorder=1)
        for k, ((n, _, t), c) in enumerate(zip(exps, SERIES)):
            ax.scatter(np.full(len(t), k), t[col], s=40, color=c, edgecolor=SURFACE, lw=1, zorder=2)
            m = t[col].mean()
            ax.plot([k - 0.2, k + 0.2], [m, m], color=INK, lw=2, zorder=3)
            ax.annotate(f"{m:.1f}", (k + 0.22, m), va="center", color=INK)
        ax.set_xticks([0, 1], [n for n, _, _ in exps])
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylim(0, max(ta[col].max(), tb[col].max()) * 1.15)
        ax.set_ylabel("events per minute")
        ax.set_title(f"{label} per minute   {fmt_p(p)}", loc="left", color=INK)

    fig.suptitle(title + ("   (trials paired by seed; bar = mean)" if paired else "   (bar = mean)"),
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return table


def _run(job):
    return job[0], process_trial(*job)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="a trial directory, or experiment directories holding trial_* folders "
                                             "(two experiments -> also a side-by-side comparison)")
    ap.add_argument("--max-dist", default="65", help="which groups_d<max_dist>.csv cache to read")
    ap.add_argument("--min-size", type=int, default=3)
    ap.add_argument("--win-s", type=float, default=1.0, help="hysteresis window in seconds (0 = off)")
    ap.add_argument("--fps", type=float, default=None, help="default: RENDER_FPS_HEADLESS from config_snapshot.json, else 60")
    ap.add_argument("--jobs", type=int, default=4, help="trials processed in parallel")
    ap.add_argument("--out-dir", default=None,
                    help="where the two-experiment comparison goes (default: their common parent directory)")
    args = ap.parse_args()

    tag = f"d{args.max_dist}_m{args.min_size}_w{args.win_s:g}"
    exps = [run_experiment(os.path.abspath(p), args, tag) for p in args.paths]
    exps = [e for e in exps if e is not None]
    if len(exps) == 2:
        out_dir = args.out_dir or os.path.commonpath([e[3] for e in exps])
        out = os.path.join(out_dir, f"compare_{exps[0][0]}_vs_{exps[1][0]}_lifetime_{tag}")
        table = compare_experiments([e[:3] for e in exps], out,
                                    f"{exps[0][0]}  vs  {exps[1][0]}  -  min size {args.min_size}, "
                                    f"hysteresis {args.win_s:g} s, d{args.max_dist}")
        print("\n==== comparison ====")
        print(table.round(4).to_string(index=False))
        print(f"\nwrote {out}.csv / .png")


def run_experiment(path, args, tag):
    """Track every trial of one experiment; returns (name, pooled tracks, per-trial table, dir)."""
    trial_dirs = sorted(d for d in glob.glob(os.path.join(path, "trial_*")) if os.path.isdir(d))
    exp_dir = path if trial_dirs else os.path.dirname(path)
    fps = args.fps
    if fps is None:
        cfg = os.path.join(exp_dir, "config_snapshot.json")
        fps = json.load(open(cfg, encoding="utf-8")).get("RENDER_FPS_HEADLESS", 60) if os.path.exists(cfg) else 60

    if not trial_dirs:
        process_trial(path, args.max_dist, args.min_size, args.win_s, fps)
        return None

    from multiprocessing import Pool
    jobs = [(d, args.max_dist, args.min_size, args.win_s, fps) for d in trial_dirs]
    with Pool(min(args.jobs, len(jobs))) as p:
        results = p.map(_run, jobs)

    exp = os.path.basename(exp_dir)
    prefix = os.path.join(exp_dir, exp)
    pool, rows = [], []
    for d, (tr, ev) in results:
        trial = os.path.basename(d)
        rows.append(trial_row(trial, tr, ev))
        pool.append(tr.assign(trial=trial))
    pool = pd.concat(pool, ignore_index=True)
    pool.insert(0, "trial", pool.pop("trial"))
    per_trial = pd.DataFrame(rows)
    pool.to_csv(f"{prefix}_tracks_{tag}.csv", index=False)
    per_trial.to_csv(f"{prefix}_lifetime_trials_{tag}.csv", index=False)
    plot_pooled(pool, per_trial, f"{prefix}_lifetime_{tag}.png",
                f"{exp}  -  min size {args.min_size}, hysteresis {args.win_s:g} s, d{args.max_dist}")

    print(f"\n==== {exp}: pooled over {len(results)} trials ====")
    summarize(pool, None, fps)
    print("\nper trial:")
    print(per_trial.round(2).to_string(index=False))
    print("\nacross trials (mean +- std):")
    for c in per_trial.columns[1:]:
        print(f"  {c:18s} {per_trial[c].mean():8.2f} +- {per_trial[c].std():.2f}")
    print(f"\nwrote {prefix}_tracks_{tag}.csv, _lifetime_trials_{tag}.csv, _lifetime_{tag}.png")
    return exp, pool, per_trial, exp_dir


if __name__ == "__main__":
    main()

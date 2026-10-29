"""
steady_state.py  -  find the stretches of a trial where grouping has settled,
and evaluate them instead of the value in one fixed window.

Why this exists
---------------
The fixed t>=400s window assumes the last 100 s is steady. It often is not:
across the gait sweep, 13 of 25 conditions still showed a residual trend inside
that window, and weakly driven gaits drift for the whole run. A fixed window
also throws away a trial that settled early and then got disturbed, and it
cannot tell "flat at 7" from "passing through 7 on the way up".

So instead: slide a window of the requested length over the whole trial, keep
the windows where the signal is flat, merge them into maximal steady segments,
and evaluate each segment. `steady_states` hands back every segment, so a
caller can count all the steady states a trial passed through; `steady_peak`
keeps only the highest one.

What "flat" means
-----------------
A window is steady when the least-squares trend across it changes the signal by
no more than `max_drift` of its own mean -- a fractional change over the window,
so the threshold means the same thing at every group size. Optionally a window
must also have a relative spread (sd/mean) below `max_cv`. On the raw samples
that spread is dominated by frame-to-frame fluctuation, so `cv_smooth_s` judges
it on a moving average instead: it then measures slow wandering of the level,
which a near-zero trend over a window can hide (a rise and a fall cancel).

Steadiness is judged on ONE reference signal and both quantities are then
measured over the same segments. Judging each separately would let the mean and
the largest come from different stretches of time, and the pair would no longer
describe the same state.
"""

import numpy as np


def moving_mean(y, w):
    """Centred moving average over w samples, shrinking at the ends (no padding)."""
    y = np.asarray(y, dtype=float)
    if w <= 1:
        return y
    c = np.concatenate(([0.0], np.cumsum(y)))
    i = np.arange(len(y))
    lo = np.maximum(0, i - w // 2)
    hi = np.minimum(len(y), i - w // 2 + w)
    return (c[hi] - c[lo]) / (hi - lo)


def steady_windows(t, y, window_s, max_drift=0.05, step_s=None, max_cv=None,
                   cv_smooth_s=None, max_dev=None):
    """
    -> boolean array, one entry per window start, True where the signal is flat.

    Also returns the window start indices so a caller can map back to time.

    cv_smooth_s, when set, judges max_cv on a moving average of that length
    instead of the raw samples, so the spread measures slow wandering of the
    level rather than frame-to-frame noise.

    max_dev caps the largest excursion of that same (smoothed) signal from the
    window mean, as a fraction of the mean. cv is an average over the window,
    so a five-second spike or a step in the last few seconds barely moves it;
    a cap on the worst point does not let either through.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(t) < 3:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)

    dt = float(np.median(np.diff(t)))
    w = max(3, int(round(window_s / dt)))
    if w > len(t):
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=int)
    step = max(1, int(round((step_s if step_s else dt) / dt)))
    yc = (moving_mean(y, int(round(cv_smooth_s / dt))) if cv_smooth_s
          else y)

    starts = np.arange(0, len(t) - w + 1, step)
    ok = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        ts, ys = t[s:s + w], y[s:s + w]
        m = ys.mean()
        if m <= 0:
            continue
        slope = np.polyfit(ts, ys, 1)[0]
        # fractional change the trend produces across the whole window
        if abs(slope) * (ts[-1] - ts[0]) / m > max_drift:
            continue
        if max_cv is not None or max_dev is not None:
            yw = yc[s:s + w]
            mw = yw.mean()
            if max_cv is not None and yw.std(ddof=1) / mw > max_cv:
                continue
            if max_dev is not None and np.abs(yw - mw).max() / mw > max_dev:
                continue
        ok[i] = True
    return ok, starts


def steady_segments(t, y, window_s, max_drift=0.05, step_s=None, max_cv=None,
                    min_len_s=None, cv_smooth_s=None, max_step=None,
                    max_dev=None):
    """
    -> list of (i0, i1) index spans, each a maximal run of overlapping steady
    windows. min_len_s drops segments shorter than that (default: one window).

    max_step, when set, also requires the windows merged into one segment to
    sit at the same level: a steady window joins the current segment only if
    its mean is within max_step (fractional) of the mean of the windows already
    in it. Without it, a plateau at 10 and one at 15 whose windows happen to
    touch come back as a single segment averaging 12.5 -- a level the system
    never held. Where a window at a new level overlaps the current segment,
    the segment is cut where that window starts.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    ok, starts = steady_windows(t, y, window_s, max_drift, step_s, max_cv,
                                cv_smooth_s=cv_smooth_s, max_dev=max_dev)
    if not ok.any():
        return []
    dt = float(np.median(np.diff(t)))
    w = max(3, int(round(window_s / dt)))
    min_len_s = window_s if min_len_s is None else min_len_s

    def keep(i0, i1):
        return t[i1] - t[i0] + dt >= min_len_s - 1e-9

    if max_step is not None:
        out, lo, hi, levels = [], None, None, []
        for s in starts[ok]:
            m = y[s:s + w].mean()
            ref = np.mean(levels) if levels else m
            if lo is not None and s <= hi + 1 and abs(m - ref) <= max_step * ref:
                hi = max(hi, s + w - 1)
                levels.append(m)
                continue
            if lo is not None:
                end = min(hi, s - 1)
                if keep(lo, end):
                    out.append((lo, end))
            lo, hi, levels = s, s + w - 1, [m]
        if keep(lo, hi):
            out.append((lo, hi))
        return out

    # A sample is steady if ANY steady window covers it; segments are the
    # maximal runs of such samples. Taking runs of steady *window starts*
    # instead would hand back overlapping spans -- two runs separated by one
    # bad window still overlap, because consecutive starts are far closer
    # together than the window is long -- and the segment durations would then
    # sum to more than the trial.
    covered = np.zeros(len(t), dtype=bool)
    for i, good in enumerate(ok):
        if good:
            covered[starts[i]:starts[i] + w] = True

    out, run = [], None
    for i, c in enumerate(covered):
        if c and run is None:
            run = i
        elif not c and run is not None:
            if t[i - 1] - t[run] + dt >= min_len_s - 1e-9:
                out.append((run, i - 1))
            run = None
    if run is not None and t[-1] - t[run] + dt >= min_len_s - 1e-9:
        out.append((run, len(t) - 1))
    return out


def steady_peak(t, largest, mean_sz, window_s, max_drift=0.05, step_s=None,
                max_cv=None, min_len_s=None, reference="largest",
                cv_smooth_s=None, max_step=None, max_dev=None):
    """
    Evaluate every steady segment and report the best one for each quantity.

    -> dict with, for both quantities, the maximum a steady segment reached,
       plus how much of the trial was steady at all.

    Within a segment each quantity is the **time average over that segment**,
    not the per-frame extreme. Taking the per-frame maximum and then the maximum
    over segments would be a max of a max: a single frame out of tens of
    thousands, with no averaging behind it, which moves with the noise rather
    than with the gait.

    `reference` picks the signal steadiness is judged on. Both quantities are
    then measured over those same segments, so the pair always describes one
    state of the system.

    When nothing is steady the values come back NaN and n_segments is 0; the
    caller decides whether that is a dropped cell or a reason to loosen
    max_drift. Silently falling back to the whole trial would hand back a number
    indistinguishable from a real steady one.
    """
    t = np.asarray(t, dtype=float)
    largest = np.asarray(largest, dtype=float)
    mean_sz = np.asarray(mean_sz, dtype=float)
    ref = largest if reference == "largest" else mean_sz

    segs = steady_segments(t, ref, window_s, max_drift, step_s, max_cv,
                           min_len_s, cv_smooth_s=cv_smooth_s,
                           max_step=max_step, max_dev=max_dev)
    if not segs:
        return {"largest_peak": np.nan, "mean_peak": np.nan,
                "n_segments": 0, "steady_frac": 0.0,
                "largest_peak_t": np.nan, "mean_peak_t": np.nan,
                "segments": []}

    span = t[-1] - t[0]
    rows = []
    for i0, i1 in segs:
        rows.append({
            "t0": float(t[i0]), "t1": float(t[i1]),
            "largest": float(largest[i0:i1 + 1].mean()),
            "mean": float(mean_sz[i0:i1 + 1].mean()),
        })
    kl = int(np.argmax([r["largest"] for r in rows]))
    km = int(np.argmax([r["mean"] for r in rows]))
    steady = sum(r["t1"] - r["t0"] for r in rows)
    return {
        "largest_peak": rows[kl]["largest"],
        "mean_peak": rows[km]["mean"],
        "largest_peak_t": 0.5 * (rows[kl]["t0"] + rows[kl]["t1"]),
        "mean_peak_t": 0.5 * (rows[km]["t0"] + rows[km]["t1"]),
        "n_segments": len(rows),
        "steady_frac": float(steady / span) if span > 0 else 0.0,
        "segments": rows,
    }


def steady_states(t, largest, mean_sz, window_s, max_drift=0.05, step_s=None,
                  max_cv=None, min_len_s=None, reference="largest",
                  cv_smooth_s=None, extra=None, max_step=None,
                  max_dev=None):
    """
    Every steady segment of a trial, none singled out.

    -> list of dicts, one per segment, in time order: t0, t1, dur (s), and the
       time average over the segment of largest and mean_sz, plus largest_cv
       (sd/mean of the raw largest inside the segment, a check on how flat the
       merged segment really is). `extra` is an optional {name: series} whose
       segment averages are added under the same names (nan-aware).

    A segment is a maximal run of samples each covered by at least one steady
    window, so every sample in it sits in a window that passed both tests.
    Judged on `reference` only, and every quantity is averaged over the same
    spans, as in steady_peak.
    """
    t = np.asarray(t, dtype=float)
    largest = np.asarray(largest, dtype=float)
    mean_sz = np.asarray(mean_sz, dtype=float)
    ref = largest if reference == "largest" else mean_sz
    segs = steady_segments(t, ref, window_s, max_drift, step_s, max_cv,
                           min_len_s, cv_smooth_s=cv_smooth_s,
                           max_step=max_step, max_dev=max_dev)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.0
    rows = []
    for i0, i1 in segs:
        L = largest[i0:i1 + 1]
        r = {"t0": float(t[i0]), "t1": float(t[i1]),
             # sample count x spacing, so back-to-back segments tile the trial
             "dur": float((i1 - i0 + 1) * dt),
             "largest": float(L.mean()),
             "mean": float(mean_sz[i0:i1 + 1].mean()),
             "largest_cv": float(L.std(ddof=1) / L.mean()) if len(L) > 1 else 0.0}
        for k, v in (extra or {}).items():
            r[k] = float(np.nanmean(np.asarray(v, dtype=float)[i0:i1 + 1]))
        rows.append(r)
    return rows

# Group-size statistics: how the numbers and error bars are produced

Applies to the scaling sets (`1005_scaling`, `1013_scale+catalyst`) and the gait
sweep (`1014_sweep`). Scripts: `scaling_stats*.py`, `sweep_stats.py`.

## What is measured

Grouping is `pos_all_grouping.compute_frame_groups` — Voronoi (Delaunay)
adjacency cut at `max_dist`, then connected components — the same function
`strategy.py` uses live and `batch_group_plots.py` uses offline.
`max_dist = 65 px`, singletons included, identical across every condition.
Body length is fixed at `L_s = 105 px` in all these runs, so 65 px is a constant
0.62 `L_s` and the threshold means the same thing in every condition.

Two quantities, both evaluated per frame and then time-averaged:

- **largest group size** — `max(size)` over the groups in that frame.
- **mean group size** — the size-weighted mean, `sum(s^2) / sum(s)`, i.e. "pick a
  robot at random, how big is its group". This is `trial_series`'s `mean_size`
  in `batch_group_plots.py`. Reported twice: including singletons (`sum` over
  all groups) and excluding them (`sum` over `s >= 2`). Singletons are 37-54% of
  all entries in these runs, so whether a lone robot counts as a "group of one"
  moves the number by about 1 — it is a definition, not a result, so both are
  given.

Note the plain per-group mean (`sum(s)/count`) is **not** used: it is dominated
by singletons and answers a different question.

## Steady-state window

`t >= 400 s` of a 500 s trial.

Not every condition is equilibrated by then. Weakly driven gaits relax slowly:
in `1014_sweep`, 15 deg / 0.5 Hz still drifts about +30% per 100 s inside the
window, while 15 deg / 4 Hz is flat. So each condition also gets a diagnostic:
fit a straight line to the largest-group series inside the window, express the
slope as percent of the mean per 100 s, and report it with its across-trial SEM.

    drift_sigma = |drift| / drift_sem
    status = lower_bound  if drift_sigma >= 2
             marginal     if 1 <= drift_sigma < 2
             steady       otherwise

A `lower_bound` condition is still rising when the trial ends, so its reported
value understates the true steady state. `drift_sigma` is kept as a number so a
different cutoff can be applied without recomputing.

## The sample is trials, not frames

Each trial is reduced to **one number** (its time-average over the window), and
the statistics are taken across the 10 trials. Frames are not independent
samples: measured on `1005_scaling` at N=100, the largest-group series has an
autocorrelation time of about 1610 frames (27 s), so a 59420-frame window
contains roughly 18 independent samples, not 59420. Treating frames as
independent gave an error bar 59x too small.

- `sd`  — spread of individual trials; use it to show how variable a single run is.
- `sem` — `sd / sqrt(10)`; uncertainty of the condition's mean. **This is the
  default error bar.**
- `ci95_halfwidth` — `sem * 2.262`, the t-distribution factor for df = 9. Not
  1.96: with n = 10 the normal approximation is wrong.

## Paired comparisons between conditions

In `1014_sweep` every condition uses the same ten initial conditions
(ICs 31-40, same order), so trial *k* of any two conditions starts from the same
layout. The IC-to-IC variation is therefore **common** to the conditions being
compared.

Comparing two conditions with their individual SEMs double-counts that shared
variation and is needlessly conservative. The correct test takes the paired
difference: for each IC compute `x_A(ic) - x_B(ic)`, then take the mean and SEM
of those 10 differences.

`_sweep_per_trial.csv` exists for exactly this — one row per (pattern, ic_idx) —
so any pair of gaits can be compared properly without recomputing.

The paired error is deliberately **not** put in the per-condition tables: it is a
property of a comparison, not of a condition, and would be misread as that
condition's uncertainty.

## What the error bars do not cover

All systematic, none of it in `sem`:

- the choice of `max_dist = 65 px`;
- the choice of window (`t >= 400 s`);
- residual non-equilibration, which is what `status` flags instead;
- for `1014_sweep` only: the commanded gait is not always the delivered one.
  The pilot measured the real joint angles and 23 of the 48 patterns were
  dropped because the arms reached less than 93% of the commanded amplitude.
  The 25 that remain are within 7%, with the worst at 6.7% — so even in the
  kept set the amplitude axis carries a few percent of systematic error.
  `gait_probe` records the delivered amplitude per condition if the exact
  figure is needed.

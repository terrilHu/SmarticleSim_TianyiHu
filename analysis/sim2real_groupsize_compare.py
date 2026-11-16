"""
sim2real_groupsize_compare.py
-----------------------------
One figure that overlays a simulation trial on the hardware experiment it
models: largest group (top) and mean group size per robot (bottom), sharing the
time axis.

    hardware   : mean across trials (bold) with a ±1 sd band
    simulation : one line per panel, in a second color

Inputs are the *_mean_timeseries.csv files that real_group_plots.py and
batch_group_plots.py already write, so nothing is regrouped here. The
*_trial_stats.csv next to each supplies the last-100 s numbers shown in the
panel titles.

Usage:
    python analysis/sim2real_groupsize_compare.py \
        --real D:/UMich-Smarticles/videos/1105_sim2real_exp/11_12/group_plots/11_12_mean_timeseries.csv \
        --sim  D:/UMich-Smarticles/videos/1105_sim2real_exp/11_12/sim_17-9/sim_a862-9/sim_a862-9_mean_timeseries.csv \
        --out  D:/UMich-Smarticles/videos/1105_sim2real_exp/11_12/group_plots/11_12_vs_sim_groupsize.png

Alignment is left out on purpose: it needs a second y-axis, and the point of
this figure is the group-size match.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
import numpy as np                                    # noqa: E402
import pandas as pd                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import batch_group_plots as bgp                       # noqa: E402

# Color = metric, as in batch_group_plots' groupsize figures; line style = source
# (hardware solid + ±1 sd band, simulation dashed).
C_METRIC = {"largest": bgp.C_LARGEST, "mean_size": bgp.C_MEANSZ}
SIM_LS = (0, (5, 2.5))
INK, MUTED, KEY = "#222222", "#6b6b6b", "#555555"


def stats_path(ts_path):
    return ts_path.replace("_mean_timeseries.csv", "_trial_stats.csv")


def steady(stats_csv, col):
    """-> (mean, sd, n) of a per-trial column, or None if the file is missing."""
    if not os.path.isfile(stats_csv):
        return None
    v = pd.read_csv(stats_csv)[col].to_numpy(float)
    return v.mean(), v.std(ddof=0), len(v)


def style_axis(ax):
    ax.grid(axis="y", alpha=0.25)
    ax.grid(axis="x", visible=False)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color("#bbbbbb")
    ax.tick_params(colors=MUTED)


def legend(fig, n_real, y=1.0):
    from matplotlib.legend_handler import HandlerTuple
    n_txt = f"n = {n_real} trials, " if n_real else ""
    # source keys in neutral gray: color belongs to the metric, not the source
    handles = [
        (plt.Line2D([], [], color=KEY, lw=2.0),
         plt.Rectangle((0, 0), 1, 1, color=KEY, alpha=0.18, lw=0)),
        plt.Line2D([], [], color=KEY, lw=2.0, ls=SIM_LS),
    ]
    fig.legend(handles,
               [f"Hardware ({n_txt}mean ± 1 sd)", "Simulation (1 trial)"],
               handler_map={tuple: HandlerTuple(ndivide=1)},
               loc="upper center", ncol=2, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, y), handlelength=3.0)


def draw_pair(ax, real, sim, wr, ws, key):
    """Hardware mean + band (solid) and simulation (dashed) for one metric. -> (m, y_sim)"""
    c = C_METRIC[key]
    m = bgp.roll_mean(real[f"{key}_mean"], wr)
    s = bgp.roll_mean(real[f"{key}_std"], wr)
    y_sim = bgp.roll_mean(sim[f"{key}_mean"], ws)
    ax.fill_between(real["time"], m - s, m + s, color=c, alpha=0.18, lw=0)
    ax.plot(real["time"], m, color=c, lw=2.0)
    ax.plot(sim["time"], y_sim, color=c, lw=2.0, ls=SIM_LS)
    return m, y_sim


def fig_single(real, sim, wr, ws, rs, ss, panels, t_end, args):
    """
    Both metrics on one axis. Color = metric (navy largest, orange mean size),
    line style = source. A label at the right edge names each metric and
    carries the last-N-s numbers.
    """
    fig, ax = plt.subplots(figsize=(10, 4.8))
    n_real = None
    for key, scol, name, _ in panels:
        m, y_sim = draw_pair(ax, real, sim, wr, ws, key)

        r, si = steady(rs, scol), steady(ss, scol)
        label = name.replace(" (per robot)", "\n(per robot)")
        if r and si:
            n_real = r[2]
            label += (f"\nlast {args.steady_s:g} s: HW {r[0]:.1f} ± {r[1]:.1f}"
                      f", sim {si[0]:.1f}")
        # anchor the label between the two curves' tail values
        tail = real["time"] >= t_end - args.steady_s
        y_lab = 0.5 * (m[tail.to_numpy()].mean() + y_sim[-1])
        ax.annotate(label, xy=(1.01, y_lab), xycoords=("axes fraction", "data"),
                    va="center", ha="left", fontsize=9.5, color=INK,
                    annotation_clip=False)

    ax.set_ylim(0, args.n_robots * 0.75)
    ax.set_xlim(0, t_end)
    ax.set_ylabel("group size (robots)", color=MUTED)
    ax.set_xlabel("time (s)", color=MUTED)
    style_axis(ax)
    legend(fig, n_real)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    bgp.save_fig(fig, args.out)
    plt.close(fig)
    print(f"-> {os.path.join(os.path.dirname(args.out), bgp.written_name(args.out))}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--real", required=True, help="hardware *_mean_timeseries.csv")
    ap.add_argument("--sim", required=True, help="simulation *_mean_timeseries.csv")
    ap.add_argument("--out", required=True, help="output .png path")
    ap.add_argument("--smooth-s", type=float, default=2.0)
    ap.add_argument("--steady-s", type=float, default=100.0,
                    help="tail length the *_steady stats were computed over (title only)")
    ap.add_argument("--n-robots", type=int, default=17)
    ap.add_argument("--formats", nargs="+", default=["png", "svg"],
                    choices=list(bgp.FIG_FORMATS))
    ap.add_argument("--font", default=None)
    ap.add_argument("--layout", choices=("stacked", "single"), default="stacked",
                    help="stacked: one panel per metric; single: both metrics on "
                         "one axis, labelled at the right edge")
    args = ap.parse_args()

    bgp.OUTPUT_FORMATS = list(args.formats)
    bgp.choose_font(args.font)

    real = pd.read_csv(args.real)
    sim = pd.read_csv(args.sim)
    t_end = min(real["time"].iloc[-1], sim["time"].iloc[-1])
    real = real[real["time"] <= t_end]
    sim = sim[sim["time"] <= t_end]
    wr = bgp.smooth_window(real["time"].to_numpy(), args.smooth_s)
    ws = bgp.smooth_window(sim["time"].to_numpy(), args.smooth_s)

    rs, ss = stats_path(args.real), stats_path(args.sim)
    panels = [
        ("largest", "largest_steady", "Largest group", "robots"),
        ("mean_size", "mean_size_steady", "Mean group size (per robot)", "robots"),
    ]

    if args.layout == "single":
        fig_single(real, sim, wr, ws, rs, ss, panels, t_end, args)
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    n_real = None
    for ax, (key, scol, name, unit) in zip(axes, panels):
        draw_pair(ax, real, sim, wr, ws, key)

        # panel title carries the steady-state comparison
        r, si = steady(rs, scol), steady(ss, scol)
        title = name
        if r and si:
            n_real = r[2]
            title += (f"   ·   last {args.steady_s:g} s:  hardware {r[0]:.1f} ± {r[1]:.1f}"
                      f",  simulation {si[0]:.1f}")
        ax.set_title(title, loc="left", fontsize=10.5, color=INK)
        ax.set_ylabel(f"size ({unit})", color=MUTED)
        style_axis(ax)

    axes[0].set_ylim(0, args.n_robots * 0.75)
    axes[1].set_ylim(0, args.n_robots * 0.5)
    axes[1].set_xlim(0, t_end)
    axes[1].set_xlabel("time (s)", color=MUTED)

    legend(fig, n_real)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    bgp.save_fig(fig, args.out)
    plt.close(fig)
    print(f"-> {os.path.join(os.path.dirname(args.out), bgp.written_name(args.out))}")


if __name__ == "__main__":
    main()

"""
catalyst_radial_maps.py
-----------------------
Only the large-group maps (fig2) and radial profiles (fig3) of catalyst_radial_figs.py,
for conditions that pool several experiment dirs and need not have the same number of
trials. Each panel title and legend entry carries its trial count; a condition with
fewer trials than the largest one is flagged in the title.

Usage:
    python catalyst_radial_maps.py <out> NAME=EXP[+EXP ...] [NAME=EXP ...] [--edge 0.75] [--t0 100] [--bare]

<out> holds cache/ (from catalyst_radial_build.py) and receives the figures. EXP is a
cache name (the part before __trial_XXXX.npz); '+' pools several. NAME picks the
condition's colour and marker in catalyst_radial_figs.COL.
"""
import argparse

import catalyst_radial_figs as crf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("conds", nargs="+", metavar="NAME=EXP[+EXP]")
    ap.add_argument("--edge", type=float, default=0.75)
    ap.add_argument("--t0", type=float, default=crf.T_EV, help="window start in s (default 100)")
    ap.add_argument("--bare", action="store_true", help="also write each figure as <name>_bare, with no text at all")
    a = ap.parse_args()
    exps = {k: v.split("+") for k, v in (x.split("=", 1) for x in a.conds)}
    for c in exps:
        if c not in crf.COL:
            raise SystemExit(f"{c}: no colour in catalyst_radial_figs.COL")
    n = {c: len(crf.cond_caches(a.out, e)) for c, e in exps.items()}
    full = max(n.values())
    for c in exps:
        crf.NAME[c] = f"{c}（n = {n[c]}）" if n[c] == full else f"{c}（仅 n = {n[c]}）"
        print(c, n[c], "trials:", " + ".join(exps[c]))
    crf.EDGE = a.edge
    crf.ALSO_BARE = a.bare
    crf.CONDS[:] = list(exps)
    crf.STRATS[:] = list(exps)[1:]
    res, ed = crf.spatial_reduction(exps, a.out, t0=a.t0)
    crf.fig_maps(res, ed, a.out)
    crf.fig_radial(res, a.out)


if __name__ == "__main__":
    main()

"""
catalyst_end_azimuth.py
-----------------------
x_end puts catalysts on the left and right of the swarm, which a purely radial analysis
averages away. Split the arena by direction from the swarm centroid, in world axes:

    caps    |cos(theta)| >= cos(45 deg)   left + right, where x_end's catalysts should sit
    flanks  the rest                      top + bottom

Writes azimuth_per_trial.csv (one row per condition x trial x window) and
azimuth_comparisons.csv into <results>. The sectors are equal in area and the baseline has
no reason to prefer either, so caps / flanks should be near 1 there and is the control.

Pre-stated expectation, from the first strategy pair: if catalysts locally reduce large
groups, x_end's caps should hold fewer large groups than its flanks and than baseline's.

major_end's axis follows the largest group, so these world-fixed sectors should show nothing
for it either; that makes it a second control rather than a test of major_end.

Usage:
    python catalyst_end_azimuth.py <out> <results> baseline=DIR major_end=DIR x_end=DIR
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(HERE)]

import catalyst_radial_stats as crs
from catalyst_radial_build import load_xy, role_matrix

C45 = np.cos(np.pi / 4)
EDGE = crs.rho_edge(crs.OUTER_K)
WINDOWS = {"t>=100": 100.0, "t>=400": 400.0}


def is_cap(dx, dy):
    return np.abs(dx) / np.maximum(np.hypot(dx, dy), 1e-9) >= C45


def main():
    out, results = sys.argv[1], sys.argv[2]
    dirs = dict(x.split("=", 1) for x in sys.argv[3:])
    rows = []
    for c, d in dirs.items():
        exp = os.path.basename(os.path.normpath(d))
        for tn in range(5):
            z = np.load(os.path.join(out, "cache", f"{exp}__trial_{tn:04d}.npz"), allow_pickle=True)
            fps, T, N, R = float(z["fps"]), int(z["T"]), int(z["N"]), float(z["R"])
            ft, fs, fxy, sw = z["frames_t"], z["frames_size"], z["frames_xy"], z["swarm_c"]
            big = fs >= 6
            p = fxy[big] - sw[ft[big]]
            rho = np.hypot(p[:, 0], p[:, 1]) / R
            cap = is_cap(p[:, 0], p[:, 1])
            weight = fs[big].astype(float)
            trial = f"trial_{tn:04d}"
            prefix = os.path.join(d, trial, trial)
            xy = load_xy(prefix + "_POS_ALL.csv", T, N)
            role = role_matrix(prefix + "_gait_log.csv", T, N, fps) if c != "baseline" else np.zeros((T, N), bool)
            step = int(fps)
            q = xy[::step] - sw[::step, None]
            rcat, ccap = np.hypot(q[..., 0], q[..., 1]) / R, is_cap(q[..., 0], q[..., 1])
            rl = role[::step]
            frames = np.arange(0, T, step)
            for w, t0 in WINDOWS.items():
                a = int(np.ceil(t0 * fps))
                keep = ft[big] >= a
                n_fr = T - a
                out_k = keep & (rho >= EDGE)
                m = dict(cond=c, trial=tn, window=w)
                m["n_large_outer_caps"] = (out_k & cap).sum() / n_fr
                m["n_large_outer_flanks"] = (out_k & ~cap).sum() / n_fr
                m["large_robots_outer_caps"] = weight[out_k & cap].sum() / n_fr / N
                m["large_robots_outer_flanks"] = weight[out_k & ~cap].sum() / n_fr / N
                m["n_large_all_caps"] = (keep & cap).sum() / n_fr
                m["n_large_all_flanks"] = (keep & ~cap).sum() / n_fr
                m["cap_share_outer"] = m["large_robots_outer_caps"] / (m["large_robots_outer_caps"] + m["large_robots_outer_flanks"])
                fk = frames >= a
                total = (rl[fk]).sum()
                m["catalysts_mean"] = total / fk.sum()
                m["catalyst_in_caps_frac"] = (rl[fk] & ccap[fk]).sum() / total if total else np.nan
                m["catalyst_in_outer_frac"] = (rl[fk] & (rcat[fk] >= EDGE)).sum() / total if total else np.nan
                rows.append(m)
            print(c, tn, flush=True)
    M = pd.DataFrame(rows)
    M.to_csv(os.path.join(results, "azimuth_per_trial.csv"), index=False)
    cmp_rows = []
    for w in WINDOWS:
        t = M[M.window == w]
        base = t[t.cond == "baseline"].sort_values("trial")
        for c in dirs:
            st = t[t.cond == c].sort_values("trial")
            # within the condition: caps vs flanks
            cf = st.assign(diff_caps=st.large_robots_outer_caps - st.large_robots_outer_flanks)
            cmp_rows.append(dict(window=w, cond=c, kind="caps-flanks", caps=st.large_robots_outer_caps.mean(), flanks=st.large_robots_outer_flanks.mean(),
                                 cap_share=st.cap_share_outer.mean(), caps_lower=int((cf.diff_caps < 0).sum())))
            if c != "baseline":
                for me in ("large_robots_outer_caps", "large_robots_outer_flanks", "cap_share_outer", "n_large_outer_caps", "n_large_outer_flanks"):
                    cmp_rows.append(dict(window=w, kind="vs baseline", **crs.compare(base, st, c, me)))
    pd.set_option("display.width", 220, "display.float_format", "{:.3f}".format, "display.max_rows", 200, "display.max_columns", 40)
    C = pd.DataFrame(cmp_rows)
    C.to_csv(os.path.join(results, "azimuth_comparisons.csv"), index=False)
    print(M.groupby(["window", "cond"])[["catalysts_mean", "catalyst_in_caps_frac", "catalyst_in_outer_frac", "cap_share_outer"]].mean().to_string())
    print(C.to_string())


if __name__ == "__main__":
    main()

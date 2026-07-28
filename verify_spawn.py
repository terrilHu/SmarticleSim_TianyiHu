"""
Spawn-layer checks.

1) REGRESSION — the legacy N<=BASE_N_REF spawner must produce byte-identical
   layouts to the original code.  (run_trial usually loads initial conditions
   from JSON, so the normal regression harness never exercises this path;
   GetSpawnPositions.py does.)

2) QUALITY — radial density profile of the layout actually produced at the
   configured N, so a large-N run can be sanity-checked against the reference.

Usage:
    python verify_spawn.py                 # check current config's N
    python verify_spawn.py <other_tree>    # also diff layouts against that tree
"""
import hashlib
import math
import os
import random
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pymunk

import config as cfg
from smarticle import add_ring
from spawn import (any_penetration, inside_ring,
                   spawn_smarticles, spawn_smarticles_auto)

NBIN = 5
CENTER = pymunk.Vec2d(cfg.W / 2, cfg.H / 2)
EFF_R = cfg.INNER_R * (math.cos(math.pi / max(3, int(cfg.RING_N_SIDES)))
                       if (cfg.RING_SHAPE or "circle").lower() == "polygon" else 1.0)


def _space():
    sp = pymunk.Space()
    add_ring(sp, CENTER, cfg.INNER_R, cfg.WALL_THICK, movable=False,
             shape=cfg.RING_SHAPE, n_sides=cfg.RING_N_SIDES)
    return sp


def layout_hash(smarts):
    h = hashlib.sha256()
    for s in smarts:
        for b in s.bodies():
            h.update(f"{b.position.x:.9f},{b.position.y:.9f},{b.angle:.9f};".encode())
    return h.hexdigest()[:16]


def radial(smarts):
    p = np.array([[s.main_body.position.x, s.main_body.position.y] for s in smarts])
    r = np.hypot(p[:, 0] - CENTER[0], p[:, 1] - CENTER[1])
    edges = cfg.INNER_R * np.sqrt(np.arange(NBIN + 1) / NBIN)   # equal-area
    return np.histogram(r, bins=edges)[0] / len(r), r.mean() / cfg.INNER_R


def main():
    seeds = (11, 12, 13, 14)

    # ---- 1) legacy spawner regression fingerprint -------------------------
    print("legacy spawn_smarticles() layout fingerprints "
          "(must match the original tree):")
    for sd in seeds[:2]:
        random.seed(sd); np.random.seed(sd)
        sp = _space()
        sm = spawn_smarticles(sp, CENTER, EFF_R, cfg.BASE_N_REF)
        print(f"  seed {sd}: {layout_hash(sm)}")

    # ---- 2) quality of the layout actually used at this N -----------------
    print(f"\nspawn_smarticles_auto() at N = {cfg.N_SMARTICLES}"
          f"   (SPAWN_LAYOUT = {cfg.SPAWN_LAYOUT})")
    print("  equal-area ring occupancy, inner -> outer; uniform = 0.20 each")
    F, M, pen, out = [], [], 0, 0
    for sd in seeds:
        random.seed(sd); np.random.seed(sd)
        sp = _space()
        sm = spawn_smarticles_auto(sp, CENTER, EFF_R, cfg.N_SMARTICLES)
        sp.step(1e-5)
        pen += sum(1 for s in sm if any_penetration(sp, s))
        out += sum(1 for s in sm if not inside_ring(s, CENTER, cfg.INNER_R))
        f, m = radial(sm)
        F.append(f); M.append(m)
    f = np.mean(F, axis=0)
    n_tot = len(seeds) * cfg.N_SMARTICLES
    print("  " + " ".join(f"{v:5.2f}" for v in f) +
          f"   mean r/R = {np.mean(M):.3f}")
    print(f"  overlapping (>3 px): {pen}/{n_tot}    outside ring: {out}/{n_tot}")
    print("  reference (N=17 legacy): mean r/R = 0.626, centre bin ~0.25")

    if f[0] < 0.13:
        print("  !! centre bin is depleted -- layout is wall-biased")
    if pen > 0.02 * n_tot:
        print("  !! significant residual overlap -- raise relax_steps")


if __name__ == "__main__":
    main()

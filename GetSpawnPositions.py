"""
getspawnpositions.py  ─  pack robots into the ring and save the layouts as an
init_conditions file.

Each entry is one trial's fully settled starting state (position, heading, both
arm angles, and the body geometry it was packed with), so simulation.py can
reload the exact layout instead of re-packing and getting a different one.

Two ways to run it
==================

**Plain** — uses config.py as it stands, writes init_conditions/<EXP_NAME>.json:

    python getspawnpositions.py

**From a condition file** — reads the same override files simulation.py and
experiments.py take (.py or .json, including a config_snapshot.json), and
generates *that condition's* init file:

    python getspawnpositions.py --config datafile/compare/_conditions/3_per_end_major_x10.py
    python getspawnpositions.py --config "datafile/compare/_conditions/*.py"
    python getspawnpositions.py --configs-from list.txt
    SMARTICLE_CONFIG=cond.py python getspawnpositions.py

The output path is taken from the condition's own **INIT_FILE** setting, which
is what makes this worth automating: a condition file already declares the init
file it expects, so running this is "produce whatever that experiment is asking
for" rather than "produce a file and then go and point the config at it". The
geometry that matters for packing — N_SMARTICLES, INNER_R, RING_SHAPE,
RING_N_SIDES, MAIN_LEN, SPAWN_LAYOUT ... — comes from the same override, so the
layout is packed into the arena that condition actually runs in.

One process, one config
-----------------------
config.py executes once on import and its consumers (spawn, smarticle) bind its
values by name at *their* import time, so a single process can only ever hold
one condition's parameters. Overrides therefore have to be in the environment
before the first `import config`, which is why the imports in this file live
inside functions rather than at the top, and why several --config files are run
as one subprocess each — the same arrangement experiments.py uses.

Several conditions usually share one init file (every 100-robot condition wants
init_conditions_200_p.json), so the work is de-duplicated by output path and an
existing file is left alone unless --force is given.

How many trials
---------------
The init file is a *pool* to draw from, not a run plan: N_TRIALS_GLOBAL in a
condition is how many trials that experiment runs, while the pool wants to be
much larger so different conditions can take different slices of it. Hence
--trials (default 200) is independent of the condition's N_TRIALS_GLOBAL, and
the only thing checked is that the pool is not smaller than what the condition
asks to run.
"""

import argparse
import glob
import json
import math
import os
import random
import subprocess
import sys
import time

DEFAULT_TRIALS = 200


# =============================================================================
# Deferred imports
# =============================================================================

def load_runtime():
    """
    Import config and everything that binds its values, and return them.

    Must not be called until SMARTICLE_CONFIG is final: `from config import X`
    copies the value, so anything imported earlier would keep the defaults and
    silently pack into the wrong arena.
    """
    import numpy as np
    import pygame
    import pymunk
    import pymunk.pygame_util

    import config as cfg
    from smarticle import Smarticle3Link, add_ring
    from spawn import spawn_smarticles_auto

    is_poly = (cfg.RING_SHAPE or "circle").lower() == "polygon"
    # Inside a polygon, pack to the inscribed radius (apothem) so robots start
    # inside the edges rather than inside the circumscribed circle; a circle
    # keeps INNER_R.
    eff_inner_r = (cfg.INNER_R * math.cos(math.pi / max(3, int(cfg.RING_N_SIDES)))
                   if is_poly else cfg.INNER_R)

    return dict(np=np, pygame=pygame, pymunk=pymunk, cfg=cfg,
                Smarticle3Link=Smarticle3Link, add_ring=add_ring,
                spawn_smarticles_auto=spawn_smarticles_auto,
                eff_inner_r=eff_inner_r)


# =============================================================================
# Visualization and debug saving
# =============================================================================

def save_layout_image(rt, space, filepath):
    """
    Render the current pymunk space (ring wall + all smarticles) to an
    in-memory Surface via debug_draw and save as an image file.

    Using space.debug_draw means the boundary is drawn from the actual
    Segment shapes in the space, so it is always correct regardless of
    whether the ring is a circle or a polygon.
    """
    pygame, pymunk, cfg = rt["pygame"], rt["pymunk"], rt["cfg"]
    if not pygame.get_init():
        pygame.init()

    surface = pygame.Surface((cfg.W, cfg.H))
    surface.fill((255, 255, 255))
    draw_options = pymunk.pygame_util.DrawOptions(surface)
    space.debug_draw(draw_options)
    pygame.image.save(surface, filepath)


# =============================================================================
# Progress bar
# =============================================================================

def print_progress_bar(iteration, total, start_time, bar_length=30):
    percent = iteration / total
    filled_len = int(bar_length * percent)
    bar = "█" * filled_len + "-" * (bar_length - filled_len)
    elapsed = time.time() - start_time
    eta = (elapsed / iteration * (total - iteration)) if iteration > 0 else 0
    eta_str = time.strftime("%M:%S", time.gmtime(eta))
    sys.stdout.write(f"\r[{bar}] {percent*100:5.1f}% ({iteration}/{total}) ETA: {eta_str}")
    sys.stdout.flush()
    if iteration == total:
        print()


# =============================================================================
# Extract state
# =============================================================================

def extract_smarticle_state(sm):
    return {
        "pos":   [float(sm.main_body.position.x), float(sm.main_body.position.y)],
        "angle": float(sm.main_body.angle),
        "thL":   float(sm.left_body.angle  - sm.main_body.angle),
        "thR":   float(sm.right_body.angle - sm.main_body.angle),
        # Persist physical geometry so the simulation reloads the EXACT body
        # this layout was packed with (essential for heterogeneous populations).
        "body": {
            "main_len":  int(sm.main_len),
            "main_w":    int(sm.main_w),
            "arm_len":   int(sm.arm_len),
            "arm_w":     int(sm.arm_w),
            "mass_main": float(sm.mass_main),
            "mass_arm":  float(sm.mass_arm),
        },
    }


def relax_system(space, steps=300, dt=1/240.0):
    old_damping = space.damping
    space.damping = 0.85
    for _ in range(steps):
        space.step(dt)
    space.damping = old_damping


# =============================================================================
# Main generation
# =============================================================================

def generate_all_initial_conditions(rt, save_path, n_trials, image_dir=None):
    """Pack n_trials layouts and write them to save_path. -> number saved."""
    np, pymunk, cfg = rt["np"], rt["pymunk"], rt["cfg"]
    all_trials = []
    start_time = time.time()

    if image_dir:
        os.makedirs(image_dir, exist_ok=True)

    for trial_id in range(n_trials):
        seed = cfg.TRIAL_SEED_BASE + trial_id
        random.seed(seed)
        np.random.seed(seed)

        # ── Build physics space ───────────────────────────
        space = pymunk.Space()
        center = pymunk.Vec2d(cfg.W / 2, cfg.H / 2)
        # Use the configured ring SHAPE, but keep it fixed during packing so the
        # wall does not drift while robots settle (mobility only matters at run
        # time, in simulation.py).
        rt["add_ring"](space, center, cfg.INNER_R, cfg.WALL_THICK,
                       movable=False, shape=cfg.RING_SHAPE,
                       n_sides=cfg.RING_N_SIDES)

        # ── Spawn smarticles ──────────────────────────────
        smarts = rt["spawn_smarticles_auto"](space, center, rt["eff_inner_r"],
                                             cfg.N_SMARTICLES)

        # Snapshot the layout whether or not spawning succeeded — a failed pack
        # is exactly the case worth looking at
        if image_dir:
            save_layout_image(rt, space,
                              os.path.join(image_dir, f"trial_{trial_id:04d}.jpg"))

        # ── Incomplete spawn: log and skip ───────────────────────────
        if len(smarts) != cfg.N_SMARTICLES:
            print(f"\n[DEBUG] Trial {trial_id + 1} failed: only placed "
                  f"{len(smarts)}/{cfg.N_SMARTICLES}."
                  f"{' Image saved.' if image_dir else ''}")
            for sm in smarts:
                sm.remove_from_space()
            continue

        # ── Save successful trial to JSON ──────────────────────────
        trial_data = {
            "trial_id": trial_id,
            "seed":     seed,
            "smarticles": [extract_smarticle_state(sm) for sm in smarts],
        }
        for sm in smarts:
            sm.remove_from_space()

        all_trials.append(trial_data)
        print_progress_bar(trial_id + 1, n_trials, start_time)

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(all_trials, f, indent=2)
    print(f"\n[spawn] Saved {len(all_trials)}/{n_trials} initial conditions "
          f"→ {save_path}")
    if image_dir:
        print(f"[spawn] Spawn images → {image_dir}")
    return len(all_trials)


# =============================================================================
# Condition files
# =============================================================================

def read_condition(path):
    """
    (settings, name) for one override file, via config.py's own loader so this
    reads exactly what simulation.py would.
    """
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from config import _load_override_file
    settings, _meta = _load_override_file(path)
    return settings, os.path.splitext(os.path.basename(path))[0]


def plan_for(settings, name, args):
    """
    What this condition needs: (out_path, n_trials, note). out_path is None when
    the condition does not name an INIT_FILE and none was given on the command
    line, which is not an error — such a condition spawns fresh every run.
    """
    out = args.out or settings.get("INIT_FILE")
    n_trials = args.trials
    note = ""
    want = settings.get("N_TRIALS_GLOBAL")
    if isinstance(want, int) and want > n_trials:
        note = (f"  [warn] {name} runs N_TRIALS_GLOBAL={want} trials but the "
                f"pool would hold only {n_trials}; raising --trials to {want}")
        n_trials = want
    return out, n_trials, note


def expand_configs(patterns):
    out, seen = [], set()
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or ([pat] if os.path.isfile(pat) else [])
        if not hits:
            print(f"[warn] no config file matches {pat!r}")
        for h in hits:
            p = os.path.normpath(h)
            if p not in seen:
                seen.add(p)
                out.append(p)
    return out


def run_in_subprocess(cfg_path, args):
    """
    One condition, one fresh interpreter -- config.py can only be configured
    once per process. Mirrors experiments.run_condition.
    """
    env = dict(os.environ)
    env["SMARTICLE_CONFIG"] = os.path.abspath(cfg_path)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    cmd = [sys.executable, os.path.abspath(__file__), "--trials", str(args.trials)]
    if args.force:
        cmd.append("--force")
    if not args.images:
        cmd.append("--no-images")
    if args.out:
        cmd += ["--out", args.out]
    # The child inherits the console and writes straight through, while this
    # process's stdout is block-buffered whenever it is piped -- without the
    # flush the plan shows up after the output of the runs it describes.
    sys.stdout.flush()
    return subprocess.run(cmd, env=env,
                          cwd=os.path.dirname(os.path.abspath(__file__))).returncode


# =============================================================================
# Entry point
# =============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="Pack robots into the ring and save the layouts as an "
                    "init_conditions file, optionally driven by condition files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python getspawnpositions.py
  python getspawnpositions.py --config datafile/compare/_conditions/3_per_end_major_x10.py
  python getspawnpositions.py --config "datafile/compare/_conditions/*.py" --list
  python getspawnpositions.py --config "datafile/compare/_conditions/*.py"

The output path comes from each condition's own INIT_FILE. Conditions sharing
one init file are generated once; an existing file is kept unless --force.
""")
    ap.add_argument("--config", nargs="+", default=None, metavar="FILE",
                    help="Condition file(s) to generate init conditions for (.py/.json, "
                         "wildcards allowed). Each runs in its own interpreter")
    ap.add_argument("--configs-from", default=None, metavar="LIST",
                    help="Read condition file paths from a text file, one per line "
                         "(# starts a comment)")
    ap.add_argument("--out", default=None, metavar="PATH",
                    help="Write to this path instead of the condition's INIT_FILE")
    ap.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                    help=f"Size of the layout pool to generate (default {DEFAULT_TRIALS}). "
                         f"Independent of a condition's N_TRIALS_GLOBAL, which is how many "
                         f"trials it runs; raised automatically if a condition needs more")
    ap.add_argument("--force", action="store_true",
                    help="Regenerate even if the target file already exists")
    ap.add_argument("--no-images", dest="images", action="store_false",
                    help="Skip the per-trial spawn snapshots (much faster)")
    ap.add_argument("--list", action="store_true",
                    help="Show what each condition would generate, then exit")
    args = ap.parse_args()

    patterns = list(args.config or [])
    if args.configs_from:
        with open(args.configs_from, encoding="utf-8-sig") as f:
            patterns += [ln.strip() for ln in f
                         if ln.strip() and not ln.lstrip().startswith("#")]

    # ── Driver mode: one subprocess per condition ─────────────────────────
    if patterns:
        if os.environ.get("SMARTICLE_CONFIG"):
            # Would be applied to the driver itself and then again to every
            # child, which is never what anyone means
            print("[warn] SMARTICLE_CONFIG is set but --config was given; "
                  "ignoring the environment variable")
            os.environ.pop("SMARTICLE_CONFIG", None)

        cfgs = expand_configs(patterns)
        if not cfgs:
            print("No condition files found.")
            return 1

        # Collapse to one job per output file: conditions at the same N share
        # an init file, and packing it once per condition would be wasted work
        # that also silently overwrites itself.
        jobs, by_out = [], {}
        for path in cfgs:
            settings, name = read_condition(path)
            out, n_trials, note = plan_for(settings, name, args)
            if note:
                print(note)
            if not out:
                print(f"  [skip] {name}: no INIT_FILE in the condition and no "
                      f"--out given (it spawns fresh at run time)")
                continue
            key = os.path.normpath(out)
            if key in by_out:
                by_out[key]["also"].append(name)
                by_out[key]["trials"] = max(by_out[key]["trials"], n_trials)
                continue
            by_out[key] = {"path": path, "name": name, "out": out,
                           "trials": n_trials,
                           "n": settings.get("N_SMARTICLES"), "also": []}
            jobs.append(by_out[key])

        print(f"{len(cfgs)} condition file(s) -> {len(jobs)} init file(s) to build")
        todo = []
        for j in jobs:
            exists = os.path.isfile(j["out"])
            shared = f"  (also used by {', '.join(j['also'])})" if j["also"] else ""
            state = "exists, skipping" if exists and not args.force else \
                    ("exists, regenerating" if exists else "to build")
            print(f"  {j['name']:34s} N={j['n']}  x{j['trials']}  "
                  f"-> {j['out']}   [{state}]{shared}")
            if not exists or args.force:
                todo.append(j)
        if args.list:
            return 0
        if not todo:
            print("Nothing to do (everything exists; use --force to rebuild).")
            return 0

        failed = 0
        for i, j in enumerate(todo, 1):
            print(f"\n[{i}/{len(todo)}] {j['name']} -> {j['out']}")
            sub = argparse.Namespace(**vars(args))
            sub.trials = j["trials"]
            sub.out = j["out"]
            if run_in_subprocess(j["path"], sub) != 0:
                print(f"  [FAIL] {j['name']}")
                failed += 1
        print(f"\nDone: {len(todo) - failed}/{len(todo)} init files built")
        return 1 if failed else 0

    # ── Worker mode: this process holds exactly one condition ─────────────
    rt = load_runtime()
    cfg = rt["cfg"]

    out = args.out or getattr(cfg, "INIT_FILE", None) \
        or os.path.join("init_conditions", f"init_conditions_{args.trials}.json")
    if os.path.isfile(out) and not args.force:
        print(f"[spawn] {out} already exists; use --force to regenerate")
        return 0

    image_dir = (os.path.join("spawn_images",
                              os.path.splitext(os.path.basename(out))[0])
                 if args.images else None)

    area_ring = math.pi * cfg.INNER_R * cfg.INNER_R
    area_sm = cfg.MAIN_LEN * cfg.MAIN_W + 2 * cfg.ARM_LEN * cfg.ARM_W
    print(f"[spawn] N={cfg.N_SMARTICLES}, INNER_R={cfg.INNER_R}, "
          f"ring={cfg.RING_SHAPE}"
          + (f"({cfg.RING_N_SIDES})" if str(cfg.RING_SHAPE).lower() == "polygon" else "")
          + f", packing ratio={cfg.N_SMARTICLES * area_sm / area_ring:.3f}")
    generate_all_initial_conditions(rt, out, args.trials, image_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())

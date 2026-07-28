import json
import random
import numpy as np
import pymunk
import time
import sys
import math
import pygame
import pymunk.pygame_util
import os
import math as _math

from config import (INNER_R, WALL_THICK, N_SMARTICLES, TRIAL_SEED_BASE,
                    MAIN_LEN, MAIN_W, ARM_LEN, ARM_W, W, H,
                    COMMAND_ARRAY, RING_SHAPE, RING_N_SIDES)
from smarticle import Smarticle3Link, add_ring
from spawn import (spawn_smarticles, spawn_smarticles_auto, spawn_smarticles_norelax,
                   any_penetration, inside_ring, build_from_initial_conditions)
from naming import generate_trial_name

# =========================
# Parameters
# =========================
N_TRIALS = 200

# _cmd0  = COMMAND_ARRAY[0];  _abs0 = abs(_cmd0)
# _z0    = _abs0 % 10;        _y0   = (_abs0 % 100 - _z0) // 10;  _x0 = _abs0 // 100
# _PTAB  = [_math.pi/4, _math.pi/2, _math.pi*3/4, _math.pi,
#           _math.pi*5/4, _math.pi*3/2, _math.pi*7/4, 2*_math.pi]
# _ATAB  = [_math.pi/12, _math.pi/6, _math.pi/4, _math.pi/3, _math.pi*5/12, _math.pi/2]
# _FTAB  = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]
# _ph0   = _PTAB[_x0 - 1] if 1 <= _x0 <= 8 else 0.0
# _om0   = (_FTAB[_z0 - 1] if 1 <= _z0 <= 9 else 0.5) * 2 * _math.pi
# _am0   = _math.degrees(_ATAB[_y0 - 1] if 1 <= _y0 <= 6 else _math.pi/4)
# _EXP_NAME = generate_trial_name(
#     N_SMARTICLES,
#     [(_ph0, _ph0)] * N_SMARTICLES,
#     omega=(_om0, _om0),
#     amplitude=(_am0, _am0),
# )
_EXP_NAME = "init_conditions_200_H"
SAVE_PATH = os.path.join("init_conditions", f"{_EXP_NAME}.json")
IMAGE_DIR = os.path.join("spawn_images",    _EXP_NAME)

# When the boundary is a polygon, pack robots inside its inscribed radius
# (apothem) so they start inside the n-gon edges; a circle keeps INNER_R.
_IS_POLYGON = (RING_SHAPE or "circle").lower() == "polygon"
EFF_INNER_R = (INNER_R * _math.cos(_math.pi / max(3, int(RING_N_SIDES)))
               if _IS_POLYGON else INNER_R)


# =========================
# Visualization and debug saving
# =========================
def save_layout_image(space, filepath):
    """
    Render the current pymunk space (ring wall + all smarticles) to an
    in-memory Surface via debug_draw and save as an image file.

    Using space.debug_draw means the boundary is drawn from the actual
    Segment shapes in the space, so it is always correct regardless of
    whether the ring is a circle or a polygon.
    """
    if not pygame.get_init():
        pygame.init()

    surface = pygame.Surface((W, H))
    surface.fill((255, 255, 255))
    draw_options = pymunk.pygame_util.DrawOptions(surface)
    space.debug_draw(draw_options)
    pygame.image.save(surface, filepath)


# =========================
# Progress bar
# =========================
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


# =========================
# Extract state
# =========================
def extract_smarticle_state(sm: Smarticle3Link):
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


# =========================
# Main function
# =========================
def generate_all_initial_conditions():
    all_trials = []
    start_time = time.time()

    # Create a dedicated image output folder
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for trial_id in range(N_TRIALS):
        seed = TRIAL_SEED_BASE + trial_id
        random.seed(seed)
        np.random.seed(seed)

        # ── Build physics space ───────────────────────────
        space = pymunk.Space()
        center = pymunk.Vec2d(W / 2, H / 2)
        # Use the configured ring SHAPE, but keep it fixed during packing so the
        # wall does not drift while robots settle (mobility only matters at run
        # time, in simulation.py).
        add_ring(space, center, INNER_R, WALL_THICK,
                 movable=False, shape=RING_SHAPE, n_sides=RING_N_SIDES)

        # ── Spawn smarticles ──────────────────────────────
        smarts = spawn_smarticles_auto(space, center, EFF_INNER_R, N_SMARTICLES)

        # ==============================================================
        # Save a snapshot image regardless of whether spawning succeeded or failed
        # ==============================================================
        # Generate filename: trial_XXXX.jpg (trial_id is 0-indexed)
        img_filename = f"trial_{trial_id:04d}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        save_layout_image(space, img_path)

        # ── Incomplete spawn: log and skip ───────────────────────────
        if len(smarts) != N_SMARTICLES:
            print(f"\n[DEBUG] Trial {trial_id + 1} failed: only placed {len(smarts)}/{N_SMARTICLES}. Image saved.")
            # Remove placed smarticles to free physics memory, skip to next seed
            for sm in smarts:
                sm.remove_from_space()
            continue

        # ── Post-relax validation (uncomment if needed) ─────────────
        # relax_system(space, steps=300)
        # valid = all(
        #    not any_penetration(space, sm) and inside_ring(sm, center, INNER_R)
        #    for sm in smarts
        # )
        # if not valid:
        #    print(f"\n[DEBUG] Trial {trial_id + 1}: post-relax validation failed, skipping")
        #    for sm in smarts:
        #        sm.remove_from_space()
        #    continue

        # ── Save successful trial to JSON ──────────────────────────
        trial_data = {
            "trial_id": trial_id,
            "seed":     seed,
            "smarticles": [extract_smarticle_state(sm) for sm in smarts],
        }
        for sm in smarts:
            sm.remove_from_space()

        all_trials.append(trial_data)
        print_progress_bar(trial_id + 1, N_TRIALS, start_time)

    # Write all trials to the final JSON file
    with open(SAVE_PATH, "w") as f:
        json.dump(all_trials, f, indent=2)
    print(f"\n[spawn] Experiment: {_EXP_NAME}")
    print(f"[spawn] Saved {len(all_trials)}/{N_TRIALS} initial conditions → {SAVE_PATH}")
    print(f"[spawn] Spawn images → {IMAGE_DIR}")


if __name__ == "__main__":
    area_circle = math.pi * INNER_R * INNER_R
    area_sm = MAIN_LEN * MAIN_W + 2 * ARM_LEN * ARM_W
    print("Packing ratio =", N_SMARTICLES * area_sm / area_circle)
    generate_all_initial_conditions()
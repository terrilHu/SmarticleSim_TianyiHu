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

from config import (INNER_R, WALL_THICK, N_SMARTICLES, TRIAL_SEED_BASE,
                    MAIN_LEN, MAIN_W, ARM_LEN, ARM_W, W, H,
                    INIT_PHASES, OMEGA_NOM1, OMEGA_NOM2, A_DEG_NOM1, A_DEG_NOM2)
from smarticle import Smarticle3Link, add_ring
from spawn import (spawn_smarticles, spawn_smarticles_norelax,
                   any_penetration, inside_ring, build_from_initial_conditions)
from naming import generate_trial_name

# =========================
# Parameters
# =========================
N_TRIALS = 200

_EXP_NAME = generate_trial_name(
    N_SMARTICLES,
    INIT_PHASES,
    omega=(OMEGA_NOM1, OMEGA_NOM2),
    amplitude=(A_DEG_NOM1, A_DEG_NOM2),
)
SAVE_PATH = os.path.join("init_conditions", f"{_EXP_NAME}.json")
IMAGE_DIR = os.path.join("spawn_images",    _EXP_NAME)


# =========================
# Visualization and debug saving (revised)
# =========================
def save_layout_image(smarts, center, inner_r, filepath):
    """
    Offline rendering without popup: draw robot positions onto an in-memory Surface and save as jpg
    """
    if not pygame.get_init():
        pygame.init()
        
    # Create a pure in-memory Surface without calling set_mode to avoid popup windows
    surface = pygame.Surface((W, H))
    surface.fill((255, 255, 255))
    
    # Draw the outer physical wall boundary (light gray)
    pygame.draw.circle(surface, (220, 220, 220), (int(center.x), int(center.y)), int(inner_r + WALL_THICK), int(WALL_THICK))
    # Draw the inner safety boundary (green solid line)
    pygame.draw.circle(surface, (0, 200, 0), (int(center.x), int(center.y)), int(inner_r), 2)

    # Draw all smarticles using the existing draw function
    for sm in smarts:
        draw_smarticle(surface, sm)
        
    # Save as a local image file
    pygame.image.save(surface, filepath)


def draw_smarticle(surface, sm):
    def draw_poly(body, shape, color):
        verts = shape.get_vertices()
        pts = [body.local_to_world(v) for v in verts]
        pts = [(int(p.x), int(p.y)) for p in pts]
        pygame.draw.polygon(surface, color, pts)

    draw_poly(sm.main_body, sm.main_shape, (0, 100, 255))
    draw_poly(sm.left_body,  sm.left_shape,  (255, 100, 100))
    draw_poly(sm.right_body, sm.right_shape, (100, 255, 100))


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
        add_ring(space, center, INNER_R, WALL_THICK)

        # ── Spawn smarticles ──────────────────────────────
        smarts = spawn_smarticles(space, center, INNER_R, N_SMARTICLES)

        # ==============================================================
        # Save a snapshot image regardless of whether spawning succeeded or failed
        # ==============================================================
        # Generate filename: trial_XXXX.jpg (trial_id is 0-indexed)
        img_filename = f"trial_{trial_id:04d}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        save_layout_image(smarts, center, INNER_R, img_path)

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
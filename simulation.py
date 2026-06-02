"""
simulation.py  ─  Core simulation loop (run_trial) and main entry point.

Run this file directly:
    python simulation.py
"""

import csv
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pygame
import pymunk
import pymunk.pygame_util

from config import (
    # experiment
    ALREADY_SPWANED, N_SMARTICLES, TRIAL_SEED_BASE, N_TRIALS_GLOBAL,
    INIT_PHASES, WARMUP_STEPS, RECORD_AFTER_WARMUP, MAX_RUNTIME,
    # geometry / physics
    W, H, INNER_R, WALL_THICK, WALL_SEGMENTS, WALL_FRICTION, WALL_ELASTICITY,
    RENDER_FPS_HEADLESS, RATE_LIM, V_MAX, W_MAX, ANG_DAMP, SPACE_DAMP, LIN_DAMP,
    # interaction model
    L, L_s, WC, R0, a0, a1, g0,
    # recording
    RECORD_VIDEO, RECORD_POLICY, RECORD_EVERY_K, RECORD_FIRST_N, VIDEO_STRIDE,
    # actuation
    OMEGA_NOM1, OMEGA_NOM2, A_DEG_NOM1, A_DEG_NOM2,
)
from smarticle import Smarticle3Link, add_ring, wrap_pi
from spawn import (
    build_from_initial_conditions, load_initial_conditions, spawn_smarticles,
)
from analysis import (
    VideoRecorder,
    actuationimpactCalculation, actutaionDetermination,
    calculate_macro_shape, calculate_orientational_order,
    compute_gr, dist_to_inner_wall, rotation_order_parameters,
    sigmoid, wall_strength, write_results_csv,
)

from pos_all_alignment import process_pos_all

from naming import generate_trial_name

# =============================================================================
# Utilities
# =============================================================================

def print_progress_bar(iteration, total, start_time, bar_length=30):
    percent    = iteration / total
    filled_len = int(bar_length * percent)
    bar        = "█" * filled_len + "-" * (bar_length - filled_len)
    elapsed    = time.time() - start_time
    eta        = (elapsed / iteration * (total - iteration)) if iteration > 0 else 0
    eta_str    = time.strftime("%M:%S", time.gmtime(eta))
    sys.stdout.write(f"\r[{bar}] {percent*100:5.1f}%  ({iteration}/{total})  ETA: {eta_str}")
    sys.stdout.flush()
    if iteration == total:
        print()


def should_record_trial(trial_id: int) -> bool:
    if not RECORD_VIDEO:
        return False
    pol      = (RECORD_POLICY or "").lower()
    do_mod   = ("mod"   in pol) and RECORD_EVERY_K and (trial_id % RECORD_EVERY_K == 0)
    do_first = ("first" in pol) and RECORD_FIRST_N and (trial_id <= RECORD_FIRST_N)
    if "mod" in pol and "first" in pol:
        return do_mod or do_first
    if "mod"   in pol: return do_mod
    if "first" in pol: return do_first
    return do_mod


# =============================================================================
# Core simulation
# =============================================================================

def run_trial(trial_id, seed, preview, video_dir, out_dir,
              actuations, ALL_INIT=None):
    """
    Run a single trial: physics loop, data collection, and file output.
    Returns a dict with summary scalars.
    """
    if not preview:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    # ── Physics space ─────────────────────────────────────────────────────────
    space                = pymunk.Space()
    space.damping        = SPACE_DAMP
    space.iterations     = 60
    space.collision_slop = 0.06

    center = pymunk.Vec2d(W / 2, H / 2)
    add_ring(space, center, inner_r=INNER_R, wall_thick=WALL_THICK,
             segments=WALL_SEGMENTS, friction=WALL_FRICTION, elasticity=WALL_ELASTICITY)

    # ── Spawn / load ──────────────────────────────────────────────────────────
    if ALREADY_SPWANED:
        smarticles = build_from_initial_conditions(space, ALL_INIT[trial_id])
    else:
        smarticles = spawn_smarticles(space, center, INNER_R, N_SMARTICLES)

    initial_positions = [s.main_body.position for s in smarticles]

    # ── Per-robot warmup and phase ────────────────────────────────────────────
    for i, s in enumerate(smarticles):
        ph1, ph2       = INIT_PHASES[i]
        s.phase1       = ph1
        s.phase2       = ph2
        s.warmup_steps = WARMUP_STEPS
        s._warmup_step = 0

    # ── Initial impulses ──────────────────────────────────────────────────────
    for s in smarticles:
        s.main_body.apply_impulse_at_local_point(
            pymunk.Vec2d(random.uniform(-500, 500), random.uniform(-500, 500)), (0, 0))

    # ── Rendering setup ───────────────────────────────────────────────────────
    orig_surf        = pygame.Surface((W, H))
    font             = pygame.font.Font(None, 16)
    draw_options_orig = pymunk.pygame_util.DrawOptions(orig_surf)

    def _blit_id_labels(dst):
        for idx, ss in enumerate(smarticles):
            x, y  = int(ss.main_body.position.x), int(ss.main_body.position.y)
            label = str(idx + 1)
            dst.blit(font.render(label, True, (255, 255, 255)), (x + 1, y + 1))
            dst.blit(font.render(label, True, (0,   0,   0  )), (x,     y    ))

    record_this   = should_record_trial(trial_id)
    recorder_orig = None
    if record_this:
        video_path    = os.path.join(video_dir, f"trial_{trial_id:04d}.mp4")
        recorder_orig = VideoRecorder(video_path, W, H, 30)

    # ── Data containers ───────────────────────────────────────────────────────
    ORDs, ORDs_abs, ORD_diffs, ORD_diffs_abs = [], [], [], []
    Mdist, GRs, ROT_ORDS, IMPACTS, POS_ALL   = [], [], [], [], []
    time_hist = []

    jam_time   = None
    t          = 0.0
    dt         = 1.0 / 180.0
    frame_i    = 0
    # MAX_RUNTIME = 45.0
    W_WALL     = 2.0
    D0_WALL    = 1.0 * L_s
    K_WALL     = 1.0

    n = N_SMARTICLES
    dx_matrix  = np.zeros((n, n))
    dy_matrix  = np.zeros((n, n))
    dPsi_matrix= np.zeros((n, n))
    dxy_matrix = np.zeros((n, n))
    Aij_matrix = np.zeros((n, n))
    A_wall     = np.zeros((n,), dtype=np.float32)

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        t_before        = t
        steps_per_frame = max(1, int(round((1.0 / RENDER_FPS_HEADLESS) / dt)))

        for _ in range(steps_per_frame):
            for s in smarticles:
                s.step_control(t)
                s.motor_L.rate = max(-RATE_LIM, min(RATE_LIM, s.motor_L.rate))
                s.motor_R.rate = max(-RATE_LIM, min(RATE_LIM, s.motor_R.rate))
                if s.main_body.velocity.length > V_MAX:
                    s.main_body.velocity = s.main_body.velocity.normalized() * V_MAX
                if abs(s.main_body.angular_velocity) > W_MAX:
                    s.main_body.angular_velocity = max(-W_MAX, min(W_MAX, s.main_body.angular_velocity))

            for s in smarticles:
                s.main_body.velocity         *= LIN_DAMP
                s.main_body.angular_velocity *= ANG_DAMP   

                # for b in s.bodies():
                #     b.angular_velocity *= ANG_DAMP

            space.step(dt)
            t += dt

        # ── Data collection ───────────────────────────────────────────────────
        warmup_active = RECORD_AFTER_WARMUP and any(
            s._warmup_step < s.warmup_steps for s in smarticles)

        if not warmup_active:
            positions = [s.main_body.position for s in smarticles]
            psis      = [s.main_body.angle    for s in smarticles]
            rot_ord   = rotation_order_parameters(psis, positions, center)
            mdists    = []

            for i in range(n):
                dgap, r      = dist_to_inner_wall(positions[i], center, INNER_R)
                A_wall[i], _ = wall_strength(dgap, L_s, k=K_WALL, d0=D0_WALL)
                dx0 = positions[i].x - initial_positions[i].x
                dy0 = positions[i].y - initial_positions[i].y
                mdists.append(dx0 ** 2 + dy0 ** 2)

                for j in range(i + 1, n):
                    dx   = positions[i].x - positions[j].x
                    dy   = positions[i].y - positions[j].y
                    dPsi = wrap_pi(psis[j] - psis[i])
                    acts = actuationimpactCalculation(
                        smarticles[i].omega1, smarticles[i].omega2,
                        smarticles[i].A1,     smarticles[i].A2)
                    inter = actutaionDetermination(dx, dy, dPsi, acts[0], acts[1])

                    dx_matrix[i][j]  =  dx;  dx_matrix[j][i]  = -dx
                    dy_matrix[i][j]  =  dy;  dy_matrix[j][i]  = -dy
                    dPsi_matrix[i][j] =  dPsi; dPsi_matrix[j][i] = -dPsi
                    dxy = math.sqrt(dx * dx + dy * dy)
                    dxy_matrix[i][j] = dxy_matrix[j][i] = dxy
                    Aij_matrix[i][j] = Aij_matrix[j][i] = (
                        L ** 2 / (dx ** 2 + dy ** 2) * (
                            (a0 + a1 * inter) * (np.sin(dPsi / 2) ** 2 + g0)
                            + WC * sigmoid((R0 - dxy) / L_s)))

            IMPACT = Aij_matrix.sum(axis=1) / (n - 1) + W_WALL * A_wall
            IMPACTS.append(IMPACT)
            _, gr = compute_gr(dxy_matrix)
            GRs.append(gr)
            # POS_ALL.append(np.array([(p.x, p.y) for p in positions], dtype=np.float32))
            POS_ALL.append(np.array([(p.x, p.y, wrap_pi(s.main_body.angle)) 
                          for p, s in zip(positions, smarticles)], dtype=np.float32))

            psis_arr = np.array([wrap_pi(s.main_body.angle) for s in smarticles])
            sys_ord, sys_ord_abs = calculate_orientational_order(psis_arr)
            sys_diff = sys_diff_abs = 0.0
            for i in range(n):
                d, d_abs        = calculate_orientational_order(dPsi_matrix[i])
                sys_diff       += d
                sys_diff_abs   += d_abs
            sys_diff     /= (n - 1)
            sys_diff_abs /= (n - 1)

            Mdist.append(np.sqrt(np.mean(mdists)))
            ORDs.append(sys_ord);       ORDs_abs.append(sys_ord_abs)
            ORD_diffs.append(sys_diff); ORD_diffs_abs.append(sys_diff_abs)
            ROT_ORDS.append(rot_ord)
            time_hist.append(t)

            if record_this:
                orig_surf.fill((255, 255, 255))
                space.debug_draw(draw_options_orig)
                _blit_id_labels(orig_surf)
                if frame_i % max(1, VIDEO_STRIDE) == 0:
                    recorder_orig.add_frame_from_surface(orig_surf)
            frame_i += 1

        if t >= MAX_RUNTIME and jam_time is None:
            jam_time = MAX_RUNTIME
            running  = False

    # ── Post-processing ───────────────────────────────────────────────────────
    final_positions          = np.array([s.main_body.position for s in smarticles])
    final_rg, final_kappa, final_area = calculate_macro_shape(final_positions)

    if record_this:
        recorder_orig.close()
    pygame.quit()

    def col(lst):
        a = np.reshape(np.array(lst), (-1, 1))
        return a

    GRs_ORDs_Mdist = np.hstack([
        col(ORDs_abs), col(ORDs), col(ORD_diffs), col(ORD_diffs_abs),
        col(ROT_ORDS),  col(Mdist),
    ])
    GRs_arr = np.array(GRs)

    # ── Save outputs ──────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    prefix = os.path.join(out_dir, f"trial_{trial_id:04d}")

    if len(GRs_arr) > 0:
        np.savez_compressed(prefix + "_GRs.npz", gr=GRs_arr)

    if len(GRs_ORDs_Mdist) > 0:
        header = ",".join([f"t{i}" for i in range(GRs_ORDs_Mdist.shape[1])])
        np.savetxt(prefix + "_GRs_ORDs_Mdist.csv",
                   GRs_ORDs_Mdist, delimiter=",", header=header, comments="")

    if IMPACTS:
        impacts_arr = np.array(IMPACTS)
        with open(prefix + "_impacts.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["time"] + [f"s{i+1}" for i in range(len(smarticles))])
            nrows = len(IMPACTS)
            times = time_hist[-nrows:] if len(time_hist) >= nrows else time_hist
            for ti, row in zip(times, impacts_arr):
                w.writerow([f"{ti:.6f}"] + [f"{float(x):.6f}" for x in row])

    if POS_ALL:
        with open(prefix + "_POS_ALL.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["Step", "Agent_ID", "X", "Y", "Theta"])
            for step_idx, frame in enumerate(POS_ALL):
                for agent_id, row in enumerate(frame):
                    w.writerow([step_idx, agent_id + 1,
                                f"{row[0]:.4f}", f"{row[1]:.4f}", f"{row[2]:.4f}"])
        # ── 计算 alignment 时间序列 ──────────────────────────────
        try:
            align_df = process_pos_all(prefix + "_POS_ALL.csv", max_dist=150.0)
            align_df.to_csv(prefix + "_alignment.csv", index=False)
        except Exception as e:
            print(f"[WARN] alignment 计算失败: {e}")

    return {"trial_id": trial_id,
            "final_rg": final_rg,
            "final_kappa": final_kappa,
            "final_area":  final_area}


# =============================================================================
# Config snapshot
# =============================================================================

def save_config_snapshot(out_dir: str):
    """将本次实验的关键 config 参数保存为 JSON 到 out_dir 根目录。"""
    import config as _cfg
    import json

    # 只记录可序列化的标量/列表参数，跳过模块级中间变量（以 _ 开头）
    snapshot = {}
    for k, v in vars(_cfg).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)):
            snapshot[k] = v
        elif isinstance(v, list):
            # INIT_PHASES 等列表，元素为 tuple/float
            try:
                snapshot[k] = [list(x) if isinstance(x, tuple) else x for x in v]
            except Exception:
                pass

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[main] Config snapshot saved: {path}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    WORKERS   = 1                          # >1 for parallel trials
    PREVIEW   = False
    #INIT_FILE = "init_conditions/init_conditions_200.json"
    INIT_FILE = os.path.join("init_conditions", EXP_NAME + ".json")
    N_TRIALS  = N_TRIALS_GLOBAL            # 0 = auto-read from init file

    # ── 用 naming 生成本次实验的统一名称 ──────────────────────────────────────
    EXP_NAME    = generate_trial_name(
        N_SMARTICLES,
        INIT_PHASES,
        omega=(OMEGA_NOM1, OMEGA_NOM2),
        amplitude=(A_DEG_NOM1, A_DEG_NOM2),
    )
    print(f"[main] Experiment name: {EXP_NAME}")

    OUT_DIR     = os.path.join("datafile",  EXP_NAME)
    VIDEO_DIR   = os.path.join("videos",    EXP_NAME)
    RESULTS_CSV = os.path.join("datafile",  EXP_NAME + "_summary.csv")

    actuations = actuationimpactCalculation(
        OMEGA_NOM1 / (2 * math.pi), OMEGA_NOM2 / (2 * math.pi),
        math.radians(A_DEG_NOM1),   math.radians(A_DEG_NOM2),
    )

    if ALREADY_SPWANED:
        ALL_INIT = load_initial_conditions(INIT_FILE)
        if not N_TRIALS:
            N_TRIALS = len(ALL_INIT)
        print(f"[main] Loaded {N_TRIALS} initial conditions from '{INIT_FILE}'.")
    else:
        ALL_INIT = None
        N_TRIALS = N_TRIALS or 1

    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # ── 保存本次实验的 config 快照 ────────────────────────────────────────────
    save_config_snapshot(OUT_DIR)

    results    = []
    start_time = time.time()
    print(f"[main] Starting {N_TRIALS} trial(s) with {WORKERS} worker(s).")

    try:
        if WORKERS == 1:
            for tid in range(N_TRIALS):
                seed    = TRIAL_SEED_BASE + tid
                out_dir = os.path.join(OUT_DIR, f"trial_{tid:04d}")
                print(f"\n[main] Trial {tid + 1}/{N_TRIALS}")
                try:
                    res = run_trial(trial_id=tid, seed=seed, preview=PREVIEW,
                                    video_dir=VIDEO_DIR, out_dir=out_dir,
                                    actuations=actuations, ALL_INIT=ALL_INIT)
                    results.append(res)
                except Exception as e:
                    print(f"[main] Trial {tid} failed: {e}")
                print_progress_bar(tid + 1, N_TRIALS, start_time)
        else:
            futures = []
            with ProcessPoolExecutor(max_workers=WORKERS) as ex:
                for tid in range(N_TRIALS):
                    seed    = TRIAL_SEED_BASE + tid
                    out_dir = os.path.join(OUT_DIR, f"trial_{tid:04d}")
                    futures.append(ex.submit(
                        run_trial, trial_id=tid, seed=seed, preview=PREVIEW,
                        video_dir=VIDEO_DIR, out_dir=out_dir,
                        actuations=actuations, ALL_INIT=ALL_INIT))
                completed = 0
                for fut in as_completed(futures):
                    try:
                        results.append(fut.result())
                    except Exception as e:
                        print(f"\n[main] Trial failed: {e}")
                    completed += 1
                    print_progress_bar(completed, N_TRIALS, start_time)

    except KeyboardInterrupt:
        print("\n[main] Interrupted — saving partial results...")

    print(f"\n[main] Completed {len(results)}/{N_TRIALS} trial(s).")
    if results:
        write_results_csv(RESULTS_CSV, results)
        print(f"[main] Saved summary CSV: {RESULTS_CSV}")


if __name__ == "__main__":
    main()

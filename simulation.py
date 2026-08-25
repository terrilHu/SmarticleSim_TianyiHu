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

# The runtime-gait switches below are read through `cfg.` at call time, never
# imported by value: `from config import X` binds once at import, which is why
# sweep.py has to patch both config.COMMAND_ARRAY and simulation.COMMAND_ARRAY.
# Reading them off the module (as bodies.py does) makes them patchable.
import config as cfg
from config import (
    # experiment
    ALREADY_SPWANED, N_SMARTICLES, TRIAL_SEED_BASE, N_TRIALS_GLOBAL,
    COMMAND_ARRAY, WARMUP_STEPS, RECORD_AFTER_WARMUP, MAX_RUNTIME, INIT_SELECTION,
    INIT_INDICES,
    # geometry / physics
    W, H, INNER_R, WALL_THICK, WALL_SEGMENTS, WALL_FRICTION, WALL_ELASTICITY,
    RING_MOVABLE, RING_SHAPE, RING_N_SIDES, RING_COLOR_BANDS,
    RENDER_FPS_HEADLESS, RATE_LIM, V_MAX, W_MAX, ANG_DAMP, SPACE_DAMP, LIN_DAMP, ENABLE_HETEROGENEOUS_BODIES,
    RING_DAMPING_ENABLED,
    # interaction model
    L, L_s, WC, R0, a0, a1, g0,
    # recording
    RECORD_VIDEO, RECORD_POLICY, RECORD_EVERY_K, RECORD_FIRST_N, VIDEO_STRIDE,
    PARALLEL_WORKERS,
    # actuation
    OMEGA_NOM1, OMEGA_NOM2, A_DEG_NOM1, A_DEG_NOM2,
)
from smarticle import Smarticle3Link, add_ring, wrap_pi
from spawn import (
    build_from_initial_conditions, load_initial_conditions,
    spawn_smarticles, spawn_smarticles_auto,
)
from analysis import (
    VideoRecorder,
    actuationimpactCalculation, actutaionDetermination,
    build_pair_matrices,
    calculate_macro_shape, calculate_orientational_order,
    compute_gr, dist_to_inner_wall, rotation_order_parameters,
    sigmoid, wall_strength, write_results_csv,
)

from pos_all_alignment import process_pos_all

from naming import generate_trial_name

import gait
from gait import GaitController, resolve_callback

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


def _phase_to_color(phase: float):
    """
    Map an initial phase (radians, any real) to an RGB color via the HSV wheel.

    Hue = phase / 2pi, so the 8 discrete phase-table values land on 8 evenly
    spaced, easily distinguished hues, while a random phase (command digit
    x == 0) gets a unique color reflecting its actual sampled value. Saturation
    and value are kept high so markers stand out against the white background.
    """
    import colorsys
    hue = (phase % (2 * math.pi)) / (2 * math.pi)
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(r * 255), int(g * 255), int(b * 255))


# =============================================================================
# Core simulation
# =============================================================================

def _show_init_and_get_commands(smarticles, center, trial_id):
    """
    Display the initial layout in a pygame window and prompt the user
    to enter a COMMAND_ARRAY via the terminal.
    Returns the parsed list of ints, or None if the user skips.
    """
    # Render at half the simulation resolution
    SCALE_WIN = 0.55
    WIN_W = int(W * SCALE_WIN)
    WIN_H = int(H * SCALE_WIN)

    def sp(v):
        """Scale a simulation coordinate to window coordinate."""
        return int(v * SCALE_WIN)

    os.environ.pop("SDL_VIDEODRIVER", None)
    pygame.init()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption(
        f"Trial {trial_id} — initial layout  (type commands in terminal then press Enter)")

    screen.fill((255, 255, 255))
    cx, cy = sp(center.x), sp(center.y)
    if (RING_SHAPE or "circle").lower() == "polygon":
        _n   = max(3, int(RING_N_SIDES))
        _pts = [(sp(center.x + INNER_R * math.cos(2 * math.pi * k / _n)),
                 sp(center.y + INNER_R * math.sin(2 * math.pi * k / _n)))
                for k in range(_n)]
        pygame.draw.polygon(screen, (220, 220, 220), _pts, max(1, sp(WALL_THICK)))
        pygame.draw.polygon(screen, (0, 200, 0), _pts, 2)
    else:
        pygame.draw.circle(screen, (220, 220, 220), (cx, cy),
                           sp(INNER_R + WALL_THICK), sp(WALL_THICK))
        pygame.draw.circle(screen, (0, 200, 0), (cx, cy), sp(INNER_R), 2)

    font_label = pygame.font.Font(None, 15)
    for idx, sm in enumerate(smarticles):
        for body, shape, color in [
            (sm.main_body,  sm.main_shape,  (0,   100, 255)),
            (sm.left_body,  sm.left_shape,  (255, 100, 100)),
            (sm.right_body, sm.right_shape, (100, 255, 100)),
        ]:
            verts = shape.get_vertices()
            pts   = [(sp(p.x), sp(p.y))
                     for p in [body.local_to_world(v) for v in verts]]
            pygame.draw.polygon(screen, color, pts)
        x, y = sp(sm.main_body.position.x), sp(sm.main_body.position.y)
        screen.blit(font_label.render(str(idx), True, (0, 0, 0)), (x + 2, y + 2))

    info_font = pygame.font.Font(None, 18)
    for li, line in enumerate([
        f"Trial {trial_id} — {len(smarticles)} robots",
        "Enter COMMAND_ARRAY in terminal, then press Enter.",
        "Format: 566, -226, 051, ...  (one int per robot)",
        "Leave blank to use preset COMMAND_ARRAY.",
    ]):
        screen.blit(info_font.render(line, True, (30, 30, 30)), (10, 10 + li * 20))

    pygame.display.flip()

    # Close the window before blocking on input to avoid OS "not responding"
    pygame.display.iconify()   # minimise first so it disappears cleanly
    pygame.event.pump()        # flush pending events
    pygame.quit()

    print(f"\n[Trial {trial_id}] Initial layout shown. "
          f"Enter {len(smarticles)} comma-separated commands "
          f"(or blank to use preset):")
    raw = input("  COMMAND_ARRAY> ").strip()

    if not raw:
        return None
    try:
        cmds = [int(x.strip()) for x in raw.split(",")]
        if len(cmds) != len(smarticles):
            print(f"[WARN] Expected {len(smarticles)} values, got {len(cmds)}. Using preset.")
            return None
        return cmds
    except ValueError as e:
        print(f"[WARN] Parse error ({e}). Using preset.")
        return None


def run_trial(trial_id, seed, preview, video_dir, out_dir,
              actuations, ALL_INIT=None, use_preset=True,
              init_idx=None):
    """
    Run a single trial: physics loop, data collection, and file output.
    When use_preset=False, displays the initial layout in a window and
    prompts the user for a per-robot COMMAND_ARRAY before starting.
    Returns a dict with summary scalars.
    """
    if not preview and use_preset:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    pygame.init()

    # ── Physics space ─────────────────────────────────────────────────────────
    space                = pymunk.Space()
    space.damping        = SPACE_DAMP
    space.iterations     = 60
    space.collision_slop = 0.06

    center = pymunk.Vec2d(W / 2, H / 2)
    ring = add_ring(space, center, inner_r=INNER_R, wall_thick=WALL_THICK,
                    segments=WALL_SEGMENTS, friction=WALL_FRICTION,
                    elasticity=WALL_ELASTICITY,
                    movable=RING_MOVABLE, shape=RING_SHAPE, n_sides=RING_N_SIDES,
                    color_bands=RING_COLOR_BANDS)

    # Effective interior radius used for live spawning: a polygon's edges sit
    # inward of the circumscribed circle, so spawn within the inscribed radius
    # (apothem) to keep robots inside the n-gon.  Circle → unchanged INNER_R.
    eff_inner_r = INNER_R
    if (RING_SHAPE or "circle").lower() == "polygon":
        eff_inner_r = INNER_R * math.cos(math.pi / max(3, int(RING_N_SIDES)))

    # ── Spawn / load ──────────────────────────────────────────────────────────
    # Track the IC index actually used, if any, so it can be reported in the
    # summary CSV regardless of how it was chosen (explicit / random /
    # sequential / the trial_id fallback below).  None for a fresh spawn.
    used_ic_idx = None
    if ALREADY_SPWANED:
        _idx = init_idx if init_idx is not None else trial_id
        used_ic_idx = _idx
        #smarticles = build_from_initial_conditions(space, ALL_INIT[trial_id])
        smarticles = build_from_initial_conditions(space, ALL_INIT[_idx])
    else:
        smarticles = spawn_smarticles_auto(space, center, eff_inner_r, N_SMARTICLES)

    # ── Robot ids ─────────────────────────────────────────────────────────────
    # The list index has always been the de-facto robot id (video labels, CSV
    # Agent_ID, COMMAND_ARRAY, BODY_ASSIGNMENT all key off it); make it explicit
    # so the runtime gait controller can be addressed by number.
    for _i, _s in enumerate(smarticles):
        _s.id = _i

    initial_positions = [s.main_body.position for s in smarticles]

    # ── Interactive command input (use_preset=False) ──────────────────────────
    active_commands = list(COMMAND_ARRAY)
    if not use_preset:
        user_cmds = _show_init_and_get_commands(smarticles, center, trial_id)
        if user_cmds is not None:
            active_commands = user_cmds
        if not preview:
            os.environ["SDL_VIDEODRIVER"] = "dummy"
            pygame.init()

    # ── Decode active_commands and apply per-robot parameters ─────────────────
    # GaitController's constructor performs exactly the decode loop that used
    # to be inlined here (same order, same bodies.gait_for_smarticle override,
    # same single random.random() draw per x == 0 robot), and additionally
    # keeps a GaitState per robot id so the motion pattern can be tracked -- and
    # changed -- individually during the run.
    #
    # Marker color encodes the INITIAL PHASE. If x is 1-8 the phase is one of
    # the 8 discrete table values; if x == 0 the phase is random, so we color by
    # the actual sampled ph1. Mapping phase in [0, 2pi) to an HSV hue
    # distinguishes the 8 discrete bins and also handles random phases.
    runtime_gait_on = bool(getattr(cfg, "ENABLE_RUNTIME_GAIT_CONTROL", False))
    gait_callback   = None
    if runtime_gait_on:
        try:
            gait_callback = resolve_callback(
                getattr(cfg, "RUNTIME_GAIT_CONTROLLER", None))
        except Exception as e:
            print(f"[Trial {trial_id}] Could not load RUNTIME_GAIT_CONTROLLER "
                  f"({e!r}); runtime gait control disabled for this trial.")
        if gait_callback is None:
            runtime_gait_on = False

    gait_ctl = GaitController(
        smarticles, active_commands,
        warmup_steps=WARMUP_STEPS,
        phase_color_fn=_phase_to_color,
        switch_mode=getattr(cfg, "RUNTIME_GAIT_SWITCH_MODE", "zero_phase"),
        callback=gait_callback,
        verbose=runtime_gait_on and bool(
            getattr(cfg, "RUNTIME_GAIT_VERBOSE", False)),
        center=(float(center.x), float(center.y)),
        ctx={"trial_id": trial_id, "n": len(smarticles),
             "center": (float(center.x), float(center.y)),
             "inner_r": INNER_R, "dt": 1.0 / 180.0,
             "max_runtime": MAX_RUNTIME},
    )
    gait_poll_every = max(1, int(getattr(cfg, "RUNTIME_GAIT_POLL_EVERY", 1)))

    # ── Initial impulses ──────────────────────────────────────────────────────
    for s in smarticles:
        s.main_body.apply_impulse_at_local_point(
            pymunk.Vec2d(random.uniform(-500, 500), random.uniform(-500, 500)), (0, 0))

    # ── Rendering setup ───────────────────────────────────────────────────────
    orig_surf        = pygame.Surface((W, H))
    font             = pygame.font.Font(None, 16)
    draw_options_orig = pymunk.pygame_util.DrawOptions(orig_surf)

    # Glyph surfaces depend only on the label text, so render each label once
    # and reuse it for every frame: identical pixels, but ~2n fewer
    # font.render() calls per frame (this dominates at n = 100).
    _label_cache = {}

    def _label_surfaces(label):
        pair = _label_cache.get(label)
        if pair is None:
            pair = (font.render(label, True, (0, 0, 0)),
                    font.render(label, True, (255, 255, 255)))
            _label_cache[label] = pair
        return pair

    def _blit_id_labels(dst):
        for ss in smarticles:
            cx, cy = int(ss.main_body.position.x), int(ss.main_body.position.y)
            surf_w, surf_s = _label_surfaces(str(ss.id + 1))
            tw, th = surf_w.get_size()
            # blit at top-left = center - half-size so the text is centred on (cx, cy)
            lx, ly = cx - tw // 2, cy - th // 2
            dst.blit(surf_s, (lx + 1, ly + 1))   # white shadow
            dst.blit(surf_w, (lx,     ly    ))    # black text

    def _blit_phase_markers(dst, radius=7):
        """Draw a filled circle on each robot colored by its initial phase."""
        for ss in smarticles:
            x, y  = int(ss.main_body.position.x), int(ss.main_body.position.y)
            color = getattr(ss, "phase_marker_color", (180, 180, 180))
            pygame.draw.circle(dst, (0, 0, 0), (x, y), radius + 1)   # outline
            pygame.draw.circle(dst, color,     (x, y), radius)

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
    poll_i     = 0     # frame boundaries seen, for RUNTIME_GAIT_POLL_EVERY
    # MAX_RUNTIME = 45.0
    W_WALL     = 2.0
    D0_WALL    = 1.0 * L_s
    K_WALL     = 1.0

    # Use the real population size instead of the config constant so a trial
    # stays consistent if an IC file holds a different count.  Identical
    # whenever len(smarticles) == N_SMARTICLES, which was always the case.
    n = len(smarticles)
    A_wall     = np.zeros((n,), dtype=np.float32)

    # Scratch arrays reused across frames (positions / angles / displacements).
    px_buf   = np.empty(n); py_buf  = np.empty(n); psi_buf = np.empty(n)
    ipx      = np.array([p.x for p in initial_positions], dtype=float)
    ipy      = np.array([p.y for p in initial_positions], dtype=float)
    cx_c, cy_c = float(center.x), float(center.y)

    # ── Per-robot geometry for the (possibly heterogeneous) coupling model ────
    # In a homogeneous run every entry equals the global L / L_s / S, so the
    # interaction formulas below reduce EXACTLY to the original ones.
    Ls_i   = np.array([s.L_s      for s in smarticles], dtype=float)  # half-span
    Lw_i   = np.array([s.main_w   for s in smarticles], dtype=float)  # body width (== L)
    Smain_i= np.array([s.main_len for s in smarticles], dtype=float)  # body length (== S)

    # actuationimpactCalculation() depends ONLY on robot i's own actuation
    # parameters, which are fixed for the whole trial.  The original evaluated
    # it once per (i, j) pair per frame -> O(n^2 * frames) scipy integrations.
    # Hoisting it here is exact: identical values, evaluated n times, once.
    _acts   = [actuationimpactCalculation(s.omega1, s.omega2, s.A1, s.A2)
               for s in smarticles]
    acts0_i = np.array([a[0] for a in _acts], dtype=float)
    acts1_i = np.array([a[1] for a in _acts], dtype=float)
    # Pre-bind the pymunk handles touched every physics step.  Attribute
    # chains like s.main_body.velocity are resolved 180 x n times per simulated
    # second; caching the objects removes that lookup without changing any
    # arithmetic.
    ctrl_plan = [(s, s.main_body, s.motor_L, s.motor_R) for s in smarticles]

    # Dynamic bodies making up a MOVABLE ring, so it can get the same per-step
    # floor damping as every robot main_body (see RING_DAMPING_ENABLED).  Empty
    # for a fixed ring (RING_MOVABLE=False) or when the toggle is off, so the
    # loop below costs nothing in that case -- which covers every run in this
    # codebase's regression baseline (RING_MOVABLE defaults to True, but this
    # flag can be set False to reproduce the old, undamped-ring trajectories).
    ring_damp_bodies = []
    if RING_DAMPING_ENABLED and RING_MOVABLE:
        if ring.body is not None:            # movable circle: one rigid body
            ring_damp_bodies = [ring.body]
        elif ring.bodies:                    # movable polygon: one body/edge
            ring_damp_bodies = list(ring.bodies)

    _vstride = max(1, VIDEO_STRIDE)

    # ── Main loop ─────────────────────────────────────────────────────────────
    running = True
    while running:
        t_before        = t
        steps_per_frame = max(1, int(round((1.0 / RENDER_FPS_HEADLESS) / dt)))

        for _ in range(steps_per_frame):
            # The clamp pass and the damping pass are fused into one loop.
            # This is exactly equivalent: neither pass reads another robot's
            # state, step_control() never reads velocities, and damping never
            # touches angles -- so per-robot the operation order is unchanged.
            # Fusing removes one full Python pass and roughly halves the number
            # of (expensive) pymunk property round-trips per robot per step.
            for s, mb, mL, mR in ctrl_plan:
                s.step_control(t)

                mL.rate = RATE_LIM * math.tanh(mL.rate / RATE_LIM)
                mR.rate = RATE_LIM * math.tanh(mR.rate / RATE_LIM)

                v = mb.velocity
                if v.length > V_MAX:
                    v = v.normalized() * V_MAX
                mb.velocity = v * LIN_DAMP

                av = mb.angular_velocity
                if abs(av) > W_MAX:
                    av = max(-W_MAX, min(W_MAX, av))
                mb.angular_velocity = av * ANG_DAMP

            # Ring floor damping -- see ring_damp_bodies above.  A plain Python
            # loop (not vectorised): pymunk Body objects aren't batchable, and
            # the list is at most RING_N_SIDES long (35-85 across N=17..100),
            # far smaller than the n-robot loop already run every step above,
            # so this is not a meaningful addition to per-step cost.
            for rb in ring_damp_bodies:
                rb.velocity         *= LIN_DAMP
                rb.angular_velocity *= ANG_DAMP

            # Queued motion-pattern changes take effect at the robot's own zero
            # phase, which is resolved to a physics step -- not a frame -- so
            # the commanded angle stays continuous.  With nothing queued (the
            # default, and always when runtime control is off) this costs one
            # attribute read per step.
            if gait_ctl.has_pending:
                for rid in gait_ctl.step(t):
                    _s = smarticles[rid]
                    # acts0_i / acts1_i cache actuationimpactCalculation() for
                    # the whole trial; a new gait invalidates that robot's row.
                    acts0_i[rid], acts1_i[rid] = actuationimpactCalculation(
                        _s.omega1, _s.omega2, _s.A1, _s.A2)

            space.step(dt)
            t += dt

        # ── Data collection ───────────────────────────────────────────────────
        warmup_active = RECORD_AFTER_WARMUP and any(
            s._warmup_step < s.warmup_steps for s in smarticles)

        if not warmup_active:
            positions = [s.main_body.position for s in smarticles]
            psis      = [s.main_body.angle    for s in smarticles]
            rot_ord   = rotation_order_parameters(psis, positions, center)

            for _i, _p in enumerate(positions):
                px_buf[_i] = _p.x
                py_buf[_i] = _p.y
            psi_buf[:] = psis

            # ── Wall coupling (was the outer half of the pair loop) ─────────
            # float_power (not **2) to match pymunk's Vec2d.length, which is
            # math.sqrt(x**2 + y**2) on Python floats -- see analysis.py.
            _rw   = np.sqrt(np.float_power(px_buf - cx_c, 2.0)
                            + np.float_power(py_buf - cy_c, 2.0))
            _dgap = INNER_R - _rw
            A_wall[:] = K_WALL * sigmoid((Ls_i - _dgap) / (Ls_i + 1e-9))

            _dx0   = px_buf - ipx
            _dy0   = py_buf - ipy
            mdists = np.float_power(_dx0, 2.0) + np.float_power(_dy0, 2.0)

            # ── Pair coupling: same formula, evaluated in NumPy ─────────────
            (dx_matrix, dy_matrix, dPsi_matrix,
             dxy_matrix, Aij_matrix) = build_pair_matrices(
                px_buf, py_buf, psi_buf, acts0_i, acts1_i,
                Ls_i, Lw_i, Smain_i, a0, a1, g0, WC)

            IMPACT = Aij_matrix.sum(axis=1) / (n - 1) + W_WALL * A_wall
            IMPACTS.append(IMPACT)
            _, gr = compute_gr(dxy_matrix)
            GRs.append(gr)
            # Same values as the previous list-of-tuples comprehension
            # (wrap_pi is elementwise, float32 cast is unchanged).
            _wpsi = wrap_pi(psi_buf)
            POS_ALL.append(
                np.stack((px_buf, py_buf, _wpsi), axis=1).astype(np.float32))

            psis_arr = _wpsi
            sys_ord, sys_ord_abs = calculate_orientational_order(psis_arr)
            # calculate_orientational_order(row) == (|mean(exp(1j*row))|,
            # |mean(exp(2j*row))|).  numpy's elementwise exp is chunk-invariant
            # (verified: exp(1j*M)[i] is bit-identical to exp(1j*M[i]) ), so the
            # two exponentials are hoisted to a single whole-matrix call.  The
            # per-row np.mean and the sequential accumulation below are kept
            # exactly as they were -- reductions are NOT chunk-invariant.
            _E1 = np.exp(1j * dPsi_matrix)
            _E2 = np.exp(1j * 2.0 * dPsi_matrix)
            sys_diff = sys_diff_abs = 0.0
            for i in range(n):
                sys_diff       += np.abs(np.mean(_E1[i]))
                sys_diff_abs   += np.abs(np.mean(_E2[i]))
            sys_diff     /= (n - 1)
            sys_diff_abs /= (n - 1)

            Mdist.append(np.sqrt(np.mean(mdists)))
            ORDs.append(sys_ord);       ORDs_abs.append(sys_ord_abs)
            ORD_diffs.append(sys_diff); ORD_diffs_abs.append(sys_diff_abs)
            ROT_ORDS.append(rot_ord)
            time_hist.append(t)

            # Only frames that are actually encoded need to be drawn.  The
            # previous code rendered every frame and then threw away
            # (VIDEO_STRIDE - 1) of them; the resulting video is identical.
            if record_this and frame_i % _vstride == 0:
                orig_surf.fill((255, 255, 255))
                space.debug_draw(draw_options_orig)
                _blit_phase_markers(orig_surf)
                _blit_id_labels(orig_surf)
                recorder_orig.add_frame_from_surface(orig_surf)
            frame_i += 1

        # ── Runtime gait control ──────────────────────────────────────────────
        # Polled at the frame boundary, after data collection, so the callback
        # sees exactly the state that was just recorded.  Whatever it asks for
        # is queued and applied by gait_ctl.step() above.  Counted on its own
        # (not frame_i, which is frozen during warm-up) so RUNTIME_GAIT_POLL_EVERY
        # means the same thing before and after the ramp.
        if runtime_gait_on:
            if poll_i % gait_poll_every == 0:
                gait_ctl.poll(t)
            poll_i += 1

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
            # Byte-for-byte the same as the csv.writer version (no field needs
            # quoting and csv's default line terminator is CRLF), but written
            # in large blocks instead of one call per row.
            nrows = len(IMPACTS)
            times = time_hist[-nrows:] if len(time_hist) >= nrows else time_hist
            f.write("time," + ",".join(f"s{i+1}" for i in range(len(smarticles)))
                    + "\r\n")
            _chunk = []
            for ti, row in zip(times, impacts_arr):
                _chunk.append(f"{ti:.6f}," +
                              ",".join(f"{float(x):.6f}" for x in row) + "\r\n")
                if len(_chunk) >= 512:
                    f.write("".join(_chunk)); _chunk.clear()
            if _chunk:
                f.write("".join(_chunk))

    if POS_ALL:
        with open(prefix + "_POS_ALL.csv", "w", newline="") as f:
            # Identical bytes to the csv.writer version (CRLF terminator, no
            # quoting needed), but this file has frames x n rows -- 360k rows at
            # n = 100 -- so per-row csv.writer overhead was significant.
            f.write("Step,Agent_ID,X,Y,Theta\r\n")
            _chunk = []
            for step_idx, frame in enumerate(POS_ALL):
                for agent_id, row in enumerate(frame):
                    _chunk.append(f"{step_idx},{agent_id + 1},"
                                  f"{row[0]:.4f},{row[1]:.4f},{row[2]:.4f}\r\n")
                if len(_chunk) >= 20000:
                    f.write("".join(_chunk)); _chunk.clear()
            if _chunk:
                f.write("".join(_chunk))
        # ── Compute alignment time series ───────────────────────
        try:
            align_df = process_pos_all(prefix + "_POS_ALL.csv", max_dist=150.0)
            align_df.to_csv(prefix + "_alignment.csv", index=False)
        except Exception as e:
            print(f"[WARN] alignment computation failed: {e}")

    return {"trial_id": trial_id,
            "ic_idx": used_ic_idx if used_ic_idx is not None else "",
            "final_rg": final_rg,
            "final_kappa": final_kappa,
            "final_area":  final_area}


# =============================================================================
# Config snapshot
# =============================================================================

def save_config_snapshot(out_dir: str):
    """Save key config parameters for this experiment as JSON in the out_dir root."""
    import config as _cfg
    import json

    # Only record serializable scalar/list parameters; skip module-level intermediates (prefixed with _)
    snapshot = {}
    for k, v in vars(_cfg).items():
        if k.startswith("_"):
            continue
        if isinstance(v, (int, float, bool, str)):
            snapshot[k] = v
        elif isinstance(v, list):
            # Lists like COMMAND_ARRAY whose elements are int
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
    WORKERS   = PARALLEL_WORKERS           # >1 for parallel trials (config.py)
    PREVIEW   = False
    INIT_FILE = "init_conditions/init_conditions_200.json"
    #INIT_FILE = os.path.join("init_conditions", EXP_NAME + ".json")
    N_TRIALS  = N_TRIALS_GLOBAL            # 0 = auto-read from init file
    USE_PRESET = True

    # ── Generate unified experiment name using naming module ────────────────────
    _cmd0  = COMMAND_ARRAY[0];  _abs0 = abs(_cmd0)
    _z0    = _abs0 % 10;        _y0   = (_abs0 % 100 - _z0) // 10;  _x0 = _abs0 // 100
    _PTAB, _ATAB, _FTAB = gait.PHASE_TABLE, gait.AMPLI_TABLE, gait.FREQ_TABLE
    _ph0   = _PTAB[_x0 - 1] if 1 <= _x0 <= 8 else 0.0
    _om0   = (_FTAB[_z0 - 1] if 1 <= _z0 <= 9 else 0.5) * 2 * math.pi
    _am0   = math.degrees(_ATAB[_y0 - 1] if 1 <= _y0 <= 6 else math.pi/4)
    EXP_NAME    = generate_trial_name(
        N_SMARTICLES,
        [(_ph0, _ph0)] * N_SMARTICLES,
        omega=(_om0, _om0),
        amplitude=(_am0, _am0),
    )
    print(f"[main] Experiment name: {EXP_NAME}")

    OUT_DIR     = os.path.join("datafile",  EXP_NAME)
    VIDEO_DIR   = os.path.join("videos",    EXP_NAME)
    RESULTS_CSV = os.path.join("datafile",  EXP_NAME + "_summary.csv")

    actuations = actuationimpactCalculation(
        _om0 / (2 * math.pi), _om0 / (2 * math.pi),
        math.radians(_am0),   math.radians(_am0),
    )

    if ALREADY_SPWANED:
        ALL_INIT = load_initial_conditions(INIT_FILE)
        if not N_TRIALS:
            N_TRIALS = len(ALL_INIT)
        print(f"[main] Loaded {len(ALL_INIT)} initial conditions from '{INIT_FILE}'.")

        # Build per-trial IC index list according to INIT_SELECTION.
        pool = list(range(len(ALL_INIT)))
        _sel = (INIT_SELECTION or 'sequential').lower()

        if _sel.startswith('exp'):
            # Explicit list: the trial count IS len(INIT_INDICES), overriding
            # N_TRIALS_GLOBAL / the auto-read-from-file count above.
            if not INIT_INDICES:
                raise ValueError(
                    "INIT_SELECTION='explicit' but config.INIT_INDICES is empty.")
            bad = [i for i in INIT_INDICES if not (0 <= i < len(pool))]
            if bad:
                raise ValueError(
                    f"INIT_INDICES has out-of-range indices {bad}; the loaded "
                    f"IC file only has indices 0..{len(pool) - 1}.")
            idx_seq  = list(INIT_INDICES)
            N_TRIALS = len(idx_seq)
            print(f'[main] IC selection: EXPLICIT '
                  f'({N_TRIALS} trial(s) from INIT_INDICES: {idx_seq})')
        elif _sel.startswith('rand'):
            idx_seq, shuffled = [], []
            for _ in range(N_TRIALS):
                if not shuffled:
                    shuffled = pool[:]
                    random.shuffle(shuffled)
                idx_seq.append(shuffled.pop())
            print(f'[main] IC selection: RANDOM (without replacement, reshuffles as needed)')
        else:
            idx_seq = [i % len(pool) for i in range(N_TRIALS)]
            print(f'[main] IC selection: SEQUENTIAL')
    else:
        ALL_INIT = None
        idx_seq  = [None] * (N_TRIALS or 1)
        N_TRIALS = len(idx_seq)

    os.makedirs(OUT_DIR,   exist_ok=True)
    os.makedirs(VIDEO_DIR, exist_ok=True)

    # ── Save config snapshot for this experiment ─────────────────────────────
    save_config_snapshot(OUT_DIR)

    results    = []
    start_time = time.time()
    print(f"[main] Starting {N_TRIALS} trial(s) with {WORKERS} worker(s).")

    try:
        if WORKERS == 1:
            for tid in range(N_TRIALS):
                seed    = TRIAL_SEED_BASE + tid
                out_dir = os.path.join(OUT_DIR, f"trial_{tid:04d}")
                ic_idx  = idx_seq[tid]
                ic_tag  = f'IC#{ic_idx}' if ic_idx is not None else 'fresh spawn'
                print(f"\n[main] Trial {tid + 1}/{N_TRIALS}  ({ic_tag})")
                try:
                    res = run_trial(trial_id=tid, seed=seed, preview=PREVIEW,
                                    video_dir=VIDEO_DIR, out_dir=out_dir,
                                    actuations=actuations, ALL_INIT=ALL_INIT,
                                    use_preset=USE_PRESET, init_idx=ic_idx)
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
                        actuations=actuations, ALL_INIT=ALL_INIT,
                        use_preset=USE_PRESET, init_idx=idx_seq[tid]))
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

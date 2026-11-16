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
from sweep_coverage import CoverageMeter

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
    return bool(RECORD_VIDEO) and _policy_selects_trial(trial_id)


def _policy_selects_trial(trial_id: int) -> bool:
    """RECORD_POLICY's choice of trials, shared by videos and snapshots."""
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


class TrialRunner:
    """
    One trial, split into phases so it can be paused at a frame boundary,
    checkpointed, and resumed:

        runner  = TrialRunner(...)     # build the world (or adopt a checkpoint)
        runner.run()                   # main loop up to MAX_RUNTIME
        summary = runner.finalize()    # write every output file

    run_trial() is exactly that sequence. With CHECKPOINT_ENABLED off its
    output is byte-identical to the single run_trial function this replaced.

    State comes in two kinds:
      world  what physics and control carry from one frame to the next: the
             pymunk space, robots and ring, the GaitController (and through its
             ctx the live strategy), the loop counters, the cached per-robot
             arrays, the coverage meter, the RNG states. world_bundle() gathers
             exactly this, and it is all a checkpoint holds.
      io     what is rebuilt from config: pygame surfaces, the video recorder,
             the recorded-data lists, scratch buffers. Never checkpointed.
    _bind() derives the hot-loop handles (ctrl_plan, ring_damp_bodies, ...)
    from the world, and is re-run whenever the world objects are replaced.
    """

    # ── Construction ─────────────────────────────────────────────────────────

    def __init__(self, trial_id, seed, preview, video_dir, out_dir,
                 actuations=None, ALL_INIT=None, use_preset=True,
                 init_idx=None, bundle=None):
        """
        bundle  a world bundle (checkpoint.loads_world of a checkpoint) to
                resume from instead of building a fresh world; ALL_INIT /
                init_idx / use_preset are then ignored.
        """
        self.trial_id  = trial_id
        self.seed      = seed
        self.preview   = preview
        self.video_dir = video_dir
        self.out_dir   = out_dir

        if not preview and (use_preset or bundle is not None):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()

        self.center = pymunk.Vec2d(W / 2, H / 2)

        # Render state is cheap next to the physics and is what lets a video be
        # redrawn later with different styling, or at all, without re-running the
        # trial. Off only if a config explicitly says so.
        self.save_render_state   = bool(getattr(cfg, "SAVE_RENDER_STATE", True))
        self.legacy_pair_metrics = bool(getattr(cfg, "LEGACY_PAIR_METRICS", False))

        # Checkpoints: see CHECKPOINT_ENABLED in config.py and checkpoint.py.
        self.ckpt_every = None
        if getattr(cfg, "CHECKPOINT_ENABLED", False):
            self.ckpt_every = float(getattr(cfg, "CHECKPOINT_EVERY", 5.0))
            if not self.ckpt_every > 0:
                raise ValueError(f"CHECKPOINT_EVERY must be > 0, "
                                 f"got {self.ckpt_every!r}")
        self.ckpt_dir     = os.path.join(out_dir, "checkpoints")
        self.ckpt_bytes   = 0        # total written, reported in the summary
        self.ckpt_count   = 0
        self._cfg_digest  = None

        if bundle is None:
            self._build_world(ALL_INIT, init_idx, use_preset)
        else:
            self._adopt(bundle)
        self._bind()
        self._setup_io()

    def _resolve_gait_callback(self):
        """(runtime_gait_on, callback) from config; resolved, never pickled."""
        on, callback = bool(getattr(cfg, "ENABLE_RUNTIME_GAIT_CONTROL", False)), None
        if on:
            try:
                callback = resolve_callback(
                    getattr(cfg, "RUNTIME_GAIT_CONTROLLER", None))
            except Exception as e:
                print(f"[Trial {self.trial_id}] Could not load RUNTIME_GAIT_CONTROLLER "
                      f"({e!r}); runtime gait control disabled for this trial.")
            if callback is None:
                on = False
        return on, callback

    def _build_world(self, ALL_INIT, init_idx, use_preset):
        trial_id, center = self.trial_id, self.center

        # ── Physics space ─────────────────────────────────────────────────────
        space                = pymunk.Space()
        space.damping        = SPACE_DAMP
        space.iterations     = 60
        space.collision_slop = 0.06

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

        # ── Spawn / load ──────────────────────────────────────────────────────
        # Track the IC index actually used, if any, so it can be reported in the
        # summary CSV regardless of how it was chosen (explicit / random /
        # sequential / the trial_id fallback below).  None for a fresh spawn.
        used_ic_idx = None
        if ALREADY_SPWANED:
            _idx = init_idx if init_idx is not None else trial_id
            used_ic_idx = _idx
            # The robot count in the initial-condition file must match N_SMARTICLES:
            # otherwise COMMAND_ARRAY / BODY_ASSIGNMENT lengths won't line up, and
            # downstream this raises an IndexError at `commands[i]` with no clue why.
            _n_ic = len(ALL_INIT[_idx].get("smarticles", ()))
            if _n_ic and _n_ic != N_SMARTICLES:
                raise ValueError(
                    f"initial condition IC#{_idx} has {_n_ic} robots, but N_SMARTICLES="
                    f"{N_SMARTICLES}. Switch config.INIT_FILE to a file with the matching size "
                    f"(*_p_N17.json=17, *_p_N50.json=50, *_p.json=100), "
                    f"or change N_SMARTICLES.")
            smarticles = build_from_initial_conditions(space, ALL_INIT[_idx])
        else:
            smarticles = spawn_smarticles_auto(space, center, eff_inner_r, N_SMARTICLES)

        # ── Robot ids ─────────────────────────────────────────────────────────
        # The list index has always been the de-facto robot id (video labels, CSV
        # Agent_ID, COMMAND_ARRAY, BODY_ASSIGNMENT all key off it); make it explicit
        # so the runtime gait controller can be addressed by number.
        for _i, _s in enumerate(smarticles):
            _s.id = _i

        initial_positions = [s.main_body.position for s in smarticles]

        # ── Interactive command input (use_preset=False) ──────────────────────
        active_commands = list(COMMAND_ARRAY)
        if not use_preset:
            user_cmds = _show_init_and_get_commands(smarticles, center, trial_id)
            if user_cmds is not None:
                active_commands = user_cmds
            if not self.preview:
                os.environ["SDL_VIDEODRIVER"] = "dummy"
                pygame.init()

        # ── Decode active_commands and apply per-robot parameters ─────────────
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
        self.runtime_gait_on, gait_callback = self._resolve_gait_callback()

        gait_ctl = GaitController(
            smarticles, active_commands,
            warmup_steps=WARMUP_STEPS,
            phase_color_fn=_phase_to_color,
            switch_mode=getattr(cfg, "RUNTIME_GAIT_SWITCH_MODE", "zero_phase"),
            callback=gait_callback,
            verbose=self.runtime_gait_on and bool(
                getattr(cfg, "RUNTIME_GAIT_VERBOSE", False)),
            center=(float(center.x), float(center.y)),
            ctx={"trial_id": trial_id, "n": len(smarticles),
                 "center": (float(center.x), float(center.y)),
                 "inner_r": INNER_R, "dt": 1.0 / 180.0,
                 "max_runtime": MAX_RUNTIME},
        )

        # ── Initial impulses ──────────────────────────────────────────────────
        for s in smarticles:
            s.main_body.apply_impulse_at_local_point(
                pymunk.Vec2d(random.uniform(-500, 500), random.uniform(-500, 500)), (0, 0))

        self.space, self.ring, self.smarticles = space, ring, smarticles
        self.gait_ctl        = gait_ctl
        self.used_ic_idx     = used_ic_idx
        self.active_commands = active_commands

        # Initial positions, for the legacy Mdist metric.
        self.ipx = np.array([p.x for p in initial_positions], dtype=float)
        self.ipy = np.array([p.y for p in initial_positions], dtype=float)

        # actuationimpactCalculation() depends ONLY on robot i's own actuation
        # parameters.  The original evaluated it once per (i, j) pair per frame
        # -> O(n^2 * frames) scipy integrations.  Hoisting it here is exact:
        # identical values, evaluated n times, once -- and re-evaluated for a
        # robot only when a runtime gait switch changes its parameters.
        _acts   = [actuationimpactCalculation(s.omega1, s.omega2, s.A1, s.A2)
                   for s in smarticles]
        self.acts0_i = np.array([a[0] for a in _acts], dtype=float)
        self.acts1_i = np.array([a[1] for a in _acts], dtype=float)

        # ── Coverage meter (k = A'/A_total) ───────────────────────────────────
        # The canvas and the ring mask are built once here and reused every frame;
        # only each robot's own bounding box is touched per measurement.  World
        # state, not io: the canvas grows to fit the swarm as the run goes, and
        # its extent sets the grid the area is counted on.
        self.cov_meter = None
        if getattr(cfg, "COVERAGE_ENABLED", False):
            self.cov_meter = CoverageMeter(
                (center.x, center.y), INNER_R, RING_SHAPE, RING_N_SIDES,
                cell=float(getattr(cfg, "COVERAGE_CELL", 4.0)))

        # ── Loop counters ─────────────────────────────────────────────────────
        self.t         = 0.0
        self.frame_i   = 0
        self.poll_i    = 0     # frame boundaries seen, for RUNTIME_GAIT_POLL_EVERY
        self.snap_k    = 0     # next snapshot division point to capture
        self.snap_t0   = None  # time of the first recorded frame; set in the loop
        self.jam_time  = None
        self.running   = True
        self.ckpt_k    = 0     # index of the next checkpoint to write
        # Frames recorded before this runner started (non-zero only on resume):
        # lets a resumed run's rows be placed on the parent's Step axis.
        self.rec_offset    = 0
        self.render_offset = 0
        self.t_rec0        = None   # sim time of the parent's first recorded frame

    # ── World state: bundle / adopt / bind ───────────────────────────────────

    _LOOP_KEYS = ("t", "frame_i", "poll_i", "snap_k", "snap_t0", "jam_time",
                  "running", "ckpt_k")

    def world_bundle(self) -> dict:
        """Everything a checkpoint holds; see the class docstring."""
        n_rec = self.rec_offset + len(getattr(self, "POS_ALL", ()))
        n_ren = self.render_offset + len(getattr(self, "RENDER_ARMS", ()))
        t_rec0 = self.t_rec0
        if t_rec0 is None and getattr(self, "time_hist", None):
            t_rec0 = self.time_hist[0]
        return {
            "space": self.space, "ring": self.ring, "smarticles": self.smarticles,
            "gait_ctl": self.gait_ctl, "cov_meter": self.cov_meter,
            "loop": {k: getattr(self, k) for k in self._LOOP_KEYS},
            "acts0_i": self.acts0_i, "acts1_i": self.acts1_i,
            "ipx": self.ipx, "ipy": self.ipy,
            "used_ic_idx": self.used_ic_idx,
            "active_commands": list(self.active_commands),
            "rec": {"n_rec": n_rec, "n_render": n_ren, "t_rec0": t_rec0},
            "rng_py": random.getstate(), "rng_np": np.random.get_state(),
        }

    def _adopt(self, b: dict):
        """Take over the world from a bundle (fresh from loads_world)."""
        self.space, self.ring = b["space"], b["ring"]
        self.smarticles       = b["smarticles"]
        self.gait_ctl         = b["gait_ctl"]
        self.cov_meter        = b["cov_meter"]
        for k, v in b["loop"].items():
            setattr(self, k, v)
        self.acts0_i, self.acts1_i = b["acts0_i"], b["acts1_i"]
        self.ipx, self.ipy    = b["ipx"], b["ipy"]
        self.used_ic_idx      = b["used_ic_idx"]
        self.active_commands  = b["active_commands"]
        rec = b.get("rec", {})
        if not hasattr(self, "POS_ALL"):
            # Resuming in a new runner: what came before lives in the parent's
            # files, so this runner's rows start after those.
            self.rec_offset    = int(rec.get("n_rec", 0))
            self.render_offset = int(rec.get("n_render", 0))
            self.t_rec0        = rec.get("t_rec0")
        random.setstate(b["rng_py"])
        np.random.set_state(b["rng_np"])
        # The callback is detached when pickled; re-resolve it from config.
        self.runtime_gait_on, cb = self._resolve_gait_callback()
        self.gait_ctl.callback = cb

    def _bind(self):
        """Derive the hot-loop handles from the current world objects."""
        sm = self.smarticles
        self.n = len(sm)

        # ── Per-robot geometry for the (possibly heterogeneous) coupling model ─
        # In a homogeneous run every entry equals the global L / L_s / S, so the
        # interaction formulas reduce EXACTLY to the original ones.
        self.Ls_i    = np.array([s.L_s      for s in sm], dtype=float)  # half-span
        self.Lw_i    = np.array([s.main_w   for s in sm], dtype=float)  # body width (== L)
        self.Smain_i = np.array([s.main_len for s in sm], dtype=float)  # body length (== S)

        # Pre-bind the pymunk handles touched every physics step.  Attribute
        # chains like s.main_body.velocity are resolved 180 x n times per simulated
        # second; caching the objects removes that lookup without changing any
        # arithmetic.
        self.ctrl_plan = [(s, s.main_body, s.motor_L, s.motor_R) for s in sm]

        # Dynamic bodies making up a MOVABLE ring, so it can get the same per-step
        # floor damping as every robot main_body (see RING_DAMPING_ENABLED).  Empty
        # for a fixed ring (RING_MOVABLE=False) or when the toggle is off, so the
        # loop below costs nothing in that case -- which covers every run in this
        # codebase's regression baseline (RING_MOVABLE defaults to True, but this
        # flag can be set False to reproduce the old, undamped-ring trajectories).
        self.ring_damp_bodies = []
        if RING_DAMPING_ENABLED and RING_MOVABLE:
            if self.ring.body is not None:            # movable circle: one rigid body
                self.ring_damp_bodies = [self.ring.body]
            elif self.ring.bodies:                    # movable polygon: one body/edge
                self.ring_damp_bodies = list(self.ring.bodies)

    # ── io ───────────────────────────────────────────────────────────────────

    def _setup_io(self):
        trial_id = self.trial_id

        # ── Rendering setup ───────────────────────────────────────────────────
        self.orig_surf    = pygame.Surface((W, H))
        self.font         = pygame.font.Font(None, 16)
        self.draw_options = pymunk.pygame_util.DrawOptions(self.orig_surf)
        self.draw_options.constraint_color = (110, 110, 110, 255)
        # Glyph surfaces depend only on the label text, so render each label once
        # and reuse it for every frame: identical pixels, but ~2n fewer
        # font.render() calls per frame (this dominates at n = 100).
        self._label_cache = {}

        self.record_this   = should_record_trial(trial_id)
        self.recorder_orig = None
        if self.record_this:
            video_path         = os.path.join(self.video_dir, f"trial_{trial_id:04d}.mp4")
            self.recorder_orig = VideoRecorder(video_path, W, H, 30)

        # Snapshots: see SNAPSHOT_DIVISIONS in config.py.
        self.snap_n = max(0, int(getattr(cfg, "SNAPSHOT_DIVISIONS", 0) or 0))
        if not _policy_selects_trial(trial_id):
            self.snap_n = 0
        self.snap_dir = os.path.join(self.video_dir, f"trial_{trial_id:04d}_snapshots")
        if self.snap_n:
            os.makedirs(self.snap_dir, exist_ok=True)

        # ── Data containers ───────────────────────────────────────────────────
        self.ORDs, self.ORDs_abs, self.ORD_diffs, self.ORD_diffs_abs = [], [], [], []
        self.Mdist, self.GRs, self.ROT_ORDS, self.IMPACTS, self.POS_ALL = [], [], [], [], []
        self.time_hist = []
        self.COVERAGE  = []     # (t, k, A_union, A0_sum, overlap, k_naive) every N frames
        # Everything a renderer needs that nothing else records: the arm angles
        # (POS_ALL carries only the body pose) and the ring outline (it is movable,
        # so the swarm pushes it out of shape and its geometry cannot be recovered
        # from the config). Collected only on the frames a video would encode, so
        # the cost follows the video rather than the physics.
        self.RENDER_ARMS = []
        self.RENDER_RING = []

        # Scratch arrays reused across frames (positions / angles / displacements).
        n = self.n
        self.A_wall  = np.zeros((n,), dtype=np.float32)
        self.px_buf  = np.empty(n); self.py_buf = np.empty(n); self.psi_buf = np.empty(n)

    # ── Drawing ──────────────────────────────────────────────────────────────

    def _ring_vertices(self):
        """
        Current ring outline as a list of (x, y).

        One vertex per edge taken from each segment's first endpoint, mapped to
        world coordinates: the loop is closed, so n endpoints describe it, and
        for a movable ring each edge is its own body whose position has to be
        read every frame. A static ring never moves, but it costs the same to
        store and keeps the file one shape.
        """
        segs = getattr(self.ring, "segments", None) or []
        out = []
        for sg in segs:
            b = sg.body
            out.append(tuple(b.local_to_world(sg.a)) if b is not None
                       else tuple(sg.a))
        return out

    def _label_surfaces(self, label):
        pair = self._label_cache.get(label)
        if pair is None:
            pair = (self.font.render(label, True, (0, 0, 0)),
                    self.font.render(label, True, (255, 255, 255)))
            self._label_cache[label] = pair
        return pair

    def _blit_id_labels(self, dst):
        for ss in self.smarticles:
            cx, cy = int(ss.main_body.position.x), int(ss.main_body.position.y)
            surf_w, surf_s = self._label_surfaces(str(ss.id + 1))
            tw, th = surf_w.get_size()
            # blit at top-left = center - half-size so the text is centred on (cx, cy)
            lx, ly = cx - tw // 2, cy - th // 2
            dst.blit(surf_s, (lx + 1, ly + 1))   # white shadow
            dst.blit(surf_w, (lx,     ly    ))    # black text

    def _blit_phase_markers(self, dst, radius=7):
        """Draw a filled circle on each robot colored by its initial phase."""
        for ss in self.smarticles:
            x, y  = int(ss.main_body.position.x), int(ss.main_body.position.y)
            color = getattr(ss, "phase_marker_color", (180, 180, 180))
            pygame.draw.circle(dst, (0, 0, 0), (x, y), radius + 1)   # outline
            pygame.draw.circle(dst, color,     (x, y), radius)

    def _draw_frame(self):
        self.orig_surf.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        self._blit_phase_markers(self.orig_surf)
        self._blit_id_labels(self.orig_surf)

    # ── Main loop ────────────────────────────────────────────────────────────

    def _half_frame(self):
        dt = 1.0 / 180.0
        return 0.5 * max(1, int(round((1.0 / RENDER_FPS_HEADLESS) / dt))) * dt

    def run(self):
        """Run to MAX_RUNTIME, writing (and reloading) checkpoints if enabled."""
        every = self.ckpt_every
        if every is None:
            self._run_frames(None)
            return
        half = self._half_frame()
        while True:
            # A checkpoint target that has been reached (k = 0 at t = 0 on a
            # fresh run) is written before any further frame is simulated, and
            # the end of the run always gets one, so a finished run can later be
            # extended. The index of the *next* target is advanced first, so the
            # checkpoint carries it and a resumed run does not rewrite this one.
            if self.t >= self.ckpt_k * every - half or not self.running:
                k = self.ckpt_k
                self.ckpt_k += 1
                while self.ckpt_k * every - half <= self.t:
                    self.ckpt_k += 1
                self.save_checkpoint(reload=self.running, k=k)
            if not self.running:
                return
            self._run_frames(self.ckpt_k * every - half)

    def _run_frames(self, stop_t=None, stop_rec=None):
        """
        The frame loop. Runs until the trial ends, or -- if stop_t is given --
        until the first frame boundary with t >= stop_t, or -- if stop_rec is
        given -- until stop_rec frames have been recorded in total (counting
        the parent's, i.e. on the POS_ALL Step axis: stop_rec = s + 1 stops
        right after Step s is recorded and the controller has polled).

        Everything touched per step is pulled into locals first and the
        counters are written back on exit, so the hot loop is the same
        local-variable code as before the TrialRunner split.
        """
        smarticles   = self.smarticles
        space        = self.space
        gait_ctl     = self.gait_ctl
        ctrl_plan    = self.ctrl_plan
        ring_damp_bodies = self.ring_damp_bodies
        acts0_i, acts1_i = self.acts0_i, self.acts1_i
        center       = self.center
        n            = self.n
        A_wall       = self.A_wall
        px_buf, py_buf, psi_buf = self.px_buf, self.py_buf, self.psi_buf
        ipx, ipy     = self.ipx, self.ipy
        cx_c, cy_c   = float(center.x), float(center.y)
        Ls_i, Lw_i, Smain_i = self.Ls_i, self.Lw_i, self.Smain_i
        legacy_pair_metrics = self.legacy_pair_metrics
        save_render_state   = self.save_render_state
        cov_meter    = self.cov_meter
        cov_every    = max(1, int(getattr(cfg, "COVERAGE_EVERY", 5)))
        record_this  = self.record_this
        recorder_orig = self.recorder_orig
        orig_surf    = self.orig_surf
        snap_n, snap_dir = self.snap_n, self.snap_dir
        runtime_gait_on  = self.runtime_gait_on
        gait_poll_every  = max(1, int(getattr(cfg, "RUNTIME_GAIT_POLL_EVERY", 1)))
        _draw_frame  = self._draw_frame
        _ring_vertices = self._ring_vertices

        ORDs, ORDs_abs, ORD_diffs, ORD_diffs_abs = (
            self.ORDs, self.ORDs_abs, self.ORD_diffs, self.ORD_diffs_abs)
        Mdist, GRs, ROT_ORDS, IMPACTS, POS_ALL = (
            self.Mdist, self.GRs, self.ROT_ORDS, self.IMPACTS, self.POS_ALL)
        time_hist, COVERAGE = self.time_hist, self.COVERAGE
        RENDER_ARMS, RENDER_RING = self.RENDER_ARMS, self.RENDER_RING

        t        = self.t
        frame_i  = self.frame_i
        poll_i   = self.poll_i
        snap_k   = self.snap_k
        snap_t0  = self.snap_t0
        jam_time = self.jam_time
        running  = self.running

        stop_rec_local = None if stop_rec is None else stop_rec - self.rec_offset

        dt         = 1.0 / 180.0
        # MAX_RUNTIME = 45.0
        W_WALL     = 2.0
        D0_WALL    = 1.0 * L_s
        K_WALL     = 1.0
        _vstride   = max(1, VIDEO_STRIDE)

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

                # Ring floor damping -- see ring_damp_bodies in _bind().  A plain
                # Python loop (not vectorised): pymunk Body objects aren't
                # batchable, and the list is at most RING_N_SIDES long (35-85
                # across N=17..100), far smaller than the n-robot loop already run
                # every step above, so this is not a meaningful addition to
                # per-step cost.
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

            # ── Data collection ───────────────────────────────────────────────
            warmup_active = RECORD_AFTER_WARMUP and any(
                s._warmup_step < s.warmup_steps for s in smarticles)

            if not warmup_active:
                positions = [s.main_body.position for s in smarticles]
                psis      = [s.main_body.angle    for s in smarticles]

                for _i, _p in enumerate(positions):
                    px_buf[_i] = _p.x
                    py_buf[_i] = _p.y
                psi_buf[:] = psis

                # Same values as the previous list-of-tuples comprehension
                # (wrap_pi is elementwise, float32 cast is unchanged).
                _wpsi = wrap_pi(psi_buf)
                POS_ALL.append(
                    np.stack((px_buf, py_buf, _wpsi), axis=1).astype(np.float32))

                # ── Legacy pair-coupling metrics ──────────────────────────────
                # Everything below feeds only IMPACTS / GRs / ORDs / ORD_diffs /
                # Mdist / ROT_ORDS, i.e. _impacts.csv, _GRs.npz and
                # _GRs_ORDs_Mdist.csv -- see LEGACY_PAIR_METRICS in config.py for
                # why that is off by default.  Kept verbatim (float_power, the
                # hoisted exponentials, the sequential per-row accumulation) so
                # that turning the flag on still reproduces the old numbers
                # element-for-element.  POS_ALL above is deliberately outside the
                # guard: it is what every live analysis script reads.
                if legacy_pair_metrics:
                    rot_ord = rotation_order_parameters(psis, positions, center)

                    # ── Wall coupling (was the outer half of the pair loop) ───
                    # float_power (not **2) to match pymunk's Vec2d.length, which
                    # is math.sqrt(x**2 + y**2) on Python floats -- see analysis.py
                    _rw   = np.sqrt(np.float_power(px_buf - cx_c, 2.0)
                                    + np.float_power(py_buf - cy_c, 2.0))
                    _dgap = INNER_R - _rw
                    A_wall[:] = K_WALL * sigmoid((Ls_i - _dgap) / (Ls_i + 1e-9))

                    _dx0   = px_buf - ipx
                    _dy0   = py_buf - ipy
                    mdists = np.float_power(_dx0, 2.0) + np.float_power(_dy0, 2.0)

                    # ── Pair coupling: same formula, evaluated in NumPy ───────
                    (dx_matrix, dy_matrix, dPsi_matrix,
                     dxy_matrix, Aij_matrix) = build_pair_matrices(
                        px_buf, py_buf, psi_buf, acts0_i, acts1_i,
                        Ls_i, Lw_i, Smain_i, a0, a1, g0, WC)

                    IMPACT = Aij_matrix.sum(axis=1) / (n - 1) + W_WALL * A_wall
                    IMPACTS.append(IMPACT)
                    _, gr = compute_gr(dxy_matrix)
                    GRs.append(gr)

                    psis_arr = _wpsi
                    sys_ord, sys_ord_abs = calculate_orientational_order(psis_arr)
                    # calculate_orientational_order(row) == (|mean(exp(1j*row))|,
                    # |mean(exp(2j*row))|).  numpy's elementwise exp is chunk-
                    # invariant (verified: exp(1j*M)[i] is bit-identical to
                    # exp(1j*M[i]) ), so the two exponentials are hoisted to a
                    # single whole-matrix call.  The per-row np.mean and the
                    # sequential accumulation below are kept exactly as they were
                    # -- reductions are NOT chunk-invariant.
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

                # Sweep-coverage k.  Strided because it is the one metric whose cost
                # is set by arena area rather than by n, and k varies slowly.
                if cov_meter is not None and frame_i % cov_every == 0:
                    _k, _ci = cov_meter.measure(smarticles)
                    COVERAGE.append((t, _k, _ci["A_union"], _ci["A0_sum"],
                                     _ci["overlap"], _ci["k_naive"]))

                # Only frames that are actually encoded need to be drawn.  The
                # previous code rendered every frame and then threw away
                # (VIDEO_STRIDE - 1) of them; the resulting video is identical.
                if frame_i % _vstride == 0 and save_render_state:
                    RENDER_ARMS.append([
                        (ss.left_body.angle - ss.main_body.angle,
                         ss.right_body.angle - ss.main_body.angle)
                        for ss in smarticles])
                    RENDER_RING.append(_ring_vertices())

                if record_this and frame_i % _vstride == 0:
                    _draw_frame()
                    recorder_orig.add_frame_from_surface(orig_surf)

                # Not tied to VIDEO_STRIDE: every recorded frame is a candidate, and
                # the half-frame tolerance picks the one nearest each target (it
                # also keeps the last target reachable despite float rounding).
                if snap_n:
                    if snap_t0 is None:
                        snap_t0 = t
                    _half_frame = 0.5 * steps_per_frame * dt
                    while (snap_k <= snap_n and t >= snap_t0
                           + snap_k * (MAX_RUNTIME - snap_t0) / snap_n - _half_frame):
                        _draw_frame()
                        pygame.image.save(orig_surf, os.path.join(
                            snap_dir,
                            f"snapshot_{snap_k:03d}of{snap_n:03d}_t{t:.2f}s.png"))
                        snap_k += 1
                frame_i += 1

            # ── Runtime gait control ──────────────────────────────────────────
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

            if stop_t is not None and t >= stop_t:
                break
            if stop_rec is not None and len(POS_ALL) >= stop_rec_local:
                break

        self.t, self.frame_i, self.poll_i = t, frame_i, poll_i
        self.snap_k, self.snap_t0         = snap_k, snap_t0
        self.jam_time, self.running       = jam_time, running

    def n_recorded(self) -> int:
        """Frames recorded so far, on the POS_ALL Step axis (parent's included)."""
        return self.rec_offset + len(self.POS_ALL)

    def truncate_records(self):
        """
        Drop everything recorded so far, keeping the Step / render-frame
        numbering where it is: the next recorded frame still gets Step
        n_recorded(). A branch calls this after replaying up to its branch
        point, so its own files hold only what comes after.
        """
        if self.t_rec0 is None and self.time_hist:
            self.t_rec0 = self.time_hist[0]
        self.rec_offset    += len(self.POS_ALL)
        self.render_offset += len(self.RENDER_ARMS)
        for name in ("ORDs", "ORDs_abs", "ORD_diffs", "ORD_diffs_abs", "Mdist",
                     "GRs", "ROT_ORDS", "IMPACTS", "POS_ALL", "time_hist",
                     "COVERAGE", "RENDER_ARMS", "RENDER_RING"):
            getattr(self, name).clear()

    # ── Checkpoints ──────────────────────────────────────────────────────────

    def save_checkpoint(self, reload=True, path=None, k=None):
        """
        Write the world to a checkpoint file. With reload=True (what run()
        does) the run then continues from pickle.loads() of the very bytes it
        wrote -- see checkpoint.py for why that is what makes replay exact.
        Returns the file path.
        """
        import checkpoint as ckpt

        if self._cfg_digest is None:
            import hashlib, json
            snap, _ = _config_snapshot_dict()
            self._cfg_digest = hashlib.sha256(
                json.dumps(snap, sort_keys=True).encode("utf-8")).hexdigest()[:16]

        if k is None:
            k = self.ckpt_k
        bundle = self.world_bundle()
        raw    = ckpt.dumps_world(bundle)
        if path is None:
            path = os.path.join(self.ckpt_dir, ckpt.checkpoint_name(k, self.t))
        rec = bundle["rec"]
        meta = {"trial_id": self.trial_id, "k": k, "t": self.t,
                "n": self.n, "frame_i": self.frame_i, "poll_i": self.poll_i,
                "n_rec": rec["n_rec"], "n_render": rec["n_render"],
                "t_rec0": rec["t_rec0"], "running": self.running,
                "ic_idx": self.used_ic_idx, "config_digest": self._cfg_digest}
        self.ckpt_bytes += ckpt.write(
            path, raw, meta,
            level=int(getattr(cfg, "CHECKPOINT_COMPRESS_LEVEL", 6)))
        self.ckpt_count += 1
        if reload:
            self._adopt(ckpt.loads_world(raw))
            self._bind()
        return path

    @classmethod
    def from_checkpoint(cls, path, out_dir, video_dir=None, trial_id=None,
                        preview=False):
        """
        A runner that continues from a checkpoint file. Its recorded data
        starts at the checkpoint (rec_offset / render_offset say where that is
        on the parent's axes). Config must match the parent run's -- load the
        parent's config_snapshot.json through SMARTICLE_CONFIG first.
        """
        import checkpoint as ckpt
        meta, raw = ckpt.read(path)
        tid = meta["trial_id"] if trial_id is None else trial_id
        return cls(trial_id=tid, seed=None, preview=preview,
                   video_dir=video_dir or out_dir, out_dir=out_dir,
                   bundle=ckpt.loads_world(raw))

    # ── Output ───────────────────────────────────────────────────────────────

    def finalize(self):
        """Close the recorder, write every output file, return the summary."""
        smarticles   = self.smarticles
        trial_id     = self.trial_id
        out_dir      = self.out_dir
        gait_ctl     = self.gait_ctl
        time_hist    = self.time_hist
        POS_ALL      = self.POS_ALL
        IMPACTS      = self.IMPACTS
        COVERAGE     = self.COVERAGE
        RENDER_ARMS, RENDER_RING = self.RENDER_ARMS, self.RENDER_RING
        _vstride     = max(1, VIDEO_STRIDE)

        # ── Post-processing ───────────────────────────────────────────────────
        final_positions          = np.array([s.main_body.position for s in smarticles])
        final_rg, final_kappa, final_area = calculate_macro_shape(final_positions)

        if self.record_this:
            self.recorder_orig.close()
        pygame.quit()

        def col(lst):
            a = np.reshape(np.array(lst), (-1, 1))
            return a

        GRs_ORDs_Mdist = np.hstack([
            col(self.ORDs_abs), col(self.ORDs), col(self.ORD_diffs),
            col(self.ORD_diffs_abs), col(self.ROT_ORDS), col(self.Mdist),
        ])
        GRs_arr = np.array(self.GRs)

        # ── Save outputs ──────────────────────────────────────────────────────
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

        # Which gait each robot was assigned, as an **event list**: one row per
        # robot for its initial COMMAND_ARRAY value, then one row per change.
        # Forward-fill by robot_id to get the full timeline. A few hundred rows next
        # to POS_ALL's hundreds of thousands, and nothing extra is computed during
        # the run.
        #
        # What gets recorded is the **decision**, not the execution: the moment the
        # strategy chose a command, not the zero-crossing where the arm picked it
        # up. That is the definition analysis uses throughout, because it is the
        # only one that can also be reconstructed for trials that predate this file
        # -- LayeredStrategy.decide() is a deterministic function of the recorded
        # positions, so replaying POS_ALL regenerates exactly this list. Recording
        # the execution instead would make new trials incomparable with old ones.
        # See batch_group_plots.reconstruct_gait_log.
        #
        # A controller that is not a LayeredStrategy has no decision log, so those
        # runs fall back to GaitState.history -- the applied times. That is noted in
        # the file's `source` column rather than silently mixed in.
        try:
            strat = getattr(gait_ctl, "ctx", {}).get("strategy")
            decisions = getattr(strat, "decisions", None)
            source = "decision" if decisions is not None else "applied"

            rows = []
            for st in gait_ctl.states:
                # history only holds changes, so the initial command is the first
                # entry's old_cmd — or the current one if it never switched
                first = st.history[0][1] if st.history else st.command
                if first is not None:
                    rows.append((0.0, st.robot_id, int(first)))
            if decisions is not None:
                rows.extend((float(t), int(rid), int(cmd))
                            for t, rid, cmd in decisions)
            else:
                rows.extend((float(ts), st.robot_id, int(new))
                            for st in gait_ctl.states
                            for ts, _old, new in st.history)

            # Rebase onto POS_ALL's clock. RECORD_AFTER_WARMUP means recording only
            # starts once every robot is through its warm-up, so POS_ALL row 0 is
            # sim time time_hist[0], not 0 — while the times above are raw sim
            # time. Without this shift the pattern figure would sit ~WARMUP_STEPS
            # ahead of every other curve plotted against the same axis.
            # Everything at or before the first recorded frame collapses to t=0,
            # keeping the **last** such value per robot: that is what a robot is
            # already running when recording opens.
            #
            # A resumed runner rebases on the parent's first frame (t_rec0), so
            # its log shares the parent's time axis.
            t0 = (self.t_rec0 if self.t_rec0 is not None
                  else (time_hist[0] if time_hist else 0.0))
            rows.sort(key=lambda r: (r[0], r[1]))
            pre, post = {}, []
            for t, rid, cmd in rows:
                if t - t0 <= 0.0:
                    pre[rid] = cmd
                else:
                    post.append((t - t0, rid, cmd))
            rows = sorted((0.0, rid, cmd) for rid, cmd in pre.items()) + post

            # Drop no-ops: an initial command the strategy then re-decides
            # identically would otherwise read as a switch to itself
            deduped, last = [], {}
            for t, rid, cmd in rows:
                if last.get(rid) != cmd:
                    last[rid] = cmd
                    deduped.append((t, rid, cmd))
            rows = deduped

            if rows:
                with open(prefix + "_gait_log.csv", "w", newline="",
                          encoding="utf-8") as f:
                    f.write("time,robot_id,command,source\r\n")
                    f.write("".join(f"{t:.6f},{rid},{cmd},{source}\r\n"
                                    for t, rid, cmd in rows))
        except Exception as e:
            print(f"[WARN] gait log write failed: {type(e).__name__}: {e}")

        # Render state: float32 is ample for pixel coordinates (the largest arena
        # is ~6900 px and float32 carries 7 significant digits) and halves the file
        # against float64. npz rather than csv because this is read by a renderer,
        # never by eye, and the text form would be 3-4x larger.
        if self.save_render_state and RENDER_ARMS:
            try:
                np.savez_compressed(
                    prefix + "_render.npz",
                    arms=np.asarray(RENDER_ARMS, dtype=np.float32),
                    ring=np.asarray(RENDER_RING, dtype=np.float32),
                    fps=np.float32(RENDER_FPS_HEADLESS / max(1, VIDEO_STRIDE)),
                    stride=np.int32(_vstride))
            except Exception as e:
                print(f"[WARN] render state write failed: {type(e).__name__}: {e}")

        if COVERAGE:
            with open(prefix + "_coverage.csv", "w", newline="", encoding="utf-8") as f:
                f.write("time,k,A_union,A0_sum,overlap,k_naive\r\n")
                f.write("".join(
                    f"{ti:.6f},{kk:.6f},{au:.2f},{a0:.2f},{ov:.6f},{kn:.6f}\r\n"
                    for ti, kk, au, a0, ov, kn in COVERAGE))

        pos_fmt = str(getattr(cfg, "POS_ALL_FORMAT", "csv")).lower()
        if POS_ALL and pos_fmt == "npz":
            # See pos_all_io.py: same data, ~1/5 the size, read back by
            # read_pos_all() into the same DataFrame the CSV gives.
            import pos_all_io
            pos_path = prefix + "_POS_ALL.npz"
            pos_all_io.write_npz(pos_path, np.stack(POS_ALL), step0=self.rec_offset)
            try:
                align_df = process_pos_all(pos_path, max_dist=150.0)
                align_df.to_csv(prefix + "_alignment.csv", index=False)
            except Exception as e:
                print(f"[WARN] alignment computation failed: {e}")
        elif POS_ALL:
            with open(prefix + "_POS_ALL.csv", "w", newline="") as f:
                # Identical bytes to the csv.writer version (CRLF terminator, no
                # quoting needed), but this file has frames x n rows -- 360k rows at
                # n = 100 -- so per-row csv.writer overhead was significant.
                # A resumed runner numbers its Steps on from the parent's.
                f.write("Step,Agent_ID,X,Y,Theta\r\n")
                _chunk = []
                for step_idx, frame in enumerate(POS_ALL, start=self.rec_offset):
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

        summary = {"trial_id": trial_id,
                   "ic_idx": self.used_ic_idx if self.used_ic_idx is not None else "",
                   "final_rg": final_rg,
                   "final_kappa": final_kappa,
                   "final_area":  final_area}
        if COVERAGE:
            # k over the whole trial, and over its last 10 s -- the steady-state
            # value is the one worth comparing across trials.
            _kk = np.array([c[1] for c in COVERAGE], dtype=float)
            _tt = np.array([c[0] for c in COVERAGE], dtype=float)
            _tail = _kk[_tt >= (_tt[-1] - 10.0)]
            summary["k_mean"]    = float(_kk.mean())
            summary["k_steady"]  = float(_tail.mean())
            summary["k_overlap"] = float(np.mean([c[4] for c in COVERAGE]))
        if self.ckpt_count:
            summary["n_checkpoints"]  = self.ckpt_count
            summary["checkpoint_mb"]  = round(self.ckpt_bytes / 1e6, 3)
        return summary


def run_trial(trial_id, seed, preview, video_dir, out_dir,
              actuations, ALL_INIT=None, use_preset=True,
              init_idx=None):
    """
    Run a single trial: physics loop, data collection, and file output.
    When use_preset=False, displays the initial layout in a window and
    prompts the user for a per-robot COMMAND_ARRAY before starting.
    Returns a dict with summary scalars.

    With config.CHECKPOINT_ENABLED, also writes a checkpoint every
    CHECKPOINT_EVERY simulated seconds to <out_dir>/checkpoints/ (see
    checkpoint.py).
    """
    runner = TrialRunner(trial_id, seed, preview, video_dir, out_dir,
                         actuations=actuations, ALL_INIT=ALL_INIT,
                         use_preset=use_preset, init_idx=init_idx)
    runner.run()
    return runner.finalize()


# =============================================================================
# Config snapshot
# =============================================================================

def _jsonable(value, _depth=0):
    """
    Convert one config value into something json can write, or raise TypeError.

    Recursive on purpose: the settings added since this file was written are
    nested -- STRATEGY_SPEC holds a list of rule dicts whose "size_range" is a
    tuple, RUNTIME_GAIT_SCRIPT is a list of (seconds, text) tuples.  A
    shallow pass would either drop them or write something json chokes on.
    """
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        # inf / nan are not valid JSON; keep them readable rather than crash
        return value if math.isfinite(value) else repr(value)
    if _depth > 8:
        raise TypeError("nested deeper than 8 levels")
    if isinstance(value, (list, tuple)):
        return [_jsonable(x, _depth + 1) for x in value]
    if isinstance(value, (set, frozenset)):
        return [_jsonable(x, _depth + 1) for x in sorted(value, key=repr)]
    if isinstance(value, dict):
        # json needs string keys; config only ever uses str/int keys anyway
        return {str(k): _jsonable(v, _depth + 1) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist(), _depth + 1)
    if isinstance(value, np.generic):
        return _jsonable(value.item(), _depth + 1)
    raise TypeError(type(value).__name__)


def _config_snapshot_dict():
    """
    (snapshot, skipped): every public config value in JSON form, and the ones
    that could not be converted. Shared by save_config_snapshot and the
    checkpoint config digest, so both describe the config the same way.
    """
    import config as _cfg
    import types

    snapshot, skipped = {}, {}
    for k, v in vars(_cfg).items():
        if k.startswith("_") or isinstance(v, types.ModuleType):
            continue
        try:
            snapshot[k] = _jsonable(v)
        except Exception as e:
            skipped[k] = {"type": type(v).__name__, "repr": repr(v)[:200],
                          "why": str(e)}
    return snapshot, skipped


def save_config_snapshot(out_dir: str):
    """
    Save the experiment's config as JSON in the out_dir root.

    Everything public in config.py that can be represented in JSON is written,
    including the dict-valued settings (STRATEGY_SPEC, GAIT_BY_TYPE,
    BODY_TYPES) that the earlier version silently dropped.  Names starting with
    "_" are module-level intermediates and are skipped, as are imported modules.
    Anything else that cannot be serialised is listed under "_skipped" with its
    type and repr, so a parameter can never disappear from the record without
    leaving a trace.
    """
    import json

    snapshot, skipped = _config_snapshot_dict()
    if skipped:
        snapshot["_skipped"] = skipped
    # Mark this as a full snapshot (including derived quantities). config.py uses
    # this to decide "skip derived quantities when applying", so this file can be
    # used directly as an override file to rerun this exact experiment.
    snapshot["_snapshot"] = {
        "version": 1,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": ("Use directly: SMARTICLE_CONFIG=<this file> python simulation.py; "
                 "or reduce it to a hand-editable config file with: "
                 "python experiments.py --from-snapshot <this file>"),
    }

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_snapshot.json")
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
    except Exception as e:
        # A snapshot problem must not abort the experiment it is describing
        print(f"[main] [WARN] Config snapshot failed: {type(e).__name__}: {e}")
        return
    note = f"  ({len(skipped)} entr{'y' if len(skipped) == 1 else 'ies'} "
    note += "not serialisable, see _skipped)" if skipped else ""
    print(f"[main] Config snapshot saved: {path}"
          f"  [{len(snapshot) - bool(skipped)} settings]"
          f"{note if skipped else ''}")


# =============================================================================
# Entry point
# =============================================================================

def main():
    # These used to be hardcoded here, now read from config — each condition in a
    # batch comparison experiment needs a different init file and output location,
    # so hardcoding them would break override-file-driven runs.
    WORKERS    = PARALLEL_WORKERS          # >1 for parallel trials (config.py)
    PREVIEW    = getattr(cfg, "PREVIEW", False)
    INIT_FILE  = cfg.INIT_FILE
    N_TRIALS   = N_TRIALS_GLOBAL           # 0 = auto-read from init file
    USE_PRESET = getattr(cfg, "USE_PRESET", True)

    # ── Generate unified experiment name using naming module ────────────────────
    _cmd0  = COMMAND_ARRAY[0];  _abs0 = abs(_cmd0)
    _z0    = _abs0 % 10;        _y0   = (_abs0 % 100 - _z0) // 10;  _x0 = _abs0 // 100
    _ATAB, _FTAB = gait.AMPLI_TABLE, gait.FREQ_TABLE
    _om0   = (_FTAB[_z0 - 1] if 1 <= _z0 <= 9 else 0.5) * 2 * math.pi
    _am0   = math.degrees(_ATAB[_y0 - 1] if 1 <= _y0 <= 6 else math.pi/4)
    # Name only carries robot count and timestamp; full parameters live in config_snapshot.json
    EXP_NAME = getattr(cfg, "EXP_NAME", None) or generate_trial_name(N_SMARTICLES)
    print(f"[main] Experiment name: {EXP_NAME}")

    _root       = getattr(cfg, "OUT_ROOT", "datafile")
    OUT_DIR     = os.path.join(_root,    EXP_NAME)
    VIDEO_DIR   = os.path.join("videos", EXP_NAME)
    RESULTS_CSV = os.path.join(_root,    EXP_NAME + "_summary.csv")

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

"""
config.py  ─  All global parameters for the Smarticle simulation.
Edit this file to change experiment settings.
"""

import math
import random

# =============================================================================
# Global Configuration  ← all tunable parameters are defined here
# =============================================================================

# ── Trial / seed ──────────────────────────────────────────────────────────────
TRIAL_SEED_BASE   = 12345
N_TRIALS_GLOBAL   = 3       # 0 means auto-read from initial-conditions file
MAX_RUNTIME       = 45.0    # in seconds

# ── Video recording ───────────────────────────────────────────────────────────
RECORD_VIDEO      = True
RECORD_POLICY     = "mod"   # "mod" | "first_n" | "both"
RECORD_EVERY_K    = 1       # record trials where (trial_id % K == 0)
RECORD_FIRST_N    = 200       # record the first N trials
VIDEO_FPS         = 30      # output video fps
VIDEO_STRIDE      = 2       # write 1 frame every N simulation frames
VIDEO_CODEC       = "mp4v"  # OpenCV fourcc

# ── Warmup ────────────────────────────────────────────────────────────────────
# WARMUP_STEPS: number of physics steps for the joint-angle warm-up ramp.
# With dt = 1/180 s, WARMUP_STEPS=180 => 1 s warm-up.  0 = instant jump.
WARMUP_STEPS        = 180
# When True, ALL data recording (CSV, npz) and video recording begin only after
# every smarticle's warm-up counter has finished.
RECORD_AFTER_WARMUP = True

# ── Screen / geometry ─────────────────────────────────────────────────────────
W, H              = 900, 760   # screen size (pixels)
SCREEN_MARGIN     = 26         # margin between outer wall and window edge

# ── Reference geometry (used for auto-scaling) ────────────────────────────────
BASE_N_REF        = 17
BASE_WALL_THICK   = 20
BASE_MAIN_LEN, BASE_MAIN_W = 70, 41
BASE_ARM_LEN,  BASE_ARM_W  = 70, 6

# ── Experiment size ───────────────────────────────────────────────────────────
N_SMARTICLES      = 17         # number of smarticles in the simulation
INNER_R_UNSCALED  = 245        # inner ring radius before scaling

# ── Auto-scaling (do not edit unless you know what you are doing) ─────────────
_outer_need       = INNER_R_UNSCALED + BASE_WALL_THICK
_outer_allow      = (min(W, H) / 2.0) - SCREEN_MARGIN
SCALE             = min(1.0, _outer_allow / max(1.0, _outer_need))
INNER_R           = max(20, int(INNER_R_UNSCALED * SCALE))
WALL_THICK        = max(8,  int(BASE_WALL_THICK   * SCALE))
WALL_SEGMENTS     = int(max(120, min(480, 240 * math.sqrt(max(1, N_SMARTICLES) / BASE_N_REF))))
MAIN_LEN, MAIN_W  = max(18, int(BASE_MAIN_LEN * SCALE)), max(10, int(BASE_MAIN_W * SCALE))
ARM_LEN,  ARM_W   = max(18, int(BASE_ARM_LEN  * SCALE)), max(3,  int(BASE_ARM_W  * SCALE))

# ── Wall physics ──────────────────────────────────────────────────────────────
WALL_FRICTION     = 0.9
WALL_ELASTICITY   = 0.0

# ── Actuation ─────────────────────────────────────────────────────────────────
# === THIS IS WHERE YOU CONTROL JOINT MOTION ===
# Desired joint angle trajectory: theta(t) = A * sin(omega * t + phase)
# Left arm  uses A1, omega1, phase1 (set per-robot in run_trial via INIT_PHASES)
# Right arm uses A2, omega2, phase2
A_DEG_NOM         = 30.0    # nominal amplitude (deg) [kept for back-compat]
A_DEG_NOM1        = 90.0    # left  arm amplitude (deg)
A_DEG_NOM2        = 90.0    # right arm amplitude (deg)
OMEGA_NOM         = 10.0    # nominal frequency (rad/s) [kept for back-compat]
OMEGA_NOM1        = 0.5 * 2 * math.pi   # left  arm frequency (rad/s)
OMEGA_NOM2        = 0.5 * 2 * math.pi   # right arm frequency (rad/s)
# RATE_LIM          = 8.0     # motor velocity clamp (rad/s)
# ANG_DAMP          = 0.965   # per-step angular velocity damping factor
# SPACE_DAMP        = 0.985   # velocity reserved each step
# JOINT_LIMIT_DEG   = 85.0    # hard joint angle limit (deg)
# KP_NOM            = 30.0    # proportional gain for motor PD controller

RATE_LIM          = 8.0     # motor velocity clamp (rad/s)
ANG_DAMP          = 0.8     # per-step angular velocity damping factor
LIN_DAMP          = 0.8     # per-step linear velocity damping factor
SPACE_DAMP        = 1.0     # velocity reserved each step
JOINT_LIMIT_DEG   = 95.0    # hard joint angle limit (deg)
KP_NOM            = 60.0    # proportional gain for motor PD controller

# Per-robot initial phases; length must equal N_SMARTICLES.
INIT_PHASES = [
    #(p := random.random() * 2 * math.pi, p)
    (math.pi*5/4, math.pi*5/4)
    for _ in range(N_SMARTICLES)
]

# ── Coupling / interaction model ──────────────────────────────────────────────
L    = MAIN_W
S    = MAIN_LEN
WC   = 1.0
L_s  = (MAIN_LEN + 2 * ARM_LEN) / 2
R0   = L_s + 0.01 * S
a0   = 0.1
a1   = 1.0
g0   = 0.01

# ── Motor torque & mass (scaled) ──────────────────────────────────────────────
BASE_MAX_MOTOR_TORQUE = 3e7
MAX_MOTOR_TORQUE_NOM  = BASE_MAX_MOTOR_TORQUE * (SCALE ** 4)
# BASE_MAX_MOTOR_TORQUE = 3e7
# MAX_MOTOR_TORQUE_NOM  = BASE_MAX_MOTOR_TORQUE * (SCALE ** 4)
BASE_MASS_MAIN        = 172.14
BASE_MASS_ARM         = 6.40
MASS_MAIN             = BASE_MASS_MAIN * (SCALE ** 2)
MASS_ARM              = BASE_MASS_ARM  * (SCALE ** 2)

# ── Physics stability ─────────────────────────────────────────────────────────
SPACE_DAMPING     = 0.985
SPACE_ITERATIONS  = 60
COLLISION_SLOP    = 0.06 * SCALE
V_MAX             = 900.0 * SCALE
W_MAX             = 35.0

# ── Spawn / packing ───────────────────────────────────────────────────────────
PEN_EPS           = 0.1 * SCALE
SETTLE_STEPS      = 10
SETTLE_DT         = 1 / 800.0

# ── Parameter jitter (diversity across robots) ────────────────────────────────
ENABLE_PARAMETER_JITTER = False
A_JITTER_FRAC     = 0.20   # +/-20% amplitude variation
OMEGA_JITTER_FRAC = 0.20   # +/-20% frequency variation
KP_JITTER_FRAC    = 0.30   # +/-30% kp variation
TORQUE_JITTER_FRAC= 0.30   # +/-30% motor torque variation

# ── Initial angle mixture (used by sample_initial_joint_angle) ────────────────
INIT_MIX_FOLDED   = 0.25
INIT_MIX_STRAIGHT = 0.35
INIT_MIX_UNIFORM  = 0.40
STRAIGHT_BAND_DEG = 12.0

# ── Stepping / rendering ──────────────────────────────────────────────────────
SIM_DT              = 1.0 / 180.0
RENDER_FPS_PREVIEW  = 60
RENDER_FPS_HEADLESS = 60

# ── Misc ──────────────────────────────────────────────────────────────────────
SCORE_VALID         = False
SAVE_NPY            = False
ALREADY_SPWANED     = True
rho                 = N_SMARTICLES / (W * H)  # number density

# =============================================================================
# End of global configuration
# =============================================================================

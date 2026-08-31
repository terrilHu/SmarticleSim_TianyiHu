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
N_TRIALS_GLOBAL   = 10       # 0 means auto-read from initial-conditions file
MAX_RUNTIME       = 100.0    # in seconds

# ── Initial-condition selection (only used when ALREADY_SPWANED = True) ────
# "sequential" : use init_conditions[0], [1], [2] ... in order (default)
# "random"     : draw without replacement each run; reshuffles when the pool
#                is exhausted (trials may repeat across reshuffles but never
#                within one)
# "explicit"   : run exactly the IC indices listed in INIT_INDICES, in that
#                order (repeats allowed, e.g. to re-run the same IC with
#                different seeds).  N_TRIALS is IGNORED in this mode -- the
#                trial count is len(INIT_INDICES).
INIT_SELECTION    = "explicit"

# Only read when INIT_SELECTION == "explicit".  0-based indices into the
# loaded initial-conditions file (same numbering as "IC#" in the console log
# and in the summary CSV's ic_idx column).
#   INIT_INDICES = [40]              # just IC#40, once
#   INIT_INDICES = [40, 40, 40]      # IC#40, three times (e.g. different seeds)
#   INIT_INDICES = list(range(50, 60))   # IC#50 .. IC#59
INIT_INDICES      = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]#[116, 87, 194]

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

# ── Experiment size ───────────────────────────────────────────────────────────
N_SMARTICLES      = 17         # number of smarticles in the simulation

# ── Reference geometry (used for auto-scaling) ────────────────────────────────
BASE_N_REF        = 17         # population the reference arena was tuned for
BASE_WALL_THICK   = 20
BASE_MAIN_LEN, BASE_MAIN_W = 70, 41
BASE_ARM_LEN,  BASE_ARM_W  = 70, 6

# ── Population scaling ────────────────────────────────────────────────────────
# Robot size is fixed, so putting N robots in the BASE_N_REF arena changes the
# areal packing fraction as N/R^2.  With AUTO_SCALE_ARENA the arena (and the
# window it is drawn in) grows as sqrt(N / BASE_N_REF), which holds the packing
# fraction — and therefore the collective regime — constant as N grows.
#
#   N = 17  -> factor 1.0 exactly  -> every derived value below is UNCHANGED.
#   N = 100 -> factor ~2.43        -> INNER_R 245 -> 594 px, window 2182x1843.
#
# Set to False to keep the fixed 900x760 / R=245 arena regardless of N (which
# is only sensible for small N: 100 robots do not physically fit in R=245).
AUTO_SCALE_ARENA  = True
_POP_SCALE        = (max(1.0, math.sqrt(N_SMARTICLES / BASE_N_REF))
                     if AUTO_SCALE_ARENA else 1.0)

# ── Screen / geometry ─────────────────────────────────────────────────────────
BASE_W, BASE_H    = 900, 760   # reference screen size (pixels) at BASE_N_REF
W, H              = int(round(BASE_W * _POP_SCALE)), int(round(BASE_H * _POP_SCALE))
SCREEN_MARGIN     = 26         # margin between outer wall and window edge

BASE_INNER_R_UNSCALED = 245    # inner ring radius at BASE_N_REF, before scaling
INNER_R_UNSCALED  = BASE_INNER_R_UNSCALED * _POP_SCALE

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

# ── Heterogeneous bodies (per-individual physical characteristics) ────────────
# When ENABLE_HETEROGENEOUS_BODIES is False the simulation is fully homogeneous
# and behaves EXACTLY as before (every robot uses the global geometry above).
#
# When True, every robot is assigned a "body type" by index via BODY_ASSIGNMENT.
# A body type is a dict of OVERRIDES on the base (UNSCALED) geometry; anything not
# listed falls back to the BASE_* defaults, and the same SCALE pipeline + clamps
# used for the global geometry are applied automatically (see bodies.py).
#
# Recognised override keys:
#   "main_len_base", "main_w_base", "arm_len_base", "arm_w_base"  (unscaled px)
#   "mass_main", "mass_arm"   (absolute, already-scaled; optional)
# If a mass is omitted it is auto-derived from the area ratio relative to the
# homogeneous default, so a bigger arm is automatically heavier.
ENABLE_HETEROGENEOUS_BODIES = False

# Library of reusable body types ("species"). "default" = the global geometry.
BODY_TYPES = {
    "default":   {},                       # global BASE_* geometry, unchanged
    "long_arm":  {"arm_len_base": 110},    # longer arms (base 70 -> 110)
    "short_arm": {"arm_len_base": 45},     # shorter arms (base 70 -> 40)
    # "heavy":   {"main_w_base": 60, "mass_main": 400.0},
}

# Per-robot assignment; length MUST equal N_SMARTICLES. Each entry is a key of
# BODY_TYPES. Example for a 17-robot mixed population:
#   BODY_ASSIGNMENT = (["long_arm"] * 6) + (["short_arm"] * 6) + (["default"] * 5)
BODY_ASSIGNMENT = ["default"] * N_SMARTICLES
random.shuffle(BODY_ASSIGNMENT)

# Per-robot command array; length must equal N_SMARTICLES.
# Each entry is a signed 3-digit integer ±XYZ where:
#   X (hundreds, 0-8) : initial phase  — 0 = random, 1-8 from table
#   Y (tens,     1-6) : amplitude      — lookup table index
#   Z (units,    1-9) : frequency      — lookup table index
#   sign (+)          : both joints same phase
#   sign (-)          : joints in antiphase (phase2 = phase1 + pi)
# Example: -226 → antiphase, |phase|=pi/2, A=pi/6, f=3Hz
#          +051 → same phase, phase=random, A=pi*5/12, f=0.5Hz
#COMMAND_ARRAY = [862] * N_SMARTICLES   # default: same phase pi*5/4, A=pi/2, f=3Hz
# Two-population mix, expressed as a fraction so it follows N_SMARTICLES.
# At N_SMARTICLES = 17 this is exactly [862] * 9 + [462] * 8 as before.
_CMD_A, _CMD_B    = 432, 832
_CMD_A_FRACTION   = 0 / 17
_n_cmd_a          = int(round(N_SMARTICLES * _CMD_A_FRACTION))
COMMAND_ARRAY = [_CMD_A] * _n_cmd_a + [_CMD_B] * (N_SMARTICLES - _n_cmd_a)
random.shuffle(COMMAND_ARRAY)

# Or in the debug shorthand (gait.py), deterministic by id rather than shuffled:
#   from gait import build_command_array
#   COMMAND_ARRAY = build_command_array("a462; 0..8862", N_SMARTICLES)

# GAIT_BY_TYPE will be automatically chosen when heterogeneous is enabled
GAIT_BY_TYPE = {
    "default":    -851,
    "long_arm":  -426,
    "short_arm":  861,
}

# ── Runtime gait control ──────────────────────────────────────────────────────
# Change any robot's motion pattern mid-run, addressed by id.  Reference lives
# with the code: gait.py (command encoding + the "a-862" debug syntax),
# gait_control.py (callback contract + ready-made controllers), strategy.py
# (grouping/roles pipeline and its tuning notes).
# False -> the callback is never resolved; results bit-identical to before.
ENABLE_RUNTIME_GAIT_CONTROL = True

# "module:function".  See gait_control.py for what each one does:
#   strategy | script | example_text | example_schedule
#   example_closed_loop | example_stagger
RUNTIME_GAIT_CONTROLLER  = "gait_control:strategy"

RUNTIME_GAIT_POLL_EVERY  = 1             # poll the callback every N frames
RUNTIME_GAIT_SWITCH_MODE = "zero_phase"  # "zero_phase" (continuous) | "immediate"
RUNTIME_GAIT_VERBOSE     = False         # log every applied switch

# For "gait_control:script": (seconds, "command text"), syntax table in gait.py.
#   a/all/*  everyone    even/odd  by parity    3,7,12  those    0..4  a range
#   a,~3  all but 3    ";" separates clauses, later wins    "a-862" -> all to -862
RUNTIME_GAIT_SCRIPT = [
    # ( 5.0, "a-862"),
    # (10.0, "0..4851"),
    # (15.0, "3,7,12=461"),
]

# ── Strategy: Voronoi grouping -> spatial roles -> layered commands ───────────
# Used by "gait_control:strategy".  Priority: group_rules > roles > leave_command.
# Pipeline, the two tuning traps, and the measured max_dist percolation table
# are all documented at the top of strategy.py.
_BODY_SPAN = (MAIN_LEN + 2 * ARM_LEN) / 2.0        # == L_s, defined further down
STRATEGY_SPEC = {
    "max_dist":      65,                 # ~116 px at N=100; percolates above 1.2
    "period":        0.25,               # seconds of sim time between recomputes
    "leave_command": 32,                 # layer 3: interior / isolated robots
   
    # Layer 1 -- by group size.  First match wins; (lo, hi) inclusive, None =
    # open.  Keep the top end bounded or layer 2 starves -- see strategy.py.
    # Optional "alignment_range" also gates on the group's nematic order.
    "group_rules": [
        {"name": "small", "size_range": (3, 7),  "command":  62},
        {"name": "mid",   "size_range": (8, None), "command": 62},
    ],
    "group_n_ticks": 6,                  # ticks holding a rule before it commits

    # Layer 2 -- spatial roles.  Earlier entries in this list win over later
    # ones.  Names: spatial_roles.SELECTORS.  Any extra key is passed straight
    # to the selector -- the useful ones for the PCA family
    # (principal/major/minor_ends, group_*_ends) are:
    #     "n_per_end":      每端取几台机器人 (默认 1)
    #     "axis":           "major" / "minor" / "both"
    #     "min_anisotropy": 长短比低于此值就判定方向没意义，该帧不选人 (默认 1.5)
    #     "min_group_size": 最大组小于此值就不选人 (group_*_ends)
    #     "select_from":    "all" 在全场选端点(默认) / "group" 只在该组内选
    # "farthest" 用的是 "n"；extremes / convex_hull 之类没有额外参数。
    #
    # "override_group": True 把这条角色提到分组层**之上** —— 默认 False，即
    # 机器人一旦被某条 group_rule 认领就执行分组指令，角色管不着它。
    "roles": [
        # {"selector": "convex_hull", "command": -52,
        #  "n_frames_join": 18, "n_frames_leave": 18},
        {"selector": "group_major_ends", "command": -52, "min_group_size": 3,
         "n_per_end": 1, "override_group": False, "n_per_end": 6},
    ],

    "verbose": False,                    # log group/role join+leave events
}

# ── Coverage ratio k = A' / A_total ───────────────────────────────────────────
# Per robot, the region its two arms can sweep: the body rectangle plus a sector
# of radius arm_len and opening 2A at each shoulder, so A_0 = main_len*main_w +
# (A1+A2)*arm_len^2 -- set purely by the amplitude.  A' is the UNION of all n
# such regions (overlaps counted once, A' <= n*A_0), A_total the ring's area.
# Large k = the swarm's reachable area fills the arena and overlaps heavily,
# i.e. more interaction and collision.  The union is NOT clipped at the wall, so
# k may exceed 1 slightly.  Definition and method: sweep_coverage.py
COVERAGE_ENABLED = True
COVERAGE_CELL    = 4.0    # raster cell (px); 0.04% at n=100, ~5 ms/frame
COVERAGE_EVERY   = 5      # measure every N recorded frames (60 fps -> 12 Hz)

# ── Ring (confining boundary) options ─────────────────────────────────────────
# The boundary can be FIXED (anchored to the world) or MOVABLE (free to move),
# and a smooth CIRCLE or a regular N-gon whose corners are free-rotating hinges.
#
#   RING_MOVABLE = False, RING_SHAPE = "circle"  →  the original fixed circular
#   ring, reproduced exactly.  Any other combination is a new variant.
#
# Combinations:
#   circle  + fixed    : original smooth static wall.
#   circle  + movable  : one rigid ring body, free to translate & rotate.
#   polygon + fixed    : regular n-gon wall held in place at its corners.
#   polygon + movable  : n rigid edge-links joined by free-rotating corner
#                        hinge joints — a deformable loop the swarm can reshape.
RING_MOVABLE   = True        # False = fixed (default); True = movable
RING_SHAPE     = "polygon"     # "circle" (default) | "polygon"

# --- Ring scaling with population -------------------------------------------
# Both knobs are exactly 1.0 at N_SMARTICLES == BASE_N_REF, so the reference
# experiment is untouched; they only matter once the arena grows.
#
# Sides: with a fixed side count a bigger ring means longer edges, so the wall
# would be geometrically *coarser* relative to a robot at large N (edge length
# 44 px at N=17 vs 106 px at N=100).  Scaling the count with the radius keeps
# the edge length — and hence what a robot "sees" locally — constant.
RING_N_SIDES_BASE     = 35
AUTO_SCALE_RING_SIDES = True
RING_N_SIDES   = (int(round(RING_N_SIDES_BASE * _POP_SCALE))
                  if AUTO_SCALE_RING_SIDES else RING_N_SIDES_BASE)

# Mass: a MOVABLE ring is pushed by the swarm, so what determines the regime is
# the ring-mass : swarm-mass ratio (0.34 at the reference settings).  Holding
# RING_MASS fixed while N grows makes the boundary ~N times easier to shove,
# which is a different experiment.  Scaling by _POP_SCALE**2 (== N/BASE_N_REF)
# preserves the ratio.
#   *** This is a physics judgement call -- see notes.  Set to False to keep the
#   *** old constant mass, or use _POP_SCALE (hoop of fixed linear density).
# Ignored when RING_MOVABLE is False.  Heavier = harder for the swarm to shove.
BASE_RING_MASS        = 1000.0
AUTO_SCALE_RING_MASS  = True
RING_MASS      = (BASE_RING_MASS * (SCALE ** 2)
                  * ((_POP_SCALE ** 2) if AUTO_SCALE_RING_MASS else 1.0))
# Number of color bands painted around the ring, for observing rotation/translation.
# None or 0  →  no special coloring (pymunk default gray/blue).
# Positive N  →  divide the ring into N equal angular bands, each a distinct
#                high-contrast color, cycling through a fixed palette.
# Good values: 2 (half-and-half), 4 (quadrants), 6, 8.
# Works with all shape / movable combinations.
RING_COLOR_BANDS = 4

# ── Actuation ─────────────────────────────────────────────────────────────────
# === THIS IS WHERE YOU CONTROL JOINT MOTION ===
# Desired joint angle trajectory: theta(t) = A * sin(omega * t + phase)
# Left arm  uses A1, omega1, phase1 (set per-robot in run_trial via COMMAND_ARRAY)
# Right arm uses A2, omega2, phase2
A_DEG_NOM         = 30.0    # nominal amplitude (deg) [kept for back-compat]
A_DEG_NOM1        = 90.0    # left  arm amplitude (deg)
A_DEG_NOM2        = 90.0    # right arm amplitude (deg)
OMEGA_NOM         = 10.0    # nominal frequency (rad/s) [kept for back-compat]
OMEGA_NOM1        = 3 * 2 * math.pi   # left  arm frequency (rad/s)
OMEGA_NOM2        = 3 * 2 * math.pi   # right arm frequency (rad/s)
# RATE_LIM          = 8.0     # motor velocity clamp (rad/s)
# ANG_DAMP          = 0.965   # per-step angular velocity damping factor
# SPACE_DAMP        = 0.985   # velocity reserved each step
# JOINT_LIMIT_DEG   = 85.0    # hard joint angle limit (deg)
# KP_NOM            = 30.0    # proportional gain for motor PD controller

RATE_LIM          = 8.0     # motor velocity clamp (rad/s)
ANG_DAMP          = 0.8     # per-step angular velocity damping factor
LIN_DAMP          = 0.8     # per-step linear velocity damping factor

# A movable ring (RING_MOVABLE = True) previously had NO per-step damping of
# its own -- only the global space.damping (SPACE_DAMP, currently 1.0 = no
# decay).  Once hit, it would coast/spin with no floor drag, unlike every
# robot main_body, which gets LIN_DAMP / ANG_DAMP each step.  This applies the
# SAME per-step damping to the ring (its single body if RING_SHAPE="circle",
# or every edge body if RING_SHAPE="polygon").
#
# *** Behaviour change: any existing RING_MOVABLE=True run will now produce a
# *** different trajectory than before (the ring no longer coasts freely).
# *** RING_MOVABLE=False runs are completely unaffected (nothing to damp).
# Set to False to restore the old undamped-ring behaviour.
RING_DAMPING_ENABLED = True
SPACE_DAMP        = 1.0     # velocity reserved each step
JOINT_LIMIT_DEG   = 95.0    # hard joint angle limit (deg)
KP_NOM            = 60.0    # proportional gain for motor PD controller


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
# Layout used by spawn_smarticles_auto() (GetSpawnPositions.py and the
# ALREADY_SPWANED = False path).
#   "legacy" : the original 8-outer-ring + 1-inner-ring geometric layout.
#              It only has two rings, so it cannot place more than ~20 robots
#              without piling the remainder onto a single small circle.
#   "rings"  : Vogel (sunflower) placement — near-uniform density at any N, and
#              with AUTO_SCALE_ARENA the neighbour spacing is N-independent.
#   "auto"   : "legacy" for N <= BASE_N_REF, "rings" above it.  Default, and
#              bit-identical to the old behaviour at the reference population.
SPAWN_LAYOUT      = "auto"
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

# ── Performance / scale-out ───────────────────────────────────────────────────
# Number of trials run concurrently in simulation.main().  1 = the original
# serial behaviour.  Each worker is a separate process with its own physics
# space, so this scales trial throughput (not the speed of a single trial).
PARALLEL_WORKERS  = 1

# Encode the video at 1/VIDEO_DOWNSCALE resolution.
#   "auto" : 1 at N <= BASE_N_REF (unchanged), then follows the arena growth so
#            the encoded frame stays roughly the reference 900x760 regardless of
#            N.  Without this, N=100 writes 2183x1843 frames -- video encoding
#            becomes ~60% of the runtime and a single 60 s trial produces a
#            ~560 MB .mp4.
#   integer: fixed factor (1 = full resolution).
VIDEO_DOWNSCALE   = "auto"
_VIDEO_DOWNSCALE_RAW = VIDEO_DOWNSCALE
VIDEO_DOWNSCALE   = (max(1, int(round(_POP_SCALE)))
                     if str(_VIDEO_DOWNSCALE_RAW).lower() == "auto"
                     else max(1, int(_VIDEO_DOWNSCALE_RAW)))

# ── Misc ──────────────────────────────────────────────────────────────────────
SCORE_VALID         = False
SAVE_NPY            = False
ALREADY_SPWANED     = True
rho                 = N_SMARTICLES / (W * H)  # number density

# =============================================================================
# End of global configuration
# =============================================================================

"""
config.py  ─  All global parameters for the Smarticle simulation.
Edit this file to change experiment settings.

Batch comparison experiments: write the parameters to change into an override
file and point to it via the SMARTICLE_CONFIG environment variable; this file
reads it **before** computing derived quantities. experiments.py drives a set
of conditions exactly this way.

    SMARTICLE_CONFIG=conditions/n50_edge.py  python simulation.py

The override file can be a .py (a series of `NAME = value` lines, same syntax
as this file — tuples/None are all fine) or a .json. Parameters wrapped by
_ov() (N_SMARTICLES, BASE_* geometry, RING_*, etc.) take effect before derived
quantities are computed; the rest are applied uniformly at the end of the file.

SMARTICLE_CONFIG accepts only **one** file — this file executes once on
import, so one process is one experiment. To run a series of configs in
sequence, hand it to the driver:

    python experiments.py --configs conditions/*.py

Two conveniences:

* COMMAND_ARRAY can be written as a shorthand string instead of a hand-written
  list of length N:
      COMMAND_ARRAY = "a-462"             # everyone -462
      COMMAND_ARRAY = "a462; 0..8-851"    # ids 0~8 differ
  See gait.py for the syntax. For a simple ratio mix of two commands, use
  CMD_A / CMD_B / CMD_A_FRACTION instead.

* **config_snapshot.json can be used directly as an override file** ("re-run
  that same experiment"): it carries the _snapshot marker, so its derived
  quantities are automatically skipped and recomputed from the input
  parameters.
      SMARTICLE_CONFIG=datafile/xxx/config_snapshot.json python simulation.py
  For a trimmed, easy-to-hand-edit version:
      python experiments.py --from-snapshot datafile/xxx/config_snapshot.json

**N_SMARTICLES can only be swept by switching processes**: smarticle.py /
spawn.py / analysis.py all do `from config import MAIN_LEN, ...`, which binds
by value, so changing N within a process does not recompute them.
experiments.py therefore spawns a new interpreter per condition.
"""

import json
import math
import os
import random

# ── Override file ─────────────────────────────────────────────────────────
def _load_override_file(path):
    """
    Read one override file, return (settings, meta). .json goes through json;
    everything else is exec'd as Python source.

    settings are the parameters to apply; meta is metadata whose keys start
    with an underscore — in particular _snapshot means this file was written
    by save_config_snapshot(), which also records the derived quantities, so
    those must be skipped when applying it (see the end of this file).

    utf-8-sig: Notepad and PowerShell's Set-Content on Windows both write a
    BOM; reading it as plain utf-8 makes the first character U+FEFF, which
    exec chokes on as a syntax error. utf-8-sig is fully equivalent to utf-8
    when there is no BOM.

    experiments.py also uses this function to read the series of config files
    the user supplies, so the overrides the driver sees are guaranteed to
    match what config.py itself reads.
    """
    if path.lower().endswith(".json"):
        with open(path, encoding="utf-8-sig") as f:
            raw = json.load(f)
    else:
        raw = {}
        exec_ns = {}
        with open(path, encoding="utf-8-sig") as f:
            exec(compile(f.read(), path, "exec"), exec_ns)
        raw = {k: v for k, v in exec_ns.items() if k != "__builtins__"}
    # Underscore-prefixed keys are metadata (config_snapshot.json's _snapshot /
    # _skipped, or a scratch _tmp variable used inside the override file
    # itself), not parameters to apply
    settings = {k: v for k, v in raw.items() if not k.startswith("_")}
    meta = {k: v for k, v in raw.items() if k.startswith("_")}
    return settings, meta


# SMARTICLE_CONFIG accepts only **one** file: config executes once on import,
# so one process is one experiment. To run a series of configs in sequence,
# use experiments.py --configs a.py b.py.
_OVERRIDES = {}
_OV_PATH = os.environ.get("SMARTICLE_CONFIG", "").strip()
if _OV_PATH:
    if os.pathsep in _OV_PATH or "," in _OV_PATH:
        raise ValueError(
            f"SMARTICLE_CONFIG can only be a single file, got {_OV_PATH!r}. "
            f"To run multiple configs in sequence, use: "
            f"python experiments.py --configs a.py b.py ...")
    _OVERRIDES, _OV_META = _load_override_file(_OV_PATH)
    _FROM_SNAPSHOT = "_snapshot" in _OV_META
    print(f"[config] overrides from {_OV_PATH}"
          f"{' (config_snapshot)' if _FROM_SNAPSHOT else ''}: "
          f"{len(_OVERRIDES)} settings")
else:
    _OV_META, _FROM_SNAPSHOT = {}, False

_OV_USED = set()


def _ov(name, default):
    """Parameters that must be settled before derived quantities go through
    here; falls back to the default when no override value is given."""
    if name in _OVERRIDES:
        _OV_USED.add(name)
        return _OVERRIDES[name]
    return default

# =============================================================================
# Global Configuration  ← all tunable parameters are defined here
# =============================================================================

# ── Trial / seed ──────────────────────────────────────────────────────────────
TRIAL_SEED_BASE   = 12345
N_TRIALS_GLOBAL   = _ov("N_TRIALS_GLOBAL", 10)       # 0 means auto-read from initial-conditions file
MAX_RUNTIME       = _ov("MAX_RUNTIME", 500.0)    # in seconds

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
N_SMARTICLES      = _ov("N_SMARTICLES", 100)         # number of smarticles in the simulation

# ── Reference geometry (used for auto-scaling) ────────────────────────────────
BASE_N_REF        = _ov("BASE_N_REF", 17)         # population the reference arena was tuned for
BASE_WALL_THICK   = _ov("BASE_WALL_THICK", 20)
BASE_MAIN_LEN = _ov("BASE_MAIN_LEN", 70)
BASE_MAIN_W   = _ov("BASE_MAIN_W", 41)
BASE_ARM_LEN  = _ov("BASE_ARM_LEN", 70)
BASE_ARM_W    = _ov("BASE_ARM_W", 6)

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
AUTO_SCALE_ARENA  = _ov("AUTO_SCALE_ARENA", True)
_POP_SCALE        = (max(1.0, math.sqrt(N_SMARTICLES / BASE_N_REF))
                     if AUTO_SCALE_ARENA else 1.0)

# ── Screen / geometry ─────────────────────────────────────────────────────────
BASE_W = _ov("BASE_W", 900); BASE_H = _ov("BASE_H", 760)    # reference screen size (pixels) at BASE_N_REF
W, H              = int(round(BASE_W * _POP_SCALE)), int(round(BASE_H * _POP_SCALE))
SCREEN_MARGIN     = _ov("SCREEN_MARGIN", 26)         # margin between outer wall and window edge

BASE_INNER_R_UNSCALED = _ov("BASE_INNER_R_UNSCALED", 245)    # inner ring radius at BASE_N_REF, before scaling
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
ENABLE_HETEROGENEOUS_BODIES = _ov("ENABLE_HETEROGENEOUS_BODIES", False)

# Library of reusable body types ("species"). "default" = the global geometry.
BODY_TYPES = _ov("BODY_TYPES", {
    "default":   {},                       # global BASE_* geometry, unchanged
    "long_arm":  {"arm_len_base": 110},    # longer arms (base 70 -> 110)
    "short_arm": {"arm_len_base": 45},     # shorter arms (base 70 -> 40)
    # "heavy":   {"main_w_base": 60, "mass_main": 400.0},
})

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
_CMD_A = _ov("CMD_A", 432); _CMD_B = _ov("CMD_B", 832)
_CMD_A_FRACTION   = _ov("CMD_A_FRACTION", 0 / 17)
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
    #     "n_per_end":      how many robots to take from each end (default 1)
    #     "axis":           "major" / "minor" / "both"
    #     "min_anisotropy": below this aspect ratio, direction is judged meaningless and no one is selected that frame (default 1.5)
    #     "min_group_size": below this size the largest group selects no one (group_*_ends)
    #     "select_from":    "all" selects endpoints across the whole field (default) / "group" selects only within that group
    # "farthest" uses "n"; extremes / convex_hull and the like take no extra parameters.
    #
    # "override_group": True promotes this role **above** the grouping layer --
    # default False, meaning once a robot is claimed by a group_rule it runs
    # the group's command and the role layer has no say over it.
    "roles": [
        # {"selector": "convex_hull", "command": -52,
        #  "n_frames_join": 18, "n_frames_leave": 18},
        {"selector": "group_major_ends", "command": -52, "min_group_size": 3,
         "n_per_end": 6, "override_group": False},
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
RING_MOVABLE   = _ov("RING_MOVABLE", True)        # False = fixed (default); True = movable
RING_SHAPE     = _ov("RING_SHAPE", "polygon")     # "circle" (default) | "polygon"

# --- Ring scaling with population -------------------------------------------
# Both knobs are exactly 1.0 at N_SMARTICLES == BASE_N_REF, so the reference
# experiment is untouched; they only matter once the arena grows.
#
# Sides: with a fixed side count a bigger ring means longer edges, so the wall
# would be geometrically *coarser* relative to a robot at large N (edge length
# 44 px at N=17 vs 106 px at N=100).  Scaling the count with the radius keeps
# the edge length — and hence what a robot "sees" locally — constant.
RING_N_SIDES_BASE     = _ov("RING_N_SIDES_BASE", 35)
AUTO_SCALE_RING_SIDES = _ov("AUTO_SCALE_RING_SIDES", True)
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
BASE_RING_MASS        = _ov("BASE_RING_MASS", 1000.0)
AUTO_SCALE_RING_MASS  = _ov("AUTO_SCALE_RING_MASS", True)
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
SPAWN_LAYOUT      = _ov("SPAWN_LAYOUT", "auto")
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
ALREADY_SPWANED     = _ov("ALREADY_SPWANED", True)
rho                 = N_SMARTICLES / (W * H)  # number density

# =============================================================================
# End of global configuration
# =============================================================================


# =============================================================================
# Run-level settings (previously hard-coded in simulation.main())
# =============================================================================
# Placed here so a single override file can fully determine one run -- in a
# batch comparison experiment, each condition has its own initial-conditions
# file and output location.
INIT_FILE = "init_conditions/init_conditions_200_p.json"
# Note: the robot count in the IC file must equal N_SMARTICLES; run_trial checks this on the spot.
#   *_p_N17.json -> 17 robots   *_p_N50.json -> 50 robots   *_p.json -> 100 robots
EXP_NAME  = None      # None = auto-generated by naming.generate_trial_name
OUT_ROOT  = "datafile"   # output root directory; each experiment lands under <OUT_ROOT>/<EXP_NAME>/
PREVIEW    = False
USE_PRESET = True


# =============================================================================
# Apply the remaining overrides
# =============================================================================
# Parameters wrapped by _ov() already took effect above (derived quantities
# depend on them, so it has to happen early); this handles the rest.
# Derived quantities themselves should never be overridden -- doing so would
# only produce a self-contradictory config (e.g. changing MAIN_LEN while L_s
# is still computed from the old value), so this raises rather than letting
# it silently apply.
_DERIVED = {
    "W", "H", "SCALE", "INNER_R", "INNER_R_UNSCALED", "WALL_THICK",
    "WALL_SEGMENTS", "MAIN_LEN", "MAIN_W", "ARM_LEN", "ARM_W",
    "RING_N_SIDES", "RING_MASS", "L", "L_s", "S", "MASS_MAIN", "MASS_ARM",
}

_skipped_derived = []
for _k, _v in _OVERRIDES.items():
    if _k in _OV_USED:
        continue                      # already took effect at its definition site
    if _k in _DERIVED:
        # config_snapshot.json is a full record, so derived quantities are naturally
        # in it too. Using a snapshot directly as an override file is a natural need
        # ("re-run that same experiment"), so snapshots skip derived quantities rather
        # than raising -- they'll be recomputed to the same values from the input
        # parameters above. A hand-written override file still raises: that's most
        # likely a genuine mistake.
        if _FROM_SNAPSHOT:
            _skipped_derived.append(_k)
            continue
        raise ValueError(
            f"{_k!r} in the override file is a derived quantity and can't be changed directly -- it's computed from N_SMARTICLES / "
            f"BASE_*. Change those input parameters instead."
            f"(If this is config_snapshot.json, it's missing the _snapshot marker, "
            f"possibly written by an old version: convert it with experiments.py --from-snapshot)")
    if _k not in globals():
        raise ValueError(
            f"{_k!r} in the override file doesn't exist in config.py (typo?)")
    globals()[_k] = _v
if _skipped_derived:
    print(f"[config] Skipped {len(_skipped_derived)} derived quantities from the snapshot"
          f"(recomputed from input parameters): {', '.join(sorted(_skipped_derived)[:6])}"
          f"{' ...' if len(_skipped_derived) > 6 else ''}")


# =============================================================================
# COMMAND_ARRAY may be written as a shorthand string
# =============================================================================
# Hand-writing a length-N list is painful, especially at N=100. Allow the
# gait.py debug shorthand instead:
#     COMMAND_ARRAY = "a-462"            # everyone -462
#     COMMAND_ARRAY = "a462; 0..8-851"   # mostly 462, ids 0~8 are -851
# gait.py only depends on naming.py, not the other way around (it never
# imports config), so it's safe to use it here.
if isinstance(COMMAND_ARRAY, str):
    from gait import build_command_array
    _spec = COMMAND_ARRAY
    COMMAND_ARRAY = build_command_array(_spec, N_SMARTICLES)
    print(f"[config] COMMAND_ARRAY = {_spec!r} -> {N_SMARTICLES} commands")
elif len(COMMAND_ARRAY) != N_SMARTICLES:
    # Without this check it fails as an IndexError deep inside gait.GaitController, obscuring the cause
    raise ValueError(
        f"COMMAND_ARRAY has {len(COMMAND_ARRAY)} entries, but N_SMARTICLES="
        f"{N_SMARTICLES}. Either write it as a shorthand string (e.g. \"a-462\") to let it auto-expand, "
        f"or use CMD_A / CMD_B / CMD_A_FRACTION instead.")

# BODY_ASSIGNMENT likewise: length must match, otherwise bodies.py only reports it halfway through
if len(BODY_ASSIGNMENT) != N_SMARTICLES:
    raise ValueError(
        f"BODY_ASSIGNMENT has {len(BODY_ASSIGNMENT)} entries, but N_SMARTICLES="
        f"{N_SMARTICLES}.")

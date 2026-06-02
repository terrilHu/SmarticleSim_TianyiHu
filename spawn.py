"""
spawn.py  ─  Smarticle placement (spawn) functions, packing checks,
             and initial-condition I/O.
"""

import json
import math
import random

import pymunk

from config import (
    MAIN_LEN, MAIN_W, ARM_LEN,
    PEN_EPS, SETTLE_STEPS, SETTLE_DT,
    MAX_MOTOR_TORQUE_NOM,
    INIT_MIX_FOLDED, INIT_MIX_STRAIGHT, INIT_MIX_UNIFORM, STRAIGHT_BAND_DEG,
    ENABLE_PARAMETER_JITTER,
    A_DEG_NOM1, A_DEG_NOM2, OMEGA_NOM1, OMEGA_NOM2, KP_NOM,
    A_JITTER_FRAC, OMEGA_JITTER_FRAC, KP_JITTER_FRAC, TORQUE_JITTER_FRAC,
)
from smarticle import Smarticle3Link, rot


# =============================================================================
# Small helpers
# =============================================================================

def random_point_in_disk(center: pymunk.Vec2d, radius: float) -> pymunk.Vec2d:
    ang = random.uniform(0.0, 2.0 * math.pi)
    r   = radius * math.sqrt(random.random())
    return center + pymunk.Vec2d(r * math.cos(ang), r * math.sin(ang))


def settle_space(space: pymunk.Space,
                 steps: int = SETTLE_STEPS, dt: float = SETTLE_DT):
    """Advance a few tiny steps so broadphase/contact info becomes reliable."""
    for _ in range(steps):
        space.step(dt)


def jitter(value: float, frac: float) -> float:
    """Multiply by (1 + uniform(-frac, +frac))."""
    return value * (1.0 + random.uniform(-frac, frac))


def relax_system(space: pymunk.Space, steps: int = 300, dt: float = 1 / 240.0):
    old_damping  = space.damping
    space.damping = 0.85
    for _ in range(steps):
        space.step(dt)
    space.damping = old_damping


# =============================================================================
# Packing checks
# =============================================================================

def inside_ring(candidate: Smarticle3Link, center: pymunk.Vec2d,
                inner_r: float, margin: float = 0.0) -> bool:
    R = inner_r - margin
    for sh in candidate.shapes():
        for v in sh.get_vertices():
            w = sh.body.local_to_world(v)
            if (w - center).length > R:
                return False
    return True


def any_penetration(space: pymunk.Space, candidate: Smarticle3Link,
                    pen_eps: float = PEN_EPS) -> bool:
    cand_shapes = candidate.shapes()
    for cs in cand_shapes:
        for info in space.shape_query(cs):
            if info.shape in cand_shapes:
                continue
            for p in info.contact_point_set.points:
                if p.distance < -3.0:
                    return True
    return False


def penetration_score(space: pymunk.Space, candidate: Smarticle3Link,
                      pen_eps: float = PEN_EPS) -> float:
    """Return total penetration depth (0 = no overlap)."""
    total       = 0.0
    cand_shapes = candidate.shapes()
    for cs in cand_shapes:
        for info in space.shape_query(cs):
            if info.shape in cand_shapes:
                continue
            for p in info.contact_point_set.points:
                if p.distance < -pen_eps:
                    total += (-p.distance)
    return total


# =============================================================================
# Joint-angle samplers
# =============================================================================

def sample_initial_joint_angle(lim: float, compact_bias: float) -> float:
    w_fold     = min(0.85, INIT_MIX_FOLDED   + 0.35 * compact_bias)
    w_straight = max(0.05, INIT_MIX_STRAIGHT - 0.20 * compact_bias)
    w_unif     = max(0.05, 1.0 - (w_fold + w_straight))
    s = w_fold + w_straight + w_unif
    w_fold, w_straight, w_unif = w_fold / s, w_straight / s, w_unif / s

    u = random.random()
    if u < w_fold:
        sgn = 1 if random.random() < 0.5 else -1
        return sgn * (lim - math.radians(random.uniform(2, 25)))
    elif u < w_fold + w_straight:
        band = math.radians(STRAIGHT_BAND_DEG)
        return random.uniform(-band, band)
    else:
        return random.uniform(-lim, lim)


def sample_compact_joint_pair(lim: float):
    low  = math.radians(60.0)
    high = min(lim, math.radians(90.0))
    def one():
        return (1 if random.random() < 0.5 else -1) * random.uniform(low, high)
    return one(), one()


def sample_tight_joint_pair(lim: float, max_deg: float = 15.0):
    max_rad = math.radians(max_deg)
    return random.uniform(-max_rad, max_rad), random.uniform(-max_rad, max_rad)


# =============================================================================
# Candidate factory
# =============================================================================

def make_candidate(space: pymunk.Space, mode: str = "normal",
                   phase1: float = None, phase2: float = None) -> Smarticle3Link:
    ph1 = phase1 if phase1 is not None else 0.0
    ph2 = phase2 if phase2 is not None else 0.0

    A_DEG_FINAL1 = (2 * A_DEG_NOM1) if mode == "booster" else A_DEG_NOM1
    A_DEG_FINAL2 = (2 * A_DEG_NOM2) if mode == "booster" else A_DEG_NOM2

    if ENABLE_PARAMETER_JITTER:
        A_deg1 = jitter(A_DEG_FINAL1, A_JITTER_FRAC)
        A_deg2 = jitter(A_DEG_FINAL2, A_JITTER_FRAC)
        omega1 = jitter(OMEGA_NOM1,   OMEGA_JITTER_FRAC)
        omega2 = jitter(OMEGA_NOM2,   OMEGA_JITTER_FRAC)
        kp     = jitter(KP_NOM,       KP_JITTER_FRAC)
        torque = jitter(MAX_MOTOR_TORQUE_NOM, TORQUE_JITTER_FRAC)
    else:
        A_deg1, A_deg2 = A_DEG_FINAL1, A_DEG_FINAL2
        omega1, omega2 = OMEGA_NOM1, OMEGA_NOM2
        kp, torque     = KP_NOM, MAX_MOTOR_TORQUE_NOM

    sm = Smarticle3Link(
        space=space, kp=kp, max_torque=torque,
        A_deg1=A_deg1, A_deg2=A_deg2,
        omega1=omega1, omega2=omega2,
        phase1=ph1,    phase2=ph2,
    )
    try:
        sm.main_shape.color = (255, 0, 0, 255) if mode == "booster" else (100, 100, 255, 255)
    except Exception:
        pass
    return sm


# =============================================================================
# Spawn functions
# =============================================================================

def spawn_smarticles(space: pymunk.Space, center: pymunk.Vec2d,
                     inner_r: float, n: int) -> list:
    """
    Geometric 8+inner layout with random jitter, followed by a physics relax.
    High success rate; good diversity via relax.
    """
    smarts      = []
    n_outer     = min(n, 8)
    n_inner     = n - n_outer
    r_outer     = inner_r * 0.50
    r_inner     = inner_r * 0.15
    g_ang_outer = random.uniform(0, 2 * math.pi)
    g_ang_inner = random.uniform(0, 2 * math.pi)

    for i in range(n):
        r_jitter     = random.uniform(-0.02, 0.02) * inner_r
        angle_jitter = random.uniform(-0.1, 0.1)

        if i < n_outer:
            angle = g_ang_outer + i * (2 * math.pi / n_outer) + angle_jitter
            r     = r_outer + r_jitter
        else:
            angle = (g_ang_inner + (i - n_outer) * (2 * math.pi / n_inner) + angle_jitter
                     if n_inner > 0 else 0)
            r     = r_inner + r_jitter

        target_pos = center + pymunk.Vec2d(r * math.cos(angle), r * math.sin(angle))
        heading    = angle + math.pi / 2 + random.uniform(-0.2, 0.2)

        sm     = Smarticle3Link(space)
        offset = target_pos - sm.main_body.position
        sm.main_body.position  += offset
        sm.left_body.position  += offset
        sm.right_body.position += offset
        sm.main_body.angle      = heading

        fold = random.uniform(-math.pi / 6, math.pi / 6)
        sm.left_body.angle  = heading + fold
        sm.right_body.angle = heading - fold
        smarts.append(sm)

    relax_system(space, steps=600)
    return smarts


def spawn_smarticles_norelax(space: pymunk.Space, center: pymunk.Vec2d,
                              inner_r: float, n: int) -> list:
    """
    Place all robots at once (allowing initial overlap), then run a position-only
    relax (arms included) and restore joint angles afterwards.
    """
    FOLD_MIN = math.radians(10.0)
    FOLD_MAX = math.radians(80.0)

    smarts     = []
    init_folds = []
    spawn_max_r = inner_r * 0.82

    # ── Step 1: place all robots ──────────────────────────────────────────────
    for _ in range(n):
        u          = random.random()
        r          = spawn_max_r * math.sqrt(u)
        angle      = random.uniform(0, 2 * math.pi)
        target_pos = center + pymunk.Vec2d(r * math.cos(angle), r * math.sin(angle))
        heading    = random.uniform(0, 2 * math.pi)
        fold       = (1 if random.random() < 0.5 else -1) * random.uniform(FOLD_MIN, FOLD_MAX)

        sm = Smarticle3Link(space)
        sm.set_folded_pose(target_pos, heading, fold, -fold)
        for b in sm.bodies():
            space.reindex_shapes_for_body(b)
        smarts.append(sm)
        init_folds.append(fold)

    # ── Step 2: prepare relax (soften joints, enable arm collisions) ──────────
    for sm in smarts:
        sm.joint_L.collide_bodies = True
        sm.joint_R.collide_bodies = True
        sm.joint_L.max_bias       = 0.0
        sm.joint_R.max_bias       = 0.0
        sm.limit_L.max_bias       = 0.0
        sm.limit_R.max_bias       = 0.0
        sm.motor_L.max_force      = 0.0
        sm.motor_R.max_force      = 0.0

    # ── Step 3: relax ─────────────────────────────────────────────────────────
    old_damping      = space.damping
    old_iterations   = space.iterations
    space.damping    = 0.60
    space.iterations = 20
    for _ in range(500):
        space.step(1 / 300.0)
    space.damping    = old_damping
    space.iterations = old_iterations

    # ── Step 4: restore joint geometry using post-relax pose ─────────────────
    for sm, fold in zip(smarts, init_folds):
        sm.set_folded_pose(sm.main_body.position, sm.main_body.angle, fold, -fold)
        for b in sm.bodies():
            space.reindex_shapes_for_body(b)

    # ── Step 5: restore constraints and motors ────────────────────────────────
    for sm in smarts:
        sm.joint_L.collide_bodies = False
        sm.joint_R.collide_bodies = False
        sm.joint_L.max_bias       = 10.0
        sm.joint_R.max_bias       = 10.0
        sm.limit_L.max_bias       = 10.0
        sm.limit_R.max_bias       = 10.0
        sm.motor_L.max_force      = MAX_MOTOR_TORQUE_NOM
        sm.motor_R.max_force      = MAX_MOTOR_TORQUE_NOM
        sm.motor_L.rate           = 0.0
        sm.motor_R.rate           = 0.0

    # ── Step 6: validate ──────────────────────────────────────────────────────
    out = [i for i, sm in enumerate(smarts) if not inside_ring(sm, center, inner_r)]
    if out:
        print(f"[spawn] WARNING: {len(out)}/{n} robots outside ring after relax: {out}")
    else:
        print(f"[spawn] SUCCESS: all {n}/{n} robots inside ring.")
    return smarts


def spawn_fallback_guaranteed(space: pymunk.Space, center: pymunk.Vec2d,
                               inner_r: float, n: int) -> list:
    """Systematic concentric-ring placement; returns however many fit."""
    smarts          = []
    ring_fracs      = [0.0, 0.15, 0.28, 0.40, 0.52, 0.64, 0.76, 0.88]
    candidate_points = []

    for frac in ring_fracs:
        r = inner_r * frac
        m = max(10, int(2 * math.pi * r / max(10.0, MAIN_W * 0.55)))
        for j in range(m):
            a = 2 * math.pi * j / m
            candidate_points.append((r, a, center + pymunk.Vec2d(r * math.cos(a), r * math.sin(a))))
    candidate_points.sort(key=lambda x: x[0])

    for _, _, pos in candidate_points:
        if len(smarts) >= n:
            break
        for ang in [0.0, math.pi/2, math.pi, -math.pi/2, math.pi/4, -math.pi/4]:
            cand = make_candidate(space)
            thL  = random.uniform(-cand.lim, cand.lim)
            thR  = random.uniform(-cand.lim, cand.lim)
            cand.set_folded_pose(pos, ang, thL, thR)
            settle_space(space)
            if inside_ring(cand, center, inner_r) and not any_penetration(space, cand):
                smarts.append(cand)
                break
            cand.remove_from_space()
    return smarts


def spawn_outside_in(space: pymunk.Space, center: pymunk.Vec2d,
                     inner_r: float, n: int) -> list:
    """Place robots from outermost ring inwards."""
    smarts = []
    for ratio in [0.75, 0.45, 0.15, 0.0]:
        if len(smarts) >= n:
            break
        r     = inner_r * ratio
        slots = max(1, int(2 * math.pi * r / (MAIN_LEN * 1.0))) if r > 0.01 * inner_r else 1
        start = random.uniform(0, 2 * math.pi)
        for i in range(slots):
            if len(smarts) >= n:
                break
            angle  = start + 2 * math.pi * i / slots
            pos    = center + pymunk.Vec2d(r * math.cos(angle), r * math.sin(angle))
            tang   = angle + math.pi / 2
            for off in [0, 0.2, -0.2, math.pi/4, -math.pi/4]:
                cand = make_candidate(space)
                fold = (1 if random.random() < 0.5 else -1) * math.radians(80.0)
                cand.set_folded_pose(pos, tang + off, fold, fold)
                for body in cand.bodies():
                    space.reindex_shapes_for_body(body)
                space.step(1e-5)
                if inside_ring(cand, center, inner_r) and not any_penetration(space, cand):
                    smarts.append(cand)
                    break
                cand.remove_from_space()
    return smarts


def spawn_space_filling(space: pymunk.Space, center: pymunk.Vec2d,
                        inner_r: float, n: int, samples: int = 800) -> list:
    """Max-clearance placement: each robot goes to the emptiest spot found."""
    smarts  = []
    spawn_r = inner_r - (MAIN_LEN / 2.0 + 2.0)

    for _ in range(n):
        best_pos, best_clr = None, -1
        for _ in range(samples):
            pos   = random_point_in_disk(center, spawn_r)
            d_wall = inner_r - (pos - center).length
            d_sm   = min((pos - s.main_body.position).length for s in smarts) if smarts else d_wall
            clr    = min(d_wall, d_sm)
            if clr > best_clr:
                best_clr, best_pos = clr, pos

        best_cand, best_score = None, float("inf")
        for _ in range(60):
            cand = make_candidate(space)
            thL  = random.uniform(-cand.lim, cand.lim)
            thR  = random.uniform(-cand.lim, cand.lim)
            cand.set_folded_pose(best_pos, random.uniform(-math.pi, math.pi), thL, thR)
            settle_space(space)
            if not inside_ring(cand, center, inner_r):
                cand.remove_from_space()
                continue
            if any_penetration(space, cand):
                score = 1e9
            else:
                score = min((cand.main_body.position - s.main_body.position).length
                            for s in smarts) if smarts else 0.0
            if score < best_score:
                if best_cand is not None:
                    best_cand.remove_from_space()
                best_cand, best_score = cand, score
            else:
                cand.remove_from_space()

        if best_cand is not None:
            smarts.append(best_cand)
        else:
            break
    return smarts


# =============================================================================
# Initial-condition I/O
# =============================================================================

def load_initial_conditions(path: str) -> list:
    with open(path, "r") as f:
        return json.load(f)


def build_from_initial_conditions(space: pymunk.Space, trial_data: dict) -> list:
    smarts = []
    for sm_data in trial_data["smarticles"]:
        sm  = make_candidate(space, mode="normal")
        pos = pymunk.Vec2d(sm_data["pos"][0], sm_data["pos"][1])
        sm.set_folded_pose(pos, sm_data["angle"], sm_data["thL"], sm_data["thR"])
        smarts.append(sm)
    return smarts

"""
smarticle.py  ─  Smarticle3Link robot class and low-level physics helpers.
"""

import math
import pymunk

from config import (
    MAIN_LEN, MAIN_W, ARM_LEN, ARM_W,
    MASS_MAIN, MASS_ARM,
    KP_NOM, MAX_MOTOR_TORQUE_NOM,
    A_DEG_NOM1, A_DEG_NOM2,
    OMEGA_NOM1, OMEGA_NOM2,
    JOINT_LIMIT_DEG,
    WALL_SEGMENTS, WALL_FRICTION, WALL_ELASTICITY,
    RING_MASS, RING_COLOR_BANDS,
)

from bodies import arm_colors

# =============================================================================
# Math helpers
# =============================================================================

def wrap_pi(a: float) -> float:
    """Wrap angle to (-pi, pi]."""
    return (a + math.pi) % (2 * math.pi) - math.pi


def rot(v: pymunk.Vec2d, ang: float) -> pymunk.Vec2d:
    """Rotate 2-D vector v by angle ang (radians)."""
    c, s = math.cos(ang), math.sin(ang)
    return pymunk.Vec2d(c * v.x - s * v.y, s * v.x + c * v.y)


# =============================================================================
# Physics body factory
# =============================================================================

def box_body(space: pymunk.Space, pos, angle, size, mass,
             friction=0.9, elasticity=0.0):
    """Create a rectangular rigid body, add it to space, and return (body, shape)."""
    moment = pymunk.moment_for_box(mass, size)
    body   = pymunk.Body(mass, moment)
    body.position = pos
    body.angle    = angle
    shape = pymunk.Poly.create_box(body, size)
    shape.friction    = friction
    shape.elasticity  = elasticity
    space.add(body, shape)
    return body, shape


# =============================================================================
# Ring wall
# =============================================================================

class RingInfo:
    """
    Lightweight handle describing a ring created by :func:`add_ring`.

    Returned so callers can render the boundary (whatever its shape / mobility)
    and inspect its bodies.  Backward compatible: callers that ignore the return
    value are completely unaffected.

    Attributes
    ----------
    shape        : "circle" | "polygon"
    movable      : bool
    center       : pymunk.Vec2d   nominal centre
    inner_r      : float          inner radius / polygon circumradius
    wall_thick   : float
    n_sides      : int or None    (polygon only)
    body         : pymunk.Body or None   single rigid body of a movable circle
    bodies       : list[pymunk.Body]     edge bodies of a polygon (else [])
    segments     : list[pymunk.Segment]  all wall segment shapes
    constraints  : list[pymunk.Constraint]  corner / anchor joints (polygon)
    """

    def __init__(self, shape, center, inner_r, wall_thick, movable,
                 n_sides=None, body=None, bodies=None,
                 segments=None, constraints=None):
        self.shape       = shape
        self.center      = center
        self.inner_r     = inner_r
        self.wall_thick  = wall_thick
        self.movable     = movable
        self.n_sides     = n_sides
        self.body        = body
        self.bodies      = bodies or []
        self.segments    = segments or []
        self.constraints = constraints or []

    def segment_world_endpoints(self):
        """
        Current world-space endpoints of every wall segment, as a list of
        (a, b) Vec2d pairs.  Works for any shape / mobility (the endpoints are
        read live from each segment's body), so it is the safe way to draw the
        boundary even after a movable ring has moved or a polygon has deformed.
        """
        out = []
        for s in self.segments:
            out.append((s.body.local_to_world(s.a),
                        s.body.local_to_world(s.b)))
        return out


# =============================================================================
# Ring color-band palette
# =============================================================================

# High-contrast RGBA palette for ring color bands.  Chosen to be visually
# distinct even when only 2 or 4 bands are used, and to stand out against
# the white background and the blue/gray pymunk default robot colors.
_BAND_PALETTE = [
    (220,  50,  50, 255),   # red
    (230, 140,  20, 255),   # orange
    ( 50, 180,  50, 255),   # green
    ( 30, 120, 220, 255),   # blue
    (160,  40, 200, 255),   # purple
    ( 20, 190, 190, 255),   # teal
    (230, 210,  20, 255),   # yellow
    (220,  80, 160, 255),   # pink
]


def _color_ring_segments(segments, n_bands, angle_fn):
    """
    Assign ``shape.color`` to each segment in ``segments`` based on its
    angular position in the ring.

    Parameters
    ----------
    segments  : list of pymunk.Segment
    n_bands   : int — number of color bands (must be >= 1)
    angle_fn  : callable(seg) -> float
        Returns the representative angle (radians) of a segment in the ring.
        The value only needs to be consistent within [0, 2π); wrapping is
        handled here.
    """
    if not n_bands:
        return
    import pygame
    palette = _BAND_PALETTE
    two_pi  = 2.0 * math.pi
    for s in segments:
        ang      = angle_fn(s) % two_pi          # normalise to [0, 2π)
        band_idx = int(ang / two_pi * n_bands) % n_bands
        r, g, b, a = palette[band_idx % len(palette)]
        s.color  = pygame.Color(r, g, b, a)


def add_ring(space: pymunk.Space, center: pymunk.Vec2d,
             inner_r: float, wall_thick: float,
             segments: int  = WALL_SEGMENTS,
             friction: float = WALL_FRICTION,
             elasticity: float = WALL_ELASTICITY,
             movable: bool = False,
             shape: str = "circle",
             n_sides: int = 6,
             ring_mass: float = None,
             color_bands: int = None) -> "RingInfo":
    """
    Add a confining ring wall to ``space`` and return a :class:`RingInfo` handle.

    Parameters
    ----------
    movable : bool, default False
        ``False`` → fixed (anchored to the world).  ``True`` → movable (the ring
        body/bodies are dynamic and can translate / rotate, and for a polygon,
        deform) when pushed by robots.
    shape : {"circle", "polygon"}, default "circle"
        ``"circle"`` → smooth circular wall of ``segments`` short arcs.
        ``"polygon"`` → regular ``n_sides``-gon with free-rotating corner hinges.
    n_sides : int, default 6
        Number of sides for the polygon (clamped to >= 3).
    ring_mass : float or None
        Total mass of a movable ring.  Ignored when ``movable`` is False.
        Defaults to ``config.RING_MASS``.
    color_bands : int or None
        Number of color bands painted around the ring so that rotation and
        translation are visible at a glance.  ``None`` or ``0`` → no special
        coloring (pymunk default).  A positive integer N divides the ring into N
        equal angular bands, each painted a distinct high-contrast color cycling
        through a fixed palette.  Defaults to ``config.RING_COLOR_BANDS``.

    Backward compatibility
    ----------------------
    ``add_ring(space, center, INNER_R, WALL_THICK)`` with the defaults
    reproduces the original fixed circular ring exactly.
    """
    shape = (shape or "circle").lower()
    if ring_mass is None:
        ring_mass = RING_MASS
    if color_bands is None:
        color_bands = RING_COLOR_BANDS or 0

    if shape == "polygon":
        return _add_polygon_ring(space, center, inner_r, wall_thick, n_sides,
                                 friction, elasticity, movable, ring_mass,
                                 color_bands)
    return _add_circle_ring(space, center, inner_r, wall_thick, segments,
                            friction, elasticity, movable, ring_mass,
                            color_bands)


def _add_circle_ring(space, center, inner_r, wall_thick, segments,
                     friction, elasticity, movable, ring_mass, color_bands):
    """Circular wall of short arc segments — fixed (static) or movable (one rigid body)."""
    wall_r     = inner_r + wall_thick / 2.0
    seg_radius = wall_thick / 2.0

    if not movable:
        # ── Original behaviour: segments welded to the world static body. ──────
        body       = space.static_body
        seg_shapes = []
        for i in range(segments):
            a0 = 2 * math.pi * i       / segments
            a1 = 2 * math.pi * (i + 1) / segments
            p0 = center + pymunk.Vec2d(wall_r * math.cos(a0), wall_r * math.sin(a0))
            p1 = center + pymunk.Vec2d(wall_r * math.cos(a1), wall_r * math.sin(a1))
            s  = pymunk.Segment(body, p0, p1, seg_radius)
            s.friction   = friction
            s.elasticity = elasticity
            space.add(s)
            seg_shapes.append(s)
        # For a fixed circle the segment endpoints are world-absolute coords
        # (p0 = center + offset), so subtract center before atan2.
        _color_ring_segments(seg_shapes, color_bands,
            lambda s, cx=center.x, cy=center.y: math.atan2(
                (s.a.y + s.b.y) / 2.0 - cy,
                (s.a.x + s.b.x) / 2.0 - cx))
        return RingInfo("circle", center, inner_r, wall_thick, movable=False,
                        body=None, segments=seg_shapes)

    # ── Movable: a single rigid body carrying all arc segments. ────────────────
    moment        = ring_mass * wall_r * wall_r
    body          = pymunk.Body(ring_mass, moment)
    body.position = center
    seg_shapes    = []
    for i in range(segments):
        a0 = 2 * math.pi * i       / segments
        a1 = 2 * math.pi * (i + 1) / segments
        p0 = pymunk.Vec2d(wall_r * math.cos(a0), wall_r * math.sin(a0))
        p1 = pymunk.Vec2d(wall_r * math.cos(a1), wall_r * math.sin(a1))
        s  = pymunk.Segment(body, p0, p1, seg_radius)
        s.friction   = friction
        s.elasticity = elasticity
        seg_shapes.append(s)
    space.add(body, *seg_shapes)
    # For a movable circle the segment coords are body-local; the body starts at
    # angle 0, so local coords equal world coords at t=0. The body angle
    # (body.angle) changes at runtime, but color.is fixed at creation time —
    # the color paint rotates with the body automatically.
    _color_ring_segments(seg_shapes, color_bands,
        lambda s: math.atan2(
            (s.a.y + s.b.y) / 2.0,
            (s.a.x + s.b.x) / 2.0))
    return RingInfo("circle", center, inner_r, wall_thick, movable=True,
                    body=body, segments=seg_shapes)


def _add_polygon_ring(space, center, inner_r, wall_thick, n_sides,
                      friction, elasticity, movable, ring_mass, color_bands):
    """
    Regular n-gon wall whose corners are free-rotating hinge joints.

    Each edge is its own rigid segment body.  Adjacent edges share a corner:

      * movable → the two edges meeting at a corner are joined by a free-rotating
        PivotJoint (the hinge), and nothing anchors the loop to the world, so the
        whole polygon can translate, rotate and *flex* at every vertex.
      * fixed   → each corner is pinned to the world at its nominal position, so
        the n-gon keeps its shape and stays put (a static polygonal wall).

    Either way the corners are realised as pivot joints, matching "a regular
    n-gon ring with free-rotating joints at the corners".
    """
    n          = max(3, int(n_sides))
    seg_radius = wall_thick / 2.0

    # Corner vertices on the circumscribed circle of radius inner_r.
    verts = [center + pymunk.Vec2d(inner_r * math.cos(2 * math.pi * k / n),
                                   inner_r * math.sin(2 * math.pi * k / n))
             for k in range(n)]

    edge_mass   = ring_mass / n
    edge_bodies = []
    seg_shapes  = []

    for k in range(n):
        a   = verts[k]
        b   = verts[(k + 1) % n]
        mid = (a + b) * 0.5
        d   = b - a
        half = d.length / 2.0
        ang  = math.atan2(d.y, d.x)

        if movable:
            moment = pymunk.moment_for_segment(edge_mass, (-half, 0), (half, 0),
                                               seg_radius)
            body = pymunk.Body(edge_mass, moment)
        else:
            body = pymunk.Body(body_type=pymunk.Body.STATIC)
        body.position = mid
        body.angle    = ang

        # Segment runs along the body's local x-axis, centred on the midpoint, so
        # local (-half,0) is corner k and local (+half,0) is corner k+1.
        s = pymunk.Segment(body, (-half, 0), (half, 0), seg_radius)
        s.friction   = friction
        s.elasticity = elasticity

        space.add(body, s)
        edge_bodies.append(body)
        seg_shapes.append(s)

    constraints = []
    if movable:
        # Free-rotating hinge between the two edges that meet at each corner.
        for k in range(n):
            prev_edge = edge_bodies[(k - 1) % n]
            this_edge = edge_bodies[k]
            pivot = pymunk.PivotJoint(prev_edge, this_edge, verts[k])
            pivot.collide_bodies = False
            space.add(pivot)
            constraints.append(pivot)
    # else: the edges are STATIC bodies welded to the world at their nominal
    # positions, so the n-gon is already fixed in place; no corner joints needed
    # (and pymunk forbids constraints between two static bodies anyway).

    # Color bands: pre-compute the representative angle (world midpoint of each
    # edge relative to ring center) so the lambda is a simple dict lookup.
    _seg_angle = {}
    for k, s in enumerate(seg_shapes):
        mid   = (verts[k] + verts[(k + 1) % n]) * 0.5
        _seg_angle[id(s)] = math.atan2(mid.y - center.y, mid.x - center.x)
    _color_ring_segments(seg_shapes, color_bands,
                         lambda s: _seg_angle[id(s)])

    return RingInfo("polygon", center, inner_r, wall_thick, movable=movable,
                    n_sides=n, body=None, bodies=edge_bodies,
                    segments=seg_shapes, constraints=constraints)


# =============================================================================
# Smarticle3Link
# =============================================================================

class Smarticle3Link:
    """
    Three-link planar robot: one main body with a left arm and a right arm
    connected by motorised pivot joints with rotary limits.
    """

    def __init__(self, space: pymunk.Space,
                 main_len  = MAIN_LEN,  main_w   = MAIN_W,
                 arm_len   = ARM_LEN,   arm_w    = ARM_W,
                 mass_main = MASS_MAIN, mass_arm = MASS_ARM,
                 kp        = KP_NOM,    max_torque = MAX_MOTOR_TORQUE_NOM,
                 A_deg1    = A_DEG_NOM1, A_deg2  = A_DEG_NOM2,
                 omega1    = OMEGA_NOM1, omega2  = OMEGA_NOM2,
                 phase1    = 0.0,        phase2  = math.pi):

        self.space = space
        self.kp    = float(kp)

        # ── Physical geometry (per-individual; read by the coupling model
        #    in simulation.py and persisted to the initial-conditions JSON) ────
        self.main_len  = main_len
        self.main_w    = main_w
        self.arm_len   = arm_len
        self.arm_w     = arm_w
        self.mass_main = mass_main
        self.mass_arm  = mass_arm
        # Half-span used as L_s in the interaction / wall model.
        self.L_s       = (main_len + 2.0 * arm_len) / 2.0

        self.A1     = math.radians(A_deg1)
        self.A2     = math.radians(A_deg2)
        self.omega1 = float(omega1)
        self.omega2 = float(omega2)
        self.phase1 = float(phase1)
        self.phase2 = float(phase2)

        # ── Bodies ────────────────────────────────────────────────────────────
        self.main_body,  self.main_shape  = box_body(
            space, pos=(0, 0), angle=0.0, size=(main_len, main_w),
            mass=mass_main, friction=10.0, elasticity=0.0)

        self.left_body,  self.left_shape  = box_body(
            space, pos=(0, 0), angle=0.0, size=(arm_len, arm_w),
            mass=mass_arm, friction=0.0, elasticity=0.0)

        self.right_body, self.right_shape = box_body(
            space, pos=(0, 0), angle=0.0, size=(arm_len, arm_w),
            mass=mass_arm, friction=0.0, elasticity=0.0)

        # try:
        #     self.left_shape.color  = (255, 100, 100, 255)  # left arm: red
        #     self.right_shape.color = (100, 100, 255, 255)  # right arm: blue
        # except Exception:
        #     pass

        try:
            self.left_shape.color, self.right_shape.color = arm_colors(self)
        except Exception:
            pass

        # ── Joint geometry ────────────────────────────────────────────────────
        self.left_anchor_local  = pymunk.Vec2d(-main_len / 2.0, 0.0)
        self.right_anchor_local = pymunk.Vec2d( main_len / 2.0, 0.0)
        self.left_inner_local   = pymunk.Vec2d( arm_len  / 2.0, 0.0)
        self.right_inner_local  = pymunk.Vec2d(-arm_len  / 2.0, 0.0)

        pL = self.main_body.local_to_world(self.left_anchor_local)
        pR = self.main_body.local_to_world(self.right_anchor_local)

        # ── Pivot joints ──────────────────────────────────────────────────────
        self.joint_L = pymunk.PivotJoint(self.left_body,  self.main_body, pL)
        self.joint_R = pymunk.PivotJoint(self.right_body, self.main_body, pR)
        self.joint_L.anchor_a = self.left_inner_local
        self.joint_L.anchor_b = self.left_anchor_local
        self.joint_R.anchor_a = self.right_inner_local
        self.joint_R.anchor_b = self.right_anchor_local
        self.joint_L.collide_bodies = False   # arm does not collide with own main
        self.joint_R.collide_bodies = False

        # ── Rotary limits ─────────────────────────────────────────────────────
        self.lim     = math.radians(JOINT_LIMIT_DEG)
        self.limit_L = pymunk.RotaryLimitJoint(self.left_body,  self.main_body, -self.lim, self.lim)
        self.limit_R = pymunk.RotaryLimitJoint(self.right_body, self.main_body, -self.lim, self.lim)

        # ── Motors ────────────────────────────────────────────────────────────
        self.motor_L = pymunk.SimpleMotor(self.left_body,  self.main_body, 0.0)
        self.motor_R = pymunk.SimpleMotor(self.right_body, self.main_body, 0.0)
        self.motor_L.max_force = max_torque
        self.motor_R.max_force = max_torque

        space.add(self.joint_L, self.joint_R,
                  self.limit_L, self.limit_R,
                  self.motor_L, self.motor_R)

        # ── Warmup state ──────────────────────────────────────────────────────
        self.warmup_steps      = 0    # set externally after spawn
        self._warmup_step      = 0    # internal counter
        self._warmup_th1_start = 0.0  # set by set_folded_pose
        self._warmup_th2_start = 0.0

    # ── Accessors ─────────────────────────────────────────────────────────────

    def shapes(self):
        return [self.main_shape, self.left_shape, self.right_shape]

    def bodies(self):
        return [self.main_body, self.left_body, self.right_body]

    def remove_from_space(self):
        for obj in [self.motor_L, self.motor_R,
                    self.limit_L, self.limit_R,
                    self.joint_L, self.joint_R,
                    self.main_shape, self.left_shape, self.right_shape,
                    self.main_body,  self.left_body,  self.right_body]:
            try:
                self.space.remove(obj)
            except Exception:
                pass

    # ── Pose ──────────────────────────────────────────────────────────────────

    def set_folded_pose(self, main_pos: pymunk.Vec2d, main_ang: float,
                        thL_rel: float, thR_rel: float):
        """
        Teleport the robot to (main_pos, main_ang) with relative arm angles
        thL_rel and thR_rel, zeroing all velocities.  Also records the arm
        angles as the warm-up start point.
        """
        thL_rel = max(-self.lim, min(self.lim, thL_rel))
        thR_rel = max(-self.lim, min(self.lim, thR_rel))

        self.main_body.position = main_pos
        self.main_body.angle    = main_ang

        pL = self.main_body.local_to_world(self.left_anchor_local)
        pR = self.main_body.local_to_world(self.right_anchor_local)

        self.left_body.angle  = main_ang + thL_rel
        self.right_body.angle = main_ang + thR_rel

        self.left_body.position  = pL - rot(self.left_inner_local,  self.left_body.angle)
        self.right_body.position = pR - rot(self.right_inner_local, self.right_body.angle)

        for b in self.bodies():
            b.velocity         = (0, 0)
            b.angular_velocity = 0.0

        self._warmup_th1_start = float(thL_rel)
        self._warmup_th2_start = float(thR_rel)

    # ── Control ───────────────────────────────────────────────────────────────

    def desired_theta(self, t: float):
        """Return (th1_des, dth1_des, th2_des, dth2_des) at time t."""
        th1  = self.A1 * math.sin(self.omega1 * t + self.phase1)
        th2  = self.A2 * math.sin(self.omega2 * t + self.phase2)
        dth1 = self.A1 * self.omega1 * math.cos(self.omega1 * t + self.phase1)
        dth2 = self.A2 * self.omega2 * math.cos(self.omega2 * t + self.phase2)
        return th1, dth1, th2, dth2

    def step_control(self, t: float):
        """
        PD motor controller.  During warm-up, linearly interpolates the target
        angle from the spawn pose to the nominal trajectory.

        Hot path: called n times per physics step (n * 180 times per simulated
        second), so attribute lookups are bound to locals, desired_theta() is
        inlined, and main_body.angle is read once instead of twice.  Every
        floating-point operation and its ordering is unchanged.
        """
        A1, A2 = self.A1, self.A2
        w1, w2 = self.omega1, self.omega2
        u1 = w1 * t + self.phase1
        u2 = w2 * t + self.phase2
        th1_des  = A1 * math.sin(u1)
        th2_des  = A2 * math.sin(u2)
        dth1_des = A1 * w1 * math.cos(u1)
        dth2_des = A2 * w2 * math.cos(u2)

        main_ang = self.main_body.angle
        th1_rel = wrap_pi(self.left_body.angle  - main_ang)
        th2_rel = wrap_pi(self.right_body.angle - main_ang)

        # ── Warm-up ramp ──────────────────────────────────────────────────────
        # if self._warmup_step < self.warmup_steps:
        #     alpha    = self._warmup_step / max(1, self.warmup_steps)  # 0 → 1
        #     th1_des  = (1.0 - alpha) * self._warmup_th1_start + alpha * th1_des
        #     th2_des  = (1.0 - alpha) * self._warmup_th2_start + alpha * th2_des
        #     dth1_des = alpha * dth1_des
        #     dth2_des = alpha * dth2_des
        #     self._warmup_step += 1

        if self._warmup_step < self.warmup_steps:
            alpha = self._warmup_step / max(1, self.warmup_steps)
            # interpolation target fixed to trajectory value at t=0, independent of current t
            th1_target = self.A1 * math.sin(self.phase1) * -1
            th2_target = self.A2 * math.sin(self.phase2) * -1
            th1_des    = (1.0 - alpha) * self._warmup_th1_start + alpha * th1_target
            th2_des    = (1.0 - alpha) * self._warmup_th2_start + alpha * th2_target
            dth1_des   = 0.0   # no feedforward during warmup; rely on P control only
            dth2_des   = 0.0
            self._warmup_step += 1

        e1 = wrap_pi(th1_des - th1_rel)
        e2 = wrap_pi(th2_des - th2_rel)

        self.motor_L.rate = dth1_des + self.kp * e1
        self.motor_R.rate = dth2_des + self.kp * e2

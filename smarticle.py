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
)


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

def add_ring(space: pymunk.Space, center: pymunk.Vec2d,
             inner_r: float, wall_thick: float,
             segments: int  = WALL_SEGMENTS,
             friction: float = WALL_FRICTION,
             elasticity: float = WALL_ELASTICITY):
    """Add a circular wall to the space made of short segment shapes."""
    wall_r     = inner_r + wall_thick / 2.0
    seg_radius = wall_thick / 2.0
    static     = space.static_body
    for i in range(segments):
        a0 = 2 * math.pi * i       / segments
        a1 = 2 * math.pi * (i + 1) / segments
        p0 = center + pymunk.Vec2d(wall_r * math.cos(a0), wall_r * math.sin(a0))
        p1 = center + pymunk.Vec2d(wall_r * math.cos(a1), wall_r * math.sin(a1))
        s  = pymunk.Segment(static, p0, p1, seg_radius)
        s.friction   = friction
        s.elasticity = elasticity
        space.add(s)


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

        try:
            self.left_shape.color  = (255, 100, 100, 255)  # left arm: red
            self.right_shape.color = (100, 100, 255, 255)  # right arm: blue
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
        """
        th1_des, dth1_des, th2_des, dth2_des = self.desired_theta(t)

        th1_rel = wrap_pi(self.left_body.angle  - self.main_body.angle)
        th2_rel = wrap_pi(self.right_body.angle - self.main_body.angle)

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
            # 插值终点固定为 t=0 时的轨迹值，与实时 t 无关
            th1_target = self.A1 * math.sin(self.phase1) * -1
            th2_target = self.A2 * math.sin(self.phase2) * -1
            th1_des    = (1.0 - alpha) * self._warmup_th1_start + alpha * th1_target
            th2_des    = (1.0 - alpha) * self._warmup_th2_start + alpha * th2_target
            dth1_des   = 0.0   # warmup 期间不施加前馈，只靠 P 控制稳定
            dth2_des   = 0.0
            self._warmup_step += 1

        e1 = wrap_pi(th1_des - th1_rel)
        e2 = wrap_pi(th2_des - th2_rel)

        self.motor_L.rate = dth1_des + self.kp * e1
        self.motor_R.rate = dth2_des + self.kp * e2

"""
gait.py  ─  Per-robot gait (运动模式) state and runtime gait control.

This module owns three things that used to be inlined in simulation.py:

1. The +/-XYZ integer command encoding and its decode tables.
2. GaitState  -- the per-robot record of "which motion pattern is this robot
   running right now", tracked individually by robot id.
3. GaitController -- applies the initial commands (exactly reproducing the old
   inline decode loop in run_trial) and, when enabled, lets a user-supplied
   callback change any robot's motion pattern MID-SIMULATION, addressed by id.

Command encoding (unchanged, see config.py)
-------------------------------------------
    signed 3-digit integer +/-XYZ
        X (hundreds, 0-8) : initial phase  -- 0 = random, 1-8 from PHASE_TABLE
        Y (tens,     1-6) : amplitude      -- AMPLI_TABLE index
        Z (units,    1-9) : frequency      -- FREQ_TABLE index
        sign (+)          : both joints in phase
        sign (-)          : joints in antiphase (phase2 = phase1 + pi)

Zero-phase switching
--------------------
The desired joint trajectory is theta(t) = A * sin(omega*t + phase) evaluated
from ABSOLUTE simulation time, so naively rewriting A/omega/phase mid-run makes
the commanded angle jump and the PD controller slams the arms.

A runtime change is therefore QUEUED and takes effect at the robot's next
rising zero crossing of its own oscillator, where theta == 0 for any amplitude.
The exact crossing time is solved analytically at request time:

    u_req   = omega_old * t_req + phase_old
    k       = ceil(u_req / 2pi)
    t_cross = (2pi*k - phase_old) / omega_old

and at the first physics step with t >= t_cross the new phase is chosen so the
commanded angle is continuous to machine precision:

    theta_now = A_old * sin(omega_old*t + phase_old)      # ~0
    phase_new = asin(clamp(theta_now / A_new)) - omega_new * t

Because the encoding forces phase2 - phase1 to be exactly 0 or pi, sin(u2) is
zero whenever sin(u1) is, so both arms cross zero together and only joint 1
needs to be watched.  That same constraint means both arms can be matched
exactly only when the new command keeps the old antiphase bit; when the bit
flips the switch anchors at zero phase instead, which splits the (sub-step)
residual evenly so neither arm moves by more than one ordinary trajectory step.

Consequence: the X digit (initial phase) of a RUNTIME command is ignored -- a
mid-run switch always restarts from zero phase.  Set
config.RUNTIME_GAIT_SWITCH_MODE = "immediate" to get the old absolute-phase
behaviour instead (which does produce a commanded-angle jump).
"""

import importlib
import math
import random
import re

# Single source of truth for the lookup tables (sweep.py imports these too).
# naming.py pulls in nothing but math, so importing this module is free of any
# config dependency -- bodies (which does import config) is imported lazily in
# GaitController.__init__ instead.  That keeps `from gait import
# build_command_array` usable from inside config.py without a circular import.
from naming import _INITIAL_DICT, _AMPLI_DICT, _FREQ_DICT

PHASE_TABLE = _INITIAL_DICT   # slots 1-8 (rad)
AMPLI_TABLE = _AMPLI_DICT     # slots 1-6 (rad)
FREQ_TABLE  = _FREQ_DICT      # slots 1-9 (Hz)

_TWO_PI = 2.0 * math.pi

# Fallbacks for out-of-range digits, matching the previous inline decode loop.
_DEFAULT_FREQ_HZ = 0.5
_DEFAULT_AMPLI   = math.pi / 4


# =============================================================================
# Decoding
# =============================================================================

def split_command(cmd: int):
    """Return (sign, x, y, z) for a +/-XYZ command integer."""
    sign    = -1 if cmd < 0 else 1
    abs_cmd = abs(cmd)
    z       = abs_cmd % 10
    y       = (abs_cmd % 100 - z) // 10
    x       = abs_cmd // 100
    return sign, x, y, z


def decode_command(cmd: int) -> dict:
    """
    Decode a command integer into its parameters.

    phase_rad is None when x == 0 ("randomise at apply time").  Digit handling
    is byte-for-byte the one that used to live in run_trial: an out-of-range
    Y or Z falls back to the defaults above, while an out-of-range X (9) is an
    IndexError, exactly as before.
    """
    sign, x, y, z = split_command(cmd)
    freq_hz   = FREQ_TABLE[z - 1]  if 1 <= z <= 9 else _DEFAULT_FREQ_HZ
    ampli_rad = AMPLI_TABLE[y - 1] if 1 <= y <= 6 else _DEFAULT_AMPLI
    phase_rad = None if x == 0 else PHASE_TABLE[x - 1]
    return {
        "phase_slot": x,
        "ampli_slot": y,
        "freq_slot":  z,
        "antiphase":  sign < 0,
        "freq_hz":    freq_hz,
        "ampli_rad":  ampli_rad,
        "ampli_deg":  math.degrees(ampli_rad),
        "phase_rad":  phase_rad,
    }


# =============================================================================
# Debug command text  ("a-862" style)
# =============================================================================
#
# A tiny text syntax for writing gait requests by hand, mirroring how commands
# are typed at the real hardware.  A clause is a target expression followed
# directly by the signed command -- there is NO separator, the leading "-" of a
# negative command is part of the command:
#
#     <targets><command>
#
#   "a-862"  =  every robot -> -862          "a862"  =  every robot -> +862
#
#   Targets                 meaning
#   ---------------------   ------------------------------------------------
#   a  /  all  /  *         every robot
#   even  /  odd            robots with even / odd id
#   7                       robot 7
#   3,7,12                  robots 3, 7 and 12
#   0..4    or   0:4        robots 0 through 4 (inclusive)
#   a,~3,~7                 every robot except 3 and 7   (~ or ! excludes)
#   (omitted)               every robot -- a bare command means "a"
#
#   Clause                  meaning
#   ---------------------   ------------------------------------------------
#   a-862                   all robots -> -862
#   a862                    all robots -> 862
#   -862                    all robots -> -862     (bare command = "a" + cmd)
#   0..4-851                robots 0-4 -> -851
#   7-862                   robot 7 -> -862
#   3,7,12=461              '=' or a space may separate for readability
#   a462; 0..8-851          two clauses; later ones win on overlap
#   even862  # comment      '#' runs to end of line, ';' or newline separates
#
# The command is the TRAILING signed 1-3 digit group, so "12862" reads as
# robot 12 -> 862 (commands are canonically three digits).
#
# Used by GaitController.request_text() / build_command_array() and accepted
# straight from a runtime callback (return the string instead of a dict).

_ALL_WORDS  = frozenset(("a", "all", "*"))
_SEP_CHARS  = "=>"          # optional, purely for readability
_CLAUSE_RE  = re.compile(r"^(?P<targets>.*?)(?P<cmd>[+-]?\d{1,3})$")
_RANGE_RE   = re.compile(r"^(\d+)(?:\.\.|:)(\d+)$")


def validate_command(cmd) -> int:
    """
    Check a command integer and return it, or raise ValueError explaining why
    it is wrong.  Stricter than decode_command() on purpose: a hand-typed debug
    command should fail loudly rather than silently fall back to a default.
    """
    try:
        cmd = int(cmd)
    except (TypeError, ValueError):
        raise ValueError(f"{cmd!r} is not an integer command")
    if abs(cmd) > 999:
        raise ValueError(f"{cmd} has more than 3 digits (expected +/-XYZ)")
    sign, x, y, z = split_command(cmd)
    bad = []
    if not 0 <= x <= 8:
        bad.append(f"phase digit X={x} (expected 0-8, 0=random)")
    if not 1 <= y <= 6:
        bad.append(f"amplitude digit Y={y} (expected 1-6)")
    if not 1 <= z <= 9:
        bad.append(f"frequency digit Z={z} (expected 1-9)")
    if bad:
        raise ValueError(f"{cmd}: " + "; ".join(bad))
    return cmd


def parse_targets(text: str, n: int) -> list:
    """
    Resolve a target expression to a sorted list of robot ids in 0..n-1.
    See the syntax table above.  Raises ValueError on anything unrecognised.
    """
    items = [it for it in text.replace(" ", "").split(",") if it]
    if not items:
        raise ValueError("empty target list")

    include, exclude = set(), set()
    for item in items:
        bucket = include
        if item[0] in "~!":
            bucket, item = exclude, item[1:]
            if not item:
                raise ValueError("'~' must be followed by a target")
        low = item.lower()
        if low in _ALL_WORDS:
            bucket |= set(range(n))
        elif low == "even":
            bucket |= set(range(0, n, 2))
        elif low == "odd":
            bucket |= set(range(1, n, 2))
        else:
            m = _RANGE_RE.match(item)
            if m:
                lo, hi = int(m.group(1)), int(m.group(2))
                if lo > hi:
                    lo, hi = hi, lo
                bucket |= set(range(lo, hi + 1))
            elif item.isdigit():
                bucket.add(int(item))
            else:
                raise ValueError(
                    f"unrecognised target {item!r} (expected a number, "
                    f"'i..j', 'a', 'even', 'odd', or '~' + one of those)")

    ids = include - exclude
    bad = sorted(i for i in ids if not 0 <= i < n)
    if bad:
        raise ValueError(f"target id(s) {bad} outside 0..{n - 1}")
    return sorted(ids)


def parse_request(text: str, n: int) -> dict:
    """
    Parse one or more clauses into {robot_id: command}.

    Clauses are separated by ';' or newlines and applied in order, so a later
    clause overrides an earlier one for the robots they share -- which is what
    makes "a462; 0..4-851" read the way it looks.
    """
    req = {}
    for raw in str(text).replace("\n", ";").split(";"):
        clause = raw.split("#", 1)[0].replace(" ", "").replace("\t", "")
        if not clause:
            continue

        # An optional '=' / '>' may be written for readability; otherwise the
        # command is simply the trailing signed 1-3 digit group, so the '-' in
        # "a-862" belongs to the command rather than separating anything.
        cut = next((i for i, ch in enumerate(clause) if ch in _SEP_CHARS), -1)
        if cut >= 0:
            targets, cmd_text = clause[:cut] or "a", clause[cut + 1:]
        else:
            m = _CLAUSE_RE.match(clause)
            if not m:
                raise ValueError(
                    f"cannot parse {raw.strip()!r}: expected <targets><command> "
                    f"with the command a signed 1-3 digit number, e.g. 'a-862' "
                    f"(all robots -> -862)")
            targets  = m.group("targets") or "a"     # bare command means "all"
            cmd_text = m.group("cmd")

        if not cmd_text:
            raise ValueError(f"{raw.strip()!r} has no command")
        try:
            cmd = validate_command(cmd_text)
        except ValueError as e:
            raise ValueError(f"in {raw.strip()!r}: {e}")
        for rid in parse_targets(targets, n):
            req[rid] = cmd
    return req


def build_command_array(text: str, n: int, default=None) -> list:
    """
    Build a length-n COMMAND_ARRAY from debug text, for use in config.py:

        from gait import build_command_array
        COMMAND_ARRAY = build_command_array("a462; 0..8862", N_SMARTICLES)

    Every robot must be covered unless `default` is given.
    """
    req = parse_request(text, n)
    missing = [i for i in range(n) if i not in req]
    if missing and default is None:
        raise ValueError(
            f"{text!r} leaves robot(s) {missing[:8]}"
            f"{'...' if len(missing) > 8 else ''} unassigned; add an 'a<cmd>' "
            f"clause or pass default=")
    if missing:
        default = validate_command(default)
    return [req.get(i, default) for i in range(n)]


def format_command(cmd) -> str:
    """Human-readable one-liner for a command, e.g. for debug logging."""
    if cmd is None:
        return "none"
    d = decode_command(cmd)
    ph = ("random" if d["phase_rad"] is None
          else f"{math.degrees(d['phase_rad']):.0f}deg")
    return (f"{cmd:+d} (f={d['freq_hz']}Hz A={d['ampli_deg']:.0f}deg "
            f"phase={ph}{', antiphase' if d['antiphase'] else ''})")


def format_ids(ids) -> str:
    """Collapse a list of ids into '0..4,9,12..14' for compact logging."""
    ids = sorted(ids)
    if not ids:
        return "-"
    out, lo, prev = [], ids[0], ids[0]
    for i in list(ids[1:]) + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        out.append(str(lo) if lo == prev else f"{lo}..{prev}")
        lo = prev = i
    return ",".join(out)


# =============================================================================
# Per-robot gait state
# =============================================================================

class GaitState:
    """
    The motion pattern robot `robot_id` is running, plus any queued change.

    Lives for the whole trial and is reachable both as controller.states[id]
    and as smarticle.gait.  In-memory only -- nothing here is written to disk.
    """

    __slots__ = ("robot_id", "command", "freq_hz", "omega", "ampli",
                 "phase1", "phase2", "antiphase", "applied_t",
                 "pending_cmd", "pending_t_cross", "pending_requested_t",
                 "history")

    def __init__(self, robot_id: int):
        self.robot_id  = robot_id
        self.command   = None      # current command int (None before first apply)
        self.freq_hz   = 0.0
        self.omega     = 0.0
        self.ampli     = 0.0
        self.phase1    = 0.0
        self.phase2    = 0.0
        self.antiphase = False
        self.applied_t = 0.0       # sim time the current command took effect

        self.pending_cmd         = None   # queued command, None if nothing queued
        self.pending_t_cross     = None   # sim time it will take effect
        self.pending_requested_t = None   # sim time it was requested

        self.history = []          # [(t_applied, old_cmd, new_cmd), ...]

    def __repr__(self):
        p = "" if self.pending_cmd is None else f" pending={self.pending_cmd}"
        return (f"<GaitState id={self.robot_id} cmd={self.command} "
                f"f={self.freq_hz}Hz A={self.ampli:.3f}{p}>")


def apply_command(sm, state: GaitState, cmd: int, t: float = 0.0,
                  mode: str = "initial") -> None:
    """
    Write the parameters encoded by `cmd` onto smarticle `sm` and record them
    in `state`.

    mode "initial" / "immediate"
        Absolute-phase path.  Identical arithmetic (including the single
        random.random() draw for x == 0) to the original decode loop, so
        "initial" reproduces the pre-existing trial setup bit for bit.
    mode "zero_phase"
        Continuity path used for mid-run switches: the new phase is solved so
        the commanded angle at time t is unchanged.  Only meaningful when
        called at/just after a zero crossing -- see the module docstring.
    """
    sign, x, y, z = split_command(cmd)
    freq_hz = FREQ_TABLE[z - 1]  if 1 <= z <= 9 else _DEFAULT_FREQ_HZ
    ampli   = AMPLI_TABLE[y - 1] if 1 <= y <= 6 else _DEFAULT_AMPLI

    if mode == "zero_phase":
        # Commanded angle right now, under the OLD parameters (read before the
        # writes below).  We are at/just past a zero crossing, so it is at most
        # one ordinary trajectory step away from 0.
        #
        # The encoding pins phase2 - phase1 to exactly 0 or pi, so both arms
        # can be matched simultaneously only when the new command keeps the old
        # antiphase bit: with the bit flipped, theta2 = -theta1 under the old
        # parameters but +theta1 (or vice versa) under the new ones.  In that
        # case anchoring at zero phase is the least-squares optimum -- it
        # splits the residual evenly, so neither arm moves by more than a
        # single ordinary step of its own trajectory.
        if (sign < 0) == state.antiphase and ampli > 0.0:
            th_now = sm.A1 * math.sin(sm.omega1 * t + sm.phase1)
            arg    = th_now / ampli
            arg    = 1.0 if arg > 1.0 else (-1.0 if arg < -1.0 else arg)
            u_new  = math.asin(arg)     # rising branch, matching a rising cross
        else:
            u_new = 0.0
        ph1 = u_new - (freq_hz * 2 * math.pi) * t
    else:
        ph1 = random.random() * 2 * math.pi if x == 0 else PHASE_TABLE[x - 1]
    ph2 = ph1 + math.pi if sign < 0 else ph1

    sm.omega1 = freq_hz * 2 * math.pi
    sm.omega2 = freq_hz * 2 * math.pi
    sm.A1     = ampli
    sm.A2     = ampli
    sm.phase1 = ph1
    sm.phase2 = ph2

    old_cmd = state.command
    state.command   = cmd
    state.freq_hz   = freq_hz
    state.omega     = sm.omega1
    state.ampli     = ampli
    state.phase1    = ph1
    state.phase2    = ph2
    state.antiphase = sign < 0
    state.applied_t = t
    if old_cmd is not None:
        state.history.append((t, old_cmd, cmd))


# =============================================================================
# Read-only view handed to the user callback
# =============================================================================

class RobotView:
    """Lightweight per-frame snapshot of one robot, for the runtime callback."""

    __slots__ = ("id", "x", "y", "angle", "vx", "vy", "dist_to_center",
                 "command", "freq_hz", "ampli", "phase1", "phase2",
                 "antiphase", "pending_cmd")

    def __init__(self, sm, state, cx, cy):
        p  = sm.main_body.position
        v  = sm.main_body.velocity
        self.id             = state.robot_id
        self.x              = p.x
        self.y              = p.y
        self.angle          = sm.main_body.angle
        self.vx             = v.x
        self.vy             = v.y
        self.dist_to_center = math.hypot(p.x - cx, p.y - cy)
        self.command        = state.command
        self.freq_hz        = state.freq_hz
        self.ampli          = state.ampli
        self.phase1         = state.phase1
        self.phase2         = state.phase2
        self.antiphase      = state.antiphase
        self.pending_cmd    = state.pending_cmd

    def __repr__(self):
        return (f"<RobotView id={self.id} cmd={self.command} "
                f"pos=({self.x:.1f},{self.y:.1f})>")


def resolve_callback(spec):
    """
    Resolve "module:function" (or "module.function") into a callable.

    Returns None if spec is falsy.  A spec that is already callable is passed
    through, so a config can hold either form.
    """
    if not spec:
        return None
    if callable(spec):
        return spec
    text = str(spec)
    if ":" in text:
        mod_name, func_name = text.split(":", 1)
    else:
        mod_name, _, func_name = text.rpartition(".")
    if not mod_name or not func_name:
        raise ValueError(
            f"RUNTIME_GAIT_CONTROLLER={spec!r} is not of the form "
            f"'module:function'.")
    fn = getattr(importlib.import_module(mod_name), func_name)
    if not callable(fn):
        raise TypeError(f"RUNTIME_GAIT_CONTROLLER={spec!r} is not callable.")
    return fn


# =============================================================================
# Controller
# =============================================================================

class GaitController:
    """
    Owns one GaitState per robot and drives runtime motion-pattern changes.

    Construction reproduces the old inline decode loop exactly (same order,
    same bodies.gait_for_smarticle() override, same single random.random()
    draw per x == 0 robot), so a run with runtime control disabled is
    numerically identical to before this module existed.
    """

    def __init__(self, smarticles, commands, warmup_steps=0,
                 phase_color_fn=None, switch_mode="zero_phase",
                 callback=None, center=(0.0, 0.0), ctx=None, verbose=False):
        self.smarticles  = smarticles
        self.switch_mode = switch_mode
        self.callback    = callback
        self.verbose     = bool(verbose)
        self.cx, self.cy = float(center[0]), float(center[1])
        self.ctx         = {} if ctx is None else ctx

        from bodies import gait_for_smarticle   # lazy: see the import note above

        self.states      = []
        self._pending    = set()
        self.has_pending = False      # hot-loop guard: read once per physics step
        self._warned     = False

        for i, s in enumerate(smarticles):
            st       = GaitState(i)
            override = gait_for_smarticle(s)   # noqa: F821 (bound just above)
            cmd      = override if override is not None else commands[i]
            apply_command(s, st, cmd, t=0.0, mode="initial")
            if phase_color_fn is not None:
                s.phase_marker_color = phase_color_fn(st.phase1)
            s.warmup_steps = warmup_steps
            s._warmup_step = 0
            s.gait         = st
            self.states.append(st)

    # ── Queries ──────────────────────────────────────────────────────────────

    def command_of(self, robot_id: int):
        """Current command integer of robot `robot_id`."""
        return self.states[robot_id].command

    def commands(self):
        """Current command of every robot, indexed by id."""
        return [st.command for st in self.states]

    def views(self):
        """Read-only per-robot snapshots (what the callback receives)."""
        cx, cy = self.cx, self.cy
        return [RobotView(sm, st, cx, cy)
                for sm, st in zip(self.smarticles, self.states)]

    def summary(self) -> str:
        """
        One-line census of who is running what, in the same id notation the
        text syntax accepts:  "462 x14 [0..3,5..14]  -851 x3 [4,15..16]".
        Handy to print at a breakpoint or from a callback while debugging.
        """
        groups = {}
        for st in self.states:
            groups.setdefault(st.command, []).append(st.robot_id)
        parts = []
        for cmd, ids in sorted(groups.items(), key=lambda kv: -len(kv[1])):
            shown = format_ids(ids)
            if len(shown) > 40:          # e.g. every even id at n = 100
                shown = shown[:37].rsplit(",", 1)[0] + ",..."
            parts.append(f"{cmd:+d} x{len(ids)} [{shown}]")
        pend = [st.robot_id for st in self.states if st.pending_cmd is not None]
        if pend:
            parts.append(f"pending [{format_ids(pend)}]")
        return "  ".join(parts)

    # ── Requesting a change ──────────────────────────────────────────────────

    def request_text(self, text, t) -> int:
        """
        Queue changes written in the debug syntax -- "a-862" (all robots to
        -862), "0..4-851", "3,7,12=461; even862" -- and return how many robots
        were queued.

        A parse error is reported and the whole string ignored: one typo must
        not half-apply a request.  See the syntax table above.
        """
        try:
            req = parse_request(text, len(self.states))
        except ValueError as e:
            self._warn(f"bad gait command {str(text).strip()!r}: {e}")
            return 0
        return sum(1 for rid, cmd in req.items() if self.request(rid, cmd, t))

    def request(self, robot_id, cmd, t) -> bool:
        """
        Queue a motion-pattern change for one robot.  It takes effect at that
        robot's next zero phase (or immediately in "immediate" mode).  A second
        request for the same robot replaces the still-pending first one.

        Returns True if the request was accepted.
        """
        try:
            rid = int(robot_id)
        except (TypeError, ValueError):
            self._warn(f"ignoring gait request with non-integer id {robot_id!r}")
            return False
        if not (0 <= rid < len(self.states)):
            self._warn(f"ignoring gait request for out-of-range id {rid} "
                       f"(valid: 0..{len(self.states) - 1})")
            return False
        try:
            cmd = int(cmd)
            decode_command(cmd)          # validate the digits
        except Exception as e:
            self._warn(f"ignoring invalid gait command {cmd!r} for id {rid}: {e}")
            return False

        sm = self.smarticles[rid]
        st = self.states[rid]

        if self.switch_mode == "immediate":
            t_cross = t
        else:
            w = sm.omega1
            if w <= 0.0:
                t_cross = t              # not oscillating: nothing to wait for
            else:
                u       = w * t + sm.phase1
                k       = math.ceil(u / _TWO_PI)
                t_cross = (k * _TWO_PI - sm.phase1) / w
                if t_cross < t:
                    t_cross = t

        st.pending_cmd         = cmd
        st.pending_t_cross     = t_cross
        st.pending_requested_t = t
        self._pending.add(rid)
        self.has_pending = True
        return True

    # ── Per-physics-step application ─────────────────────────────────────────

    def step(self, t):
        """
        Apply every pending change whose zero crossing has been reached.
        Returns the ids applied this step (empty tuple when none), so the
        caller can refresh anything it cached from the gait parameters.
        """
        if not self._pending:
            return ()
        changed = None
        log     = {}                     # (old, new, requested_t) -> [ids]
        for rid in tuple(self._pending):
            st = self.states[rid]
            if t < st.pending_t_cross:
                continue
            old_cmd = st.command
            req_t   = st.pending_requested_t
            apply_command(self.smarticles[rid], st, st.pending_cmd,
                          t=t, mode=self.switch_mode)
            if self.verbose:
                log.setdefault((old_cmd, st.command, req_t), []).append(rid)
            st.pending_cmd         = None
            st.pending_t_cross     = None
            st.pending_requested_t = None
            self._pending.discard(rid)
            if changed is None:
                changed = []
            changed.append(rid)
        # One line per (old -> new) group: a mass switch like "a-862" moves
        # every robot on the same step and would otherwise print n copies.
        for (old_cmd, new_cmd, req_t), ids in log.items():
            print(f"[gait] t={t:8.4f}s  {format_ids(ids)} "
                  f"({len(ids)} robot{'s' if len(ids) > 1 else ''})  "
                  f"{old_cmd:+d} -> {format_command(new_cmd)}"
                  f"   [requested t={req_t:.4f}s]")
        self.has_pending = bool(self._pending)
        return changed if changed is not None else ()

    # ── Per-frame polling of the user callback ───────────────────────────────

    def poll(self, t):
        """
        Ask the user callback whether any robot should change motion pattern.

        The callback signature is  fn(t, robots, ctx) -> {robot_id: command}
        (or None).  It may equally return a debug string in the text syntax --
        "a-862", "0..4-851; 7461" -- which is the quickest way to script a run
        by hand.  A raising callback never kills the trial: the frame is
        skipped and the problem is reported once.
        """
        if self.callback is None:
            return
        try:
            req = self.callback(t, self.views(), self.ctx)
        except Exception as e:
            self._warn(f"runtime gait callback raised {e!r}; frame skipped")
            return
        if not req:
            return
        if isinstance(req, str):
            self.request_text(req, t)
            return
        try:
            items = req.items()
        except AttributeError:
            self._warn(f"runtime gait callback returned {type(req).__name__}, "
                       f"expected a dict of {{robot_id: command}} or a command "
                       f"string like 'a-862'")
            return
        for rid, cmd in items:
            self.request(rid, cmd, t)

    # ── Internals ────────────────────────────────────────────────────────────

    def _warn(self, msg):
        """Report the first problem only -- a bad callback fires every frame."""
        if self._warned:
            return
        self._warned = True
        print(f"[gait] {msg} (further gait warnings suppressed)")

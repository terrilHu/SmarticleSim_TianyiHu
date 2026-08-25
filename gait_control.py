"""
gait_control.py  ─  User-editable runtime gait controllers.

Point config.RUNTIME_GAIT_CONTROLLER at one of these ("gait_control:name") and
set config.ENABLE_RUNTIME_GAIT_CONTROL = True.  Edit freely -- nothing else in
the simulator imports from this module.

Callback contract
-----------------
    def my_controller(t, robots, ctx) -> dict[int, int] | None

    t       simulation time in seconds at the current frame boundary.
    robots  list of read-only RobotView snapshots, indexed by robot id:
                .id .x .y .angle .vx .vy .dist_to_center
                .command .freq_hz .ampli .phase1 .phase2 .antiphase
                .pending_cmd        (a change already queued for this robot)
    ctx     mutable dict that persists for the whole trial -- use it to keep
            your own state.  Pre-seeded with:
                "trial_id", "n", "center" (cx, cy), "inner_r", "dt", "max_runtime"

    Return {robot_id: command} for the robots whose motion pattern should
    change, or None / {} for "no change".  Commands are the usual signed
    3-digit +/-XYZ integers (see config.py).

    You may also return a DEBUG COMMAND STRING instead of a dict, written the
    way commands are typed at the real hardware -- a target expression followed
    directly by the signed command, with no separator between them:

        return "a-862"              # every robot -> -862
        return "a862"               # every robot -> +862
        return "-862"               # same: a bare command means "all"
        return "0..4-851"           # robots 0-4 -> -851
        return "3,7,12=461"         # a specific few ('=' optional, readability)
        return "a462; even-851"     # later clauses win on overlap
        return "a,~3-862"           # everyone except robot 3

    Full syntax table (targets a / all / * / even / odd / i..j / ~exclude) is
    at the top of gait.py.  For a purely time-driven run you do not need a
    callback at all: list the steps in config.RUNTIME_GAIT_SCRIPT and point
    RUNTIME_GAIT_CONTROLLER at "gait_control:script" below.

Timing
------
A returned change is QUEUED, not applied on the spot: it takes effect at that
robot's next zero phase so the commanded joint angle stays continuous.  This
means the X (initial-phase) digit of a runtime command is ignored -- a mid-run
switch always restarts from zero phase.  Re-requesting a robot that already has
a pending change simply replaces it, so returning the same dict every frame is
harmless (but check `.command` / `.pending_cmd` if you want to fire only once).
"""


def example_schedule(t, robots, ctx):
    """
    Fixed timetable.  Each entry is (seconds, what-to-do), and what-to-do may
    be written either way -- mix them freely:

        (5.0, {3: -426, 7: -426, 12: 861})   explicit {robot_id: command}
        (5.0, "3,7-426; 12861")              the same thing in shorthand
        (5.0, "a-862")                       shorthand's real advantage: "all"

    The two differ in one way worth knowing.  A dict is filtered to the robots
    that exist (rid < n), so an entry written for 17 robots quietly does less
    at n = 3.  A string is validated as a whole: an explicit id past the end of
    the population is reported and the WHOLE clause is skipped.  Target words
    that are resolved against the real population -- a / all / even / odd /
    i..j -- are n-safe either way, so prefer those in a plan you reuse across
    population sizes.
    """
    plan = ctx.setdefault("plan", [
        ( 5.0, "a-62"),
        (10.0, {3: 62}),
    ])
    done = ctx.setdefault("done", 0)
    if done < len(plan) and t >= plan[done][0]:
        ctx["done"] = done + 1
        entry = plan[done][1]
        if isinstance(entry, str):
            return entry                      # parsed by the gait controller
        n = ctx["n"]
        return {rid: cmd for rid, cmd in entry.items() if rid < n}
    return None

def example_closed_loop(t, robots, ctx):
    """
    State-driven: robots that have drifted close to the wall drop to a slow,
    small-amplitude gait (-412); robots back in the interior return to -462.

    Only robots that are not already running the target command are returned,
    so each robot switches at most once per crossing of the threshold.
    """
    if t < 5.0:                      # let the swarm settle first
        return None
    r_out = 0.85 * ctx["inner_r"]
    r_in  = 0.70 * ctx["inner_r"]
    req = {}
    for r in robots:
        if r.pending_cmd is not None:
            continue
        if r.dist_to_center > r_out and r.command != -412:
            req[r.id] = -412
        elif r.dist_to_center < r_in and r.command != -462:
            req[r.id] = -462
    return req or None


def example_stagger(t, robots, ctx):
    """
    Convert the swarm one robot at a time: every 2 s the lowest-numbered robot
    still running the old command is switched over.  Useful for measuring how a
    collective state responds to a slowly growing minority population.
    """
    OLD, NEW, PERIOD = 462, -851, 2.0
    if t < ctx.setdefault("next_t", PERIOD):
        return None
    ctx["next_t"] = t + PERIOD
    for r in robots:
        if r.command == OLD and r.pending_cmd is None:
            return {r.id: NEW}
    return None


def script(t, robots, ctx):
    """
    Replay config.RUNTIME_GAIT_SCRIPT -- a list of (time_seconds, "command
    text") pairs -- with no Python needed for the common "do X at t, then Y at
    t'" case:

        RUNTIME_GAIT_SCRIPT = [
            ( 5.0, "a-862"),          # everyone switches to -862 at t = 5 s
            (10.0, "0..4-851"),       # robots 0-4 go to -851 at t = 10 s
            (20.0, "3,7,12=461"),
        ]

    Steps are sorted by time; several sharing a timestamp are merged into one
    request, so they all take effect at the same zero crossing.
    """
    steps = ctx.get("_script_steps")
    if steps is None:
        import config as cfg
        raw = getattr(cfg, "RUNTIME_GAIT_SCRIPT", ()) or ()
        # Sort on the timestamp ONLY.  Python's sort is stable, so steps
        # sharing a timestamp keep the order they were written in -- which
        # matters, because merged clauses override left to right.
        steps = sorted(((float(when), str(text)) for when, text in raw),
                       key=lambda step: step[0])
        ctx["_script_steps"] = steps
        ctx["_script_i"] = 0

    i   = ctx["_script_i"]
    due = []
    while i < len(steps) and t >= steps[i][0]:
        due.append(steps[i][1])
        i += 1
    ctx["_script_i"] = i
    return "; ".join(due) if due else None


def example_text(t, robots, ctx):
    """
    The same idea written inline, to show the debug string syntax in place.
    """
    plan = ctx.setdefault("plan", [
        ( 5.0, "a-862"),              # all -> -862
        (10.0, "0..4-851"),           # robots 0-4 -> -851
        (15.0, "even461; 7862"),      # even ids -> 461, then robot 7 -> 862
        (20.0, "a,~0-426"),           # everyone except robot 0
    ])
    i = ctx.setdefault("i", 0)
    if i < len(plan) and t >= plan[i][0]:
        ctx["i"] = i + 1
        return plan[i][1]
    return None

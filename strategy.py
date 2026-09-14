"""
strategy.py  ─  runtime strategy: grouping + spatial roles -> layered commands.

Pipeline (run once per strategy tick):

    position/heading
      │
      ├─ Voronoi(Delaunay) adjacency + max_dist threshold   pos_all_alignment.voronoi_adjacency
      ├─ connected components -> groups                      pos_all_grouping.compute_frame_groups
      │      -> rec = {ids, pos, theta, labels, sizes, alignment, centroid}
      │
      ├─ Layer 1  match group size to a rule -> group command   GroupRuleLayer (with hysteresis)
      ├─ Layer 2  spatial role -> role command                  spatial_roles.SpatialRoleTracker
      └─ Layer 3  fallback -> leave_command
                     │
                     └─ merge into {robot_id: command} handed to gait.GaitController

Default layer priority matches the real-robot side: **group commands override role
commands**. A robot that is both a long-axis endpoint and in a large group executes
the group command — roles only govern robots not claimed by any group rule.
Add "override_group": True to a role to bump it above the group layer instead.
Within the same layer, roles **earlier** in the list have higher priority.

Grouping reuses the exact functions from pos_all_alignment / pos_all_grouping rather
than reimplementing them: the grouping seen in offline analysis (_alignment.csv,
the A(n) curve) is the same definition the strategy acts on live, so "why was this
command sent at that moment" can be cross-checked against the analysis output later.

Commands are always the ±XYZ integers from config.py. What gets sent out is only a
**request**: gait.GaitController schedules it to take effect at that robot's next
phase zero-crossing, so a role handover never yanks the arm mid-swing.

Tuning
======

max_dist — the single most important number here, and it **percolates**
-------------------------------------------------------------------------
Measured on a real 100-robot trial (6 snapshots, t = 5~17.5 s), threshold in units
of body length L_s = (MAIN_LEN + 2*ARM_LEN)/2 = 105 px:

    max_dist/L_s   #groups  largest  top few
       0.70        46.7     7.5    [7, 6, 4, 4, 4, 3]
       1.00        26.8    19.0    [17, 13, 11, 5, 5, 5]
       1.10        17.2    29.5    [21, 19, 18, 6, 5, 5]    <- default
       1.20         4.8    84.5    [89, 10, 1]
       1.30         1.8    99.0    [100]
       1.43         1.0   100.0    [100]     (= 150 px, used for offline analysis)

Above ~1.2*L_s the whole swarm collapses into one group: every robot matches the
same rule, and the PCA role selector also abstains because the whole field is
circular, so the strategy degenerates outright. 150.0 is fine for offline alignment
analysis (a loose neighborhood is the point there) but useless for distinguishing
groups. The default is expressed as a multiple of body length so its meaning stays
the same when N_SMARTICLES (and the arena that scales with it) changes.

Two pitfalls — check before touching a layer
----------------------------------------------
(a) **An unbounded Layer-1 rule starves Layer 2.** Layer 1 has the highest priority,
    so an open-ended "size_range": (8, None) will claim the whole field the moment
    the swarm percolates into a single blob, and roles never fire again. Observed:
    running with that rule produced one 94-robot group + a single uniform command
    -851 across the field. Cap the upper bound so the percolated state is left to
    the role layer.
(b) **PCA-family selectors abstain on circular aggregates** — by design: the major
    axis direction of a disk is noise (see min_anisotropy in spatial_roles).
    group_major_ends suits **elongated** formations; for compact swarms use a
    shape-agnostic selector instead, e.g. convex_hull (boundary) or farthest.
"""

import inspect
import math

import numpy as np

from spatial_roles import (SELECTORS, SpatialRoleTracker,
                           sel_largest_group_axis_ends)


# =============================================================================
# Grouping result for a single frame
# =============================================================================

def frame_record(ids, pos, theta, max_dist, include_singletons=True):
    """
    Run one frame of Voronoi+threshold grouping, return the rec selectors expect.

    Key names are the contract fixed on the spatial_roles side:
        ids     (n,)   robot ids
        pos     (n,2)  positions
        theta   (n,)   headings (radians)
        labels  (n,)   group index each robot belongs to, indexes into sizes/alignment/centroid
        sizes   (g,)   member count per group
        alignment (g,) nematic order parameter per group |<exp(2i*theta)>| ∈ [0,1]
        centroid  (g,2) centroid per group
        groups  raw return of compute_frame_groups(), use directly for more detail
    """
    # Lazy import: pos_all_grouping pulls in pandas, which strategy shouldn't pay
    # for when it isn't enabled.
    from pos_all_grouping import compute_frame_groups

    ids   = np.asarray(ids)
    pos   = np.asarray(pos, dtype=float).reshape(-1, 2)
    theta = np.asarray(theta, dtype=float)

    groups = compute_frame_groups(pos, theta, ids, max_dist,
                                  include_singletons=include_singletons)

    labels = np.full(len(ids), -1, dtype=int)      # -1 = not in any group
    for gi, g in enumerate(groups):
        labels[g["local_idx"]] = gi

    return {
        "ids":       ids,
        "pos":       pos,
        "theta":     theta,
        "labels":    labels,
        "sizes":     np.array([g["size"] for g in groups], dtype=int),
        "alignment": np.array([g["alignment"] for g in groups], dtype=float),
        "centroid":  np.array([g["centroid"] for g in groups],
                              dtype=float).reshape(-1, 2),
        "groups":    groups,
    }


def _check_selector_kwargs(selector, kwargs, where):
    """
    Early validation of a selector's extra kwargs. A misspelled parameter name
    (e.g. passing n_per_end to convex_hull) would normally only raise a TypeError
    at call time, but strategy is constructed inside a gait callback, where the
    exception gets swallowed by GaitController.poll into a single warning line,
    after which nothing happens silently every frame. Raise it here instead.
    """
    if not kwargs:
        return
    fn = SELECTORS[selector] if isinstance(selector, str) else selector
    if isinstance(selector, str) and selector not in SELECTORS:
        raise ValueError(f"{where}: unknown selector {selector!r}; "
                         f"choices: {sorted(SELECTORS)}")
    def _named(f):
        # Named parameters only: **kw / *args themselves are not passable names
        return {name for name, prm in inspect.signature(f).parameters.items()
                if prm.kind not in (inspect.Parameter.VAR_KEYWORD,
                                    inspect.Parameter.VAR_POSITIONAL)}

    accepted = _named(fn)
    if any(prm.kind is inspect.Parameter.VAR_KEYWORD
           for prm in inspect.signature(fn).parameters.values()):
        # e.g. sel_largest_group_major_ends(ids, pos, rec=None, **kw), whose
        # kwargs ultimately land on the function it delegates to
        accepted |= _named(sel_largest_group_axis_ends)
    accepted -= {"ids", "pos", "rec"}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"{where}: selector {selector!r} doesn't accept parameters {unknown}; "
            f"it accepts {sorted(accepted)}")


def _in_range(value, rng):
    """rng = (lo, hi); either end being None means unbounded; both ends inclusive."""
    if rng is None:
        return True
    lo, hi = rng
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


# =============================================================================
# Layer 1: issue commands by group size (with hysteresis)
# =============================================================================

class GroupRuleLayer:
    """
    Match each robot's group size (and optionally nematic order) to a rule, and
    issue that rule's command.

    Needs hysteresis just like the role layer. Grouping is recomputed every frame,
    so a robot wobbling at a cluster's edge can have its Voronoi edge to the
    cluster fall under max_dist this frame and exceed it the next, bouncing the
    group size across the threshold. Issuing commands straight off the per-frame
    result would make it switch gaits every few frames — something the real robot
    can't do, and it would pollute the causal link between "group size" and
    "gait". So: only after n_ticks consecutive ticks matching the same rule is a
    robot considered **confirmed** to belong to that rule, and only then is the
    command issued.

    Rules are matched in list order, the first hit wins, so put narrower ranges
    first.
    """

    def __init__(self, rules, n_ticks=6, verbose=False, label="group"):
        """
        rules   [{"size_range": (lo, hi), "command": int,
                  "alignment_range": (lo, hi) optional,
                  "name": str optional}, ...]
        n_ticks how many consecutive ticks must hit before confirming (1 = no hysteresis)
        """
        self.rules = []
        for i, r in enumerate(rules):
            rule = dict(r)
            if "command" not in rule:
                raise ValueError(f"group rule #{i} is missing 'command'")
            rule.setdefault("size_range", None)
            rule.setdefault("alignment_range", None)
            rule.setdefault("name", f"rule{i}")
            self.rules.append(rule)

        self.n_ticks = max(1, int(n_ticks))
        self.verbose = verbose
        self.label   = label

        self.assigned = {}      # robot_id -> confirmed rule index
        self._streak  = {}      # robot_id -> (rule index, consecutive hit count)
        self.events   = []
        self.ticks    = 0

    def _match(self, size, alignment):
        for i, rule in enumerate(self.rules):
            if not _in_range(size, rule["size_range"]):
                continue
            if not _in_range(alignment, rule["alignment_range"]):
                continue
            return i
        return None

    def update(self, ts, rec) -> dict:
        """Return {robot_id: command}, only for confirmed robots."""
        self.ticks += 1
        ids    = rec["ids"]
        labels = rec["labels"]
        sizes  = rec["sizes"]
        align  = rec["alignment"]

        out = {}
        for k, rid in enumerate(ids):
            rid = int(rid)
            gi  = int(labels[k])
            hit = None if gi < 0 else self._match(int(sizes[gi]), float(align[gi]))

            prev_rule, streak = self._streak.get(rid, (None, 0))
            streak = streak + 1 if hit == prev_rule else 1
            self._streak[rid] = (hit, streak)

            if streak < self.n_ticks:
                # Not yet confirmed: keep the last confirmed result, don't let
                # boundary jitter leak through
                if rid in self.assigned:
                    out[rid] = self.rules[self.assigned[rid]]["command"]
                continue

            if hit is None:
                if rid in self.assigned:
                    old = self.assigned.pop(rid)
                    self._log(ts, "group_leave", rid, self.rules[old]["name"], gi)
                    if self.verbose:
                        print(f"[group] {rid} left {self.rules[old]['name']}")
                continue

            if self.assigned.get(rid) != hit:
                self.assigned[rid] = hit
                self._log(ts, "group_join", rid, self.rules[hit]["name"], gi)
                if self.verbose:
                    print(f"[group] {rid} -> {self.rules[hit]['name']} "
                          f"(group size {int(sizes[gi])}) cmd={self.rules[hit]['command']}")
            out[rid] = self.rules[hit]["command"]

        return out

    def _log(self, ts, event, rid, rule_name, gi):
        self.events.append({"Time": ts, "tick": self.ticks, "event": event,
                            "robot_id": rid, "rule": rule_name, "group": gi,
                            "layer": self.label})


# =============================================================================
# Merge: group > role > fallback
# =============================================================================

class LayeredStrategy:
    """
    A strategy object usable directly as a runtime gait callback:

        config.RUNTIME_GAIT_CONTROLLER = "gait_control:strategy"

    Recomputes grouping and updates each layer every tick, then merges the three
    layers into {robot_id: command}. Only issues a request when the target
    command differs from the current one and the robot has no pending switch, so
    at steady state this returns an empty dict rather than re-flushing the queue
    every tick.
    """

    def __init__(self, max_dist, leave_command,
                 group_rules=(), roles=(),
                 period=0.25, group_n_ticks=6,
                 include_singletons=True, verbose=False):
        """
        max_dist       Voronoi adjacency distance threshold (pixels). Two robots
                       count as connected only if they are Delaunay neighbors and
                       distance <= this value. Offline analysis uses the same parameter.
        leave_command  Layer 3 fallback: executed by robots not claimed by any
                       group rule or role
        group_rules    see GroupRuleLayer
        roles          [{"selector": str, "command": int,
                         "n_frames_join": int, "n_frames_leave": int,
                         "override_group": bool optional (default False),
                         "enabled_when": {"largest_group_size": (lo, hi)} optional,
                         "label": str optional,
                         remaining keys passed through as-is to the selector, e.g.
                         axis / n_per_end / min_anisotropy / min_group_size /
                         select_from}, ...]

                       Roles earlier in the list have higher priority (when two
                       roles both select the same robot, the earlier one's command
                       wins).

                       With override_group=False (default), the role ranks below
                       the group layer: as long as a robot is claimed by some
                       group_rule, the group command is what executes.
                       With override_group=True it's reversed, the role command
                       overrides the group command — use this when you want
                       "the two ends of a cluster" to run a specific gait no
                       matter how large the cluster is.
        period         how often to recompute, in seconds. The hysteresis "frame
                       count" counts ticks, so period=0.25 + n_frames_join=6 means
                       it takes ~1.5 seconds to confirm a role.
        """
        self.max_dist      = float(max_dist)
        self.leave_command = int(leave_command)
        self.period        = float(period)
        self.include_singletons = bool(include_singletons)
        self.verbose       = bool(verbose)

        self.group_layer = (GroupRuleLayer(group_rules, n_ticks=group_n_ticks,
                                           verbose=verbose)
                            if group_rules else None)

        self.roles = []
        for i, spec in enumerate(roles):
            spec = dict(spec)
            selector     = spec.pop("selector")
            command      = int(spec.pop("command"))
            enabled_when = spec.pop("enabled_when", None)
            override     = bool(spec.pop("override_group", False))
            label        = spec.pop("label", None)
            n_join       = spec.pop("n_frames_join", 6)
            n_leave      = spec.pop("n_frames_leave", 6)
            miss_tol     = spec.pop("miss_tolerance", 2)
            vb           = spec.pop("verbose", verbose)
            # Remaining kwargs pass through to the selector as-is; validate against
            # its signature first, otherwise a misspelled parameter name raises a
            # TypeError inside the callback that GaitController.poll swallows into
            # a single warning line, after which nothing happens silently every
            # frame — very hard to debug.
            _check_selector_kwargs(selector, spec, f"roles[{i}]")
            tracker = SpatialRoleTracker(
                selector=selector, n_frames_join=n_join,
                n_frames_leave=n_leave, miss_tolerance=miss_tol,
                verbose=vb, label=label, **spec)
            self.roles.append({"tracker": tracker, "command": command,
                               "enabled_when": enabled_when,
                               "override_group": override})

        self._next_t = 0.0
        self.rec     = None       # last frame's grouping result, inspect directly when debugging
        self.ticks   = 0

    # ── Callback interface ──────────────────────────────────────────────────

    def __call__(self, t, robots, ctx):
        if t < self._next_t:
            return None
        self._next_t = t + self.period
        self.ticks += 1

        ids   = np.fromiter((r.id for r in robots), dtype=int, count=len(robots))
        pos   = np.array([(r.x, r.y) for r in robots], dtype=float).reshape(-1, 2)
        theta = np.fromiter((r.angle for r in robots), dtype=float,
                            count=len(robots))

        rec = frame_record(ids, pos, theta, self.max_dist,
                           include_singletons=self.include_singletons)
        self.rec = rec

        target = self.decide(t, rec, ids)

        # Only send what actually needs to change: empty here at steady state
        req = {}
        for r in robots:
            want = target.get(r.id)
            if want is None or want == r.command or r.pending_cmd is not None:
                continue
            req[r.id] = want
        return req or None

    def decide(self, t, rec, ids) -> dict:
        """
        Merge all three layers, return the command each robot **should** be
        executing (regardless of what it's currently executing).
        Can be called standalone to test the strategy without running the simulation.
        """
        target = {int(i): self.leave_command for i in ids}          # Layer 3

        # Update all role layers first (hysteresis counters must advance every
        # tick), record members, then write into target by priority afterward.
        largest = int(rec["sizes"].max()) if len(rec["sizes"]) else 0
        members = []
        for role in self.roles:
            enabled = True
            cond = role["enabled_when"]
            if cond is not None:
                enabled = _in_range(largest, cond.get("largest_group_size"))
            members.append(role["tracker"].update(t, rec["ids"], rec["pos"],
                                                  rec=rec, enabled=enabled))

        def _apply_roles(want_override):
            # Write in reverse order, so roles **earlier** in the list are
            # written last and overwrite later ones — this is what the docstring
            # means by "earlier has higher priority".
            for role, mids in zip(reversed(self.roles), reversed(members)):
                if role["override_group"] != want_override:
                    continue
                for mid in mids:
                    target[int(mid)] = role["command"]

        _apply_roles(False)                                         # Layer 2
        if self.group_layer is not None:                            # Layer 1
            for rid, cmd in self.group_layer.update(t, rec).items():
                target[int(rid)] = cmd
        _apply_roles(True)      # override_group roles sit on top of group commands

        return target

    # ── Debugging ────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """One-line overview of current grouping and role membership."""
        if self.rec is None:
            return "(no tick run yet)"
        sizes = sorted(self.rec["sizes"].tolist(), reverse=True)
        parts = [f"groups {len(sizes)} sizes {sizes[:8]}"
                 f"{'...' if len(sizes) > 8 else ''}"]
        for role in self.roles:
            tr = role["tracker"]
            parts.append(f"{tr.label}={sorted(tr.members)}->{role['command']:+d}")
        if self.group_layer is not None:
            n_assigned = len(self.group_layer.assigned)
            parts.append(f"grouped {n_assigned}")
        return "  |  ".join(parts)

    def events(self) -> list:
        """Event stream from all layers, sorted by time. In-memory only, not persisted."""
        out = list(self.group_layer.events) if self.group_layer else []
        for role in self.roles:
            out.extend(role["tracker"].events)
        return sorted(out, key=lambda e: (e["Time"], e.get("event", "")))


# =============================================================================
# Construct from declarative config
# =============================================================================

def build_strategy(spec) -> LayeredStrategy:
    """
    Turn a dict like config.STRATEGY_SPEC into a LayeredStrategy.
    Keys not given in spec fall back to LayeredStrategy's defaults.
    """
    spec = dict(spec or {})
    if "max_dist" not in spec or "leave_command" not in spec:
        raise ValueError("STRATEGY_SPEC must have at least 'max_dist' and 'leave_command'")
    return LayeredStrategy(**spec)

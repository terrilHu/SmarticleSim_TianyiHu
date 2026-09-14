"""
spatial_roles.py  ─  Pick out a set of "role" robots by spatial relationship.

For example "the ones with min/max x, min/max y" (the four extreme points of
the formation), or "the robots at the ends of the principal axis (PCA)
major/minor axis" (independent of the coordinate frame, follows the
formation's own orientation) -- the principal axis can be fit from every
robot in the field, or from only the current largest group to determine the
direction, then all robots are projected onto that direction to pick the two
ends.
These robots are fixed to run a single separate command, at a priority lower
than "belongs to a large group".

The relationship to group control is layered; strategy.py composes the final
command from them:

    Layer 1 (highest)  confirmed group members    -> that group's command
    Layer 2            spatial role                -> role command
    Layer 3 (fallback) everything else             -> leave_command

So a robot that is both the x-minimum AND in a large group runs the group's
command -- the role never overrides it.

Selectors are all pure functions that "eat one frame's (ids, pos) and spit
out a set of marker ids", so adding a new rule only requires writing a
function and registering it in SELECTORS -- no need to touch the controller.

This file keeps the same function names and semantics as its counterpart on
the real-robot side, so the same role definitions give comparable results in
simulation and on real robots; robots never go missing in simulation, so the
branches related to miss_tolerance never actually trigger -- they're kept
purely so the two sides stay structurally identical.
"""

from typing import Callable, Dict, List, Set

import numpy as np


def needs_group_record(fn):
    """
    Mark that this selector needs the whole frame's grouping result rec, in
    addition to (ids, pos).
    Uses an explicit marker rather than inferring it from the signature: it's
    obvious at a glance whether a hand-written selector has this decorator or
    not, and one without it still only ever receives (ids, pos) -- it won't
    get rec passed in accidentally from a typo'd parameter name.
    """
    fn.needs_rec = True
    return fn


# =============================================================================
# Selectors: (ids, pos) -> set(marker id)
#   ids (k,)  marker id
#   pos (k,2) pixel coordinates; note the image coordinate system has y pointing down
# =============================================================================

def sel_extremes(ids, pos) -> Set[int]:
    """Min/max x, min/max y -- the four extreme points of the formation.

    Ties take the first argmin/argmax, so the same robot may occupy two roles
    at once (e.g. it's both leftmost and topmost), in which case the returned
    set naturally contains just that one robot, fewer than 4.
    """
    if len(ids) == 0:
        return set()
    return {int(ids[i]) for i in (np.argmin(pos[:, 0]), np.argmax(pos[:, 0]),
                                  np.argmin(pos[:, 1]), np.argmax(pos[:, 1]))}


def sel_x_extremes(ids, pos) -> Set[int]:
    """Only take min/max x"""
    if len(ids) == 0:
        return set()
    return {int(ids[np.argmin(pos[:, 0])]), int(ids[np.argmax(pos[:, 0])])}


def sel_y_extremes(ids, pos) -> Set[int]:
    """Only take min/max y"""
    if len(ids) == 0:
        return set()
    return {int(ids[np.argmin(pos[:, 1])]), int(ids[np.argmax(pos[:, 1])])}


def sel_convex_hull(ids, pos) -> Set[int]:
    """Robots on the convex hull of the whole formation -- a more complete "edge" definition than the four extreme points"""
    if len(ids) < 3:
        return {int(i) for i in ids}
    try:
        from scipy.spatial import ConvexHull
        return {int(ids[i]) for i in ConvexHull(pos).vertices}
    except Exception:
        return sel_extremes(ids, pos)


def sel_farthest_from_centroid(ids, pos, n=4) -> Set[int]:
    """The n robots farthest from the centroid"""
    if len(ids) == 0:
        return set()
    d = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
    return {int(ids[i]) for i in np.argsort(d)[::-1][:n]}


def _ends_by_projection(ids, pos, V, cols, n_per_end):
    """Project points onto the given axes, taking n robots from each end of each axis.

    The origin used for projection doesn't affect the result: a translation
    just adds the same constant to every projection, and argsort is
    unaffected. So there's no need to fuss over whether to use the group's
    centroid or the whole field's centroid.
    """
    proj = np.asarray(pos, dtype=float) @ V
    out = set()
    k = max(1, int(n_per_end))
    for c in cols:
        order = np.argsort(proj[:, c])
        for i in order[:k]:
            out.add(int(ids[i]))
        for i in order[-k:]:
            out.add(int(ids[i]))
    return out


def principal_axes(pos):
    """
    Run principal component analysis (PCA) on the point cloud, return (center,
    axis standard deviations, eigenvectors, aspect ratio).

        sigma    ascending [σ_minor, σ_major], i.e. sqrt(eigenvalues), same units as the coordinates (pixels)
        eigvecs  column vectors; eigvecs[:,0]=minor-axis direction, eigvecs[:,1]=major-axis direction, mutually orthogonal
        ratio    σ_major/σ_minor, i.e. "how many times longer the major axis is than the minor axis"

    Standard deviation is returned here rather than the eigenvalue: the
    eigenvalue is variance, and its ratio is the **square** of the length
    ratio, which is easy to misread as a threshold (a ratio of 1.5 actually
    corresponds to only a 1.22:1 shape). After taking the square root, ratio
    is the intuitive aspect ratio.

    eigh is used instead of eig: the covariance matrix is symmetric, so eigh
    guarantees real solutions with eigenvalues in ascending order.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    ctr = pos.mean(axis=0)
    cov = np.cov((pos - ctr).T)
    w, V = np.linalg.eigh(cov)
    sigma = np.sqrt(np.maximum(w, 0.0))          # numerical error can produce tiny negative values
    ratio = float(sigma[1] / sigma[0]) if sigma[0] > 1e-9 else float("inf")
    return ctr, sigma, V, ratio


def sel_principal_ends(ids, pos, axis="both", n_per_end=1, min_anisotropy=1.5):
    """
    Fit the principal axes (PCA) from every robot's position, and take the
    robots at the ends of the major and/or minor axis.

        axis           "major" only takes the major-axis ends / "minor" only takes the minor-axis ends / "both" takes both axes
        n_per_end      how many robots to take from each end (the n farthest out by projection)
        min_anisotropy below this aspect ratio (σ_major/σ_minor) the formation is
                       judged "too round, principal-axis direction meaningless"
                       and an empty set is returned. This matters: near a
                       circle the eigenvector direction is driven by noise --
                       measured, at an aspect ratio of 1.1 the major-axis
                       direction swings by more than 40 degrees frame to
                       frame -- so without a threshold the selection would
                       just be two random robots. Default 1.5 corresponds to
                       a 1.5:1 shape.

    Note the principal-axis direction has a sign ambiguity (both V and -V are
    valid eigenvectors), but since both ends are taken together here, the
    selected set is independent of sign and won't flip between ends due to a
    sign flip.

    Returning an empty set means "this role cannot be defined this frame";
    SpatialRoleTracker treats that as "no information this frame" and keeps
    the current role membership unchanged rather than dismissing everyone.
    """
    ids = np.asarray(ids)
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    if len(ids) < 3:
        return {int(i) for i in ids}

    _ctr, sigma, V, ratio = principal_axes(pos)
    if sigma[1] < 1e-9:            # all points coincide, there isn't even a major axis
        return set()
    if ratio < min_anisotropy:     # too round, axis direction is driven by noise
        return set()

    cols = {"minor": [0], "major": [1], "both": [0, 1]}[axis]
    # when the minor-axis length is negligible relative to the major axis (the formation is nearly collinear),
    # the minor-axis ends are likewise noise, so skip that axis
    if sigma[0] < 1e-6 * sigma[1]:
        cols = [c for c in cols if c != 0]
        if not cols:
            return set()

    return _ends_by_projection(ids, pos, V, cols, n_per_end)


def sel_major_ends(ids, pos, **kw):
    """Major-axis ends -- the leading and trailing robots along the formation's most-extended direction"""
    return sel_principal_ends(ids, pos, axis="major", **kw)


def sel_minor_ends(ids, pos, **kw):
    """Minor-axis ends -- the two side edges along the formation's narrowest direction"""
    return sel_principal_ends(ids, pos, axis="minor", **kw)


@needs_group_record
def sel_largest_group_axis_ends(ids, pos, rec=None, axis="both", n_per_end=1,
                                min_anisotropy=1.5, min_group_size=3,
                                max_group_size=None, min_lead=0,
                                select_from="all"):
    """
    Fit the principal axes from **only the current largest group**, take just
    its **direction**, then project all robots onto those two directions and
    pick the individuals at the ends.

    Splitting this into two steps is intentional:
      - direction is decided by that group ── robots outside it and small
        groups don't skew the axis; what's measured is the group's own
        orientation;
      - the endpoints default to being picked from **every robot in the
        field** (select_from="all") ── the ones selected may be robots
        outside the group that are farther out along that direction, which is
        exactly what "leading/trailing along the group's orientation" means.
        Set select_from to "group" to pick only from within the group.

        axis            "major" / "minor" / "both"
        n_per_end       how many robots to take from each end
        min_anisotropy  below this aspect ratio for the group, direction is driven by noise -> returns an empty set
        min_group_size  below this threshold for the largest group's size -> returns an empty set (no group worth tracking has formed yet)
        max_group_size  above this cap for the largest group's size -> returns an empty set. None = no cap.
                        Usually easier to control via the role layer's
                        group_size_range; kept here so the selector can also
                        lock the range when used standalone
        min_lead        the largest group must exceed the second-largest by at
                        least this many robots to count. When two groups are
                        close in size, "which one is largest" flips frame to
                        frame and the axis jumps wholesale to a different
                        cluster of robots; set 1~2 to treat this case as
                        undetermined. Default 0 = no requirement.
        select_from     "all" picks endpoints across the whole field (default) / "group" picks only within that group

    Returning an empty set means "this role cannot be defined this frame";
    SpatialRoleTracker keeps the current role membership unchanged.
    """
    if rec is None or not len(rec.get("sizes", ())):
        return set()
    sizes = np.asarray(rec["sizes"])
    order = np.argsort(-sizes, kind="mergesort")   # stable sort: preserves the original order when sizes are equal
    gi = int(order[0])
    if int(sizes[gi]) < min_group_size:
        return set()
    if max_group_size is not None and int(sizes[gi]) > max_group_size:
        return set()
    if min_lead > 0 and len(order) > 1 and int(sizes[gi]) - int(sizes[order[1]]) < min_lead:
        return set()

    all_ids = np.asarray(rec["ids"])
    all_pos = np.asarray(rec["pos"], dtype=float).reshape(-1, 2)
    sel = np.asarray(rec["labels"]) == gi
    if sel.sum() < 3:
        return set()                      # too few points to fit a direction

    # Step 1: use only this group's members to determine direction
    _ctr, sigma, V, ratio = principal_axes(all_pos[sel])
    if sigma[1] < 1e-9 or ratio < min_anisotropy:
        return set()
    cols = {"minor": [0], "major": [1], "both": [0, 1]}[axis]
    if sigma[0] < 1e-6 * sigma[1]:        # this group is nearly collinear, minor-axis direction is meaningless
        cols = [c for c in cols if c != 0]
        if not cols:
            return set()

    # Step 2: project robots onto these two directions and take the ends
    if select_from == "group":
        return _ends_by_projection(all_ids[sel], all_pos[sel], V, cols, n_per_end)
    return _ends_by_projection(all_ids, all_pos, V, cols, n_per_end)


@needs_group_record
def sel_largest_group_major_ends(ids, pos, rec=None, **kw):
    """Take the robots at the ends along the largest group's major-axis direction"""
    return sel_largest_group_axis_ends(ids, pos, rec, axis="major", **kw)


@needs_group_record
def sel_largest_group_minor_ends(ids, pos, rec=None, **kw):
    """Take the robots at the ends along the largest group's minor-axis direction"""
    return sel_largest_group_axis_ends(ids, pos, rec, axis="minor", **kw)


SELECTORS: Dict[str, Callable] = {
    "extremes": sel_extremes,
    "x_extremes": sel_x_extremes,
    "y_extremes": sel_y_extremes,
    "convex_hull": sel_convex_hull,
    "farthest": sel_farthest_from_centroid,
    "principal_ends": sel_principal_ends,     # major + minor axis, four ends total
    "major_ends": sel_major_ends,             # major-axis ends only
    "minor_ends": sel_minor_ends,             # minor-axis ends only
    # the three below fit the principal axes using only "the current largest group", unaffected by individuals outside it or small groups
    "group_axis_ends": sel_largest_group_axis_ends,
    "group_major_ends": sel_largest_group_major_ends,
    "group_minor_ends": sel_largest_group_minor_ends,
}


# =============================================================================
# Role tracking with hysteresis
# =============================================================================

class SpatialRoleTracker:
    """
    Maintains the array of robots "currently holding a spatial role".

    An extreme point can change hands every frame from just a pixel or two of
    jitter ── when two robots' x coordinates differ by only 3px, which one is
    the minimum is pure noise. So, as with grouping, hysteresis is applied via
    a run of consecutive frames:

        selected for n_frames_join  consecutive frames -> joins the array
        not selected for n_frames_leave consecutive frames -> leaves the array

    The cost is that membership count is no longer always equal to what the
    selector returns, and can drift either way:
      - too many: during a handover the old one hasn't left yet while the new
        one has already joined;
      - too few: when two robots strictly alternate leading, neither
        accumulates enough consecutive frames, so neither holds the role
        (in this case "who is leftmost" is genuinely undecidable anyway --
        leaving it empty is better than swapping every frame).
    To force a strictly fixed role count, set n_frames_join to 1, at the cost
    of going back to changing hands every frame.
    """

    def __init__(self, selector="extremes", n_frames_join=6, n_frames_leave=6,
                 miss_tolerance=2, verbose=True, label=None, **kwargs):
        """
        label   the role layer's name, used to distinguish logs and event records when multiple layers coexist
        kwargs  passed straight through to the selector, e.g. axis / n_per_end / min_anisotropy for the PCA family
        """
        if callable(selector):
            self.selector = selector
            self.name = getattr(selector, "__name__", "custom")
        else:
            if selector not in SELECTORS:
                raise ValueError(f"unknown spatial-role selector: {selector!r}; "
                                 f"choices: {sorted(SELECTORS)}")
            self.selector = SELECTORS[selector]
            self.name = selector
        self.label = label or self.name
        self.kwargs = kwargs
        self.n_join = n_frames_join
        self.n_leave = n_frames_leave
        self.miss_tolerance = miss_tolerance
        self.verbose = verbose

        self.members: Set[int] = set()
        self.streak_in: Dict[int, int] = {}
        self.streak_out: Dict[int, int] = {}
        self.miss: Dict[int, int] = {}
        self.events: List[dict] = []
        self.frames = 0

    def update(self, ts, ids, pos, rec=None, enabled=True) -> Set[int]:
        """
        Eats one frame's (ids, pos), returns the current set of role members.
        rec is the whole frame's grouping result, passed through only to
        selectors marked with @needs_group_record.

        enabled=False means "this layer doesn't apply this frame" (e.g. the
        largest group's size fell outside the configured range). In that
        case it's treated as "explicitly not selected", so current members
        exit through the normal n_frames_leave hysteresis rather than being
        dismissed all at once ── this keeps commands from flapping when the
        size jitters right at the range boundary.
        """
        self.frames += 1
        ids = np.asarray(ids)
        pos = np.asarray(pos, dtype=float).reshape(-1, 2)

        if enabled:
            kw = dict(self.kwargs)
            if getattr(self.selector, "needs_rec", False):
                kw["rec"] = rec
            chosen = self.selector(ids, pos, **kw) if len(ids) else set()
        else:
            chosen = set()
        seen = {int(x) for x in ids}

        # An empty set from the selector has two possible meanings: no data
        # this frame, or a determination that "this role can't be defined
        # right now" (the PCA selectors do this when the formation is nearly
        # circular). Neither case should immediately dismiss all current
        # members ── that would cause commands to flap. Treated here as "no
        # information this frame", counters are left as-is. But enabled=False
        # is an explicit "not applicable" and must go through the exit flow,
        # so it's excluded from this.
        if enabled and not chosen and len(ids):
            return set(self.members)

        # Robots not seen this frame are tolerated for a few frames before counting as "not selected", to avoid a missed detection immediately kicking the role out
        tracked = self.members | set(self.streak_in) | set(self.streak_out)
        for mid in tracked - seen:
            self.miss[mid] = self.miss.get(mid, 0) + 1
        for mid in seen:
            self.miss[mid] = 0

        candidates = seen | {m for m in self.members
                             if self.miss.get(m, 0) <= self.miss_tolerance}
        for mid in candidates:
            if mid in chosen:
                self.streak_in[mid] = self.streak_in.get(mid, 0) + 1
                self.streak_out[mid] = 0
            elif self.miss.get(mid, 0) <= self.miss_tolerance:
                self.streak_out[mid] = self.streak_out.get(mid, 0) + 1
                self.streak_in[mid] = 0

        for mid in sorted(candidates):
            if mid not in self.members and self.streak_in.get(mid, 0) >= self.n_join:
                self.members.add(mid)
                self._log(ts, "role_join", mid)
                if self.verbose:
                    print(f"[role] {mid} joined role layer {self.label}, "
                          f"current members {sorted(self.members)}")
            elif mid in self.members and self.streak_out.get(mid, 0) >= self.n_leave:
                self.members.discard(mid)
                self._log(ts, "role_leave", mid)
                if self.verbose:
                    print(f"[role] {mid} left role layer {self.label}, "
                          f"current members {sorted(self.members)}")

        # a robot unseen for a long time is removed outright -- it may have exited the field of view
        for mid in list(self.members):
            if self.miss.get(mid, 0) > max(self.n_leave, self.miss_tolerance):
                self.members.discard(mid)
                self._log(ts, "role_lost", mid)
        return set(self.members)

    def reset(self):
        self.members.clear()
        self.streak_in.clear()
        self.streak_out.clear()
        self.miss.clear()

    def _log(self, ts, event, mid):
        self.events.append({"Time": ts, "frame": self.frames, "event": event,
                            "marker_id": mid, "role": self.label, "selector": self.name,
                            "members": " ".join(str(m) for m in sorted(self.members))})

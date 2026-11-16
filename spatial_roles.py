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

Because of that ordering, strategy.py runs Layer 1 first and hands the robots
it already claimed to the selector as ``exclude``. Every selector here takes
that set and **drops those robots from the pool it picks out of, without
changing the geometry it measures** -- the PCA axis is still fit from the
whole group, the centroid still from every robot; only the candidates change.
So a fixed-count selector (anything with n_per_end / n) keeps returning its
full count, drawn from robots the group layer has not taken, instead of
returning a set that is mostly overwritten one layer later. See
_ends_by_projection.

Selectors are all pure functions that "eat one frame's (ids, pos) and spit
out a set of marker ids", so adding a new rule only requires writing a
function and registering it in SELECTORS -- no need to touch the controller.

This file keeps the same function names and semantics as its counterpart on
the real-robot side, so the same role definitions give comparable results in
simulation and on real robots; robots never go missing in simulation, so the
branches related to miss_tolerance never actually trigger -- they're kept
purely so the two sides stay structurally identical.
"""

import inspect
from typing import Callable, Dict, List, Set

import numpy as np


def _accepts_kwarg(fn, name) -> bool:
    """Whether fn can be called with name=... -- either it names the parameter or it takes **kwargs."""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    return (name in params
            or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()))


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


def delegates_to(target, pinned=()):
    """
    Mark a thin wrapper as forwarding its **kw to target.

    The wrappers below (sel_major_ends, sel_x_midline, ...) exist only to pin
    one argument and pass the rest on, so their own signature says nothing
    about what a role may set. strategy.py validates a role's extra keys
    against this marker, which keeps that check correct no matter which
    function a wrapper delegates to -- it used to assume every **kw selector
    ended up at sel_largest_group_axis_ends, which quietly accepted
    min_group_size on major_ends and then raised TypeError a layer deeper.

    pinned names the arguments the wrapper fixes itself; passing one of them
    from a role spec would collide, so it is reported as unaccepted rather
    than left to fail at call time.
    """
    def deco(fn):
        fn.delegates_to = target
        fn.pinned_kwargs = tuple(pinned)
        return fn
    return deco


def _pool_mask(ids, exclude) -> np.ndarray:
    """Boolean mask over ids: True = still a candidate, False = excluded.

    exclude holds **robot ids**, not indices into ids -- the caller (the role
    layer) knows which robots a higher layer claimed, it doesn't know this
    frame's row order.
    """
    ids = np.asarray(ids)
    if exclude is None or len(exclude) == 0 or len(ids) == 0:
        return np.ones(len(ids), dtype=bool)
    ex = {int(x) for x in exclude}
    return np.fromiter((int(i) not in ex for i in ids), dtype=bool, count=len(ids))


# =============================================================================
# Selectors: (ids, pos) -> set(marker id)
#   ids (k,)  marker id
#   pos (k,2) pixel coordinates; note the image coordinate system has y pointing down
#
# Every selector takes exclude=<iterable of robot ids>: those robots are removed
# from the candidate pool but still count toward any geometry the selector
# measures (centroid, PCA axis, group membership).
# =============================================================================

def sel_axis_extremes(ids, pos, axis="both", n_per_end=1,
                      exclude=None) -> Set[int]:
    """
    The robots at the ends of a **coordinate** axis: n_per_end from each end.

    The frame-fixed counterpart of sel_principal_ends. Both project every
    robot onto an axis and take n from each end; there the axis is fitted by
    PCA and follows the formation's own orientation, here it is the world x or
    y axis and does not move. So this one never abstains the way the PCA family
    does on a round formation -- "the 30 leftmost and the 30 rightmost" is
    well defined whatever shape the swarm is in, whereas "the 30 at each end of
    the long axis" is not when there is no long axis.

        axis       "x" -> min/max x only, "y" -> min/max y only,
                   "both" -> both axes, i.e. up to 4*n_per_end robots
        n_per_end  how many to take from each end of each axis. The default 1
                   is the plain extreme-point behaviour this selector had
                   before the parameter existed.

    Shares _ends_by_projection with the PCA family, so it inherits the same
    exclude handling: a claimed robot's rank is filled by the next one in
    rather than lost, which is what keeps the count at n_per_end per end once
    the group layer has taken the true extremes.

    With axis="both" the ends can overlap -- a robot that is both leftmost and
    topmost occupies two of them -- so the set can hold fewer than
    4*n_per_end robots. Same for a pool smaller than 2*n_per_end on one axis,
    where the two ends meet and the union is the whole pool.
    """
    ids = np.asarray(ids)
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    if len(ids) == 0:
        return set()
    cols = {"x": [0], "y": [1], "both": [0, 1]}[axis]
    # Projecting onto the identity basis is just reading the coordinate off,
    # which is what makes this the same code path as the PCA selectors with a
    # fixed basis substituted for the fitted one.
    return _ends_by_projection(ids, pos, np.eye(2), cols, n_per_end, exclude)


@delegates_to(sel_axis_extremes, pinned=("axis",))
def sel_extremes(ids, pos, **kw) -> Set[int]:
    """Min/max x and min/max y -- the four extreme points of the formation (n_per_end each)"""
    return sel_axis_extremes(ids, pos, axis="both", **kw)


@delegates_to(sel_axis_extremes, pinned=("axis",))
def sel_x_extremes(ids, pos, **kw) -> Set[int]:
    """Only the x axis: the n_per_end leftmost and the n_per_end rightmost robots"""
    return sel_axis_extremes(ids, pos, axis="x", **kw)


@delegates_to(sel_axis_extremes, pinned=("axis",))
def sel_y_extremes(ids, pos, **kw) -> Set[int]:
    """Only the y axis: the n_per_end topmost and the n_per_end bottommost robots"""
    return sel_axis_extremes(ids, pos, axis="y", **kw)


def sel_convex_hull(ids, pos, exclude=None) -> Set[int]:
    """Robots on the convex hull of the whole formation -- a more complete "edge" definition than the four extreme points

    With exclude given, the hull is **re-fit on the remaining robots**: unlike
    the fixed-count selectors there is no "next candidate" to fall back on, so
    the only meaningful reading of "skip these and look again" is the boundary
    of what is left.
    """
    ids = np.asarray(ids)
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    idx = np.flatnonzero(_pool_mask(ids, exclude))
    if len(idx) < 3:
        return {int(ids[i]) for i in idx}
    try:
        from scipy.spatial import ConvexHull
        return {int(ids[idx[i]]) for i in ConvexHull(pos[idx]).vertices}
    except Exception:
        return sel_extremes(ids, pos, exclude=exclude)


def _resolve_center(pos, center) -> np.ndarray:
    """Reference point for the radial / midline selectors.

        None or "centroid"  the formation's own centroid, i.e. the reference
                            every other selector in this file measures against.
                            Moves with the swarm, so "outer" means "outer of
                            the swarm" even if the swarm has drifted to one
                            side of the arena.
        "arena"             the arena's geometric centre (W/2, H/2), fixed in
                            world coordinates for the whole trial. This is the
                            reference to use when the question is about the
                            arena -- "do groups form in the outer ring" is
                            about the ring, not about wherever the swarm
                            happens to sit.
        (x, y)              any explicit point, in the same pixel coordinates
                            as pos.

    config is imported lazily, inside the "arena" branch only: this module is
    otherwise pure geometry with no config dependency, which is what lets the
    selectors be unit-tested on hand-written point sets.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    if center is None or center == "centroid":
        return pos.mean(axis=0)
    if isinstance(center, str):
        if center != "arena":
            raise ValueError(f"center must be None, 'centroid', 'arena' or an "
                             f"(x, y) pair, not {center!r}")
        from config import H, W
        return np.array([W / 2.0, H / 2.0], dtype=float)
    return np.asarray(center, dtype=float).reshape(2)


def _radial_order(ids, pos, center, exclude) -> np.ndarray:
    """Pool indices into ids, ordered by distance from the reference point, inner first.

    The reference point is computed from **every** robot passed in, including
    any that `exclude` removes from the pool -- excluding a robot must not move
    the reference point, or the role would redefine "the middle" every time a
    higher layer claimed someone. Same rule as the PCA family.

    exclude is applied to the ordering after it is built, so the rank a claimed
    robot used to occupy is filled by the next one in rather than being lost --
    the same mechanism _ends_by_projection uses to hold its count.

    Ties are broken by argsort order, i.e. by position in `ids`. That only
    matters when two robots are equidistant to floating-point exactness, which
    in practice means the field is symmetric and either answer is as good.
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    d = np.linalg.norm(pos - _resolve_center(pos, center), axis=1)
    idx = np.flatnonzero(_pool_mask(ids, exclude))
    return idx[np.argsort(d[idx], kind="mergesort")]


def _by_centroid_distance(ids, pos, n, farthest, center=None,
                          exclude=None) -> Set[int]:
    """
    Shared core of the two centroid selectors: rank by distance to the
    reference point and take n from one end of that ranking.
    """
    ids = np.asarray(ids)
    if len(ids) == 0:
        return set()
    order = _radial_order(ids, pos, center, exclude)
    if len(order) == 0:
        return set()
    if farthest:
        order = order[::-1]
    return {int(ids[i]) for i in order[:max(1, int(n))]}


def sel_farthest_from_centroid(ids, pos, n=4, center=None,
                               exclude=None) -> Set[int]:
    """
    The n robots farthest from the centroid -- the formation's outermost
    stragglers, whichever direction they happen to lie in.

    Shape-agnostic, unlike the PCA family: it never abstains on a round
    formation, which makes it the fallback when group_*_ends gives up because
    the swarm has no meaningful long axis.

    center picks the reference point; see _resolve_center. Pass "arena" to mean
    "nearest the wall" rather than "outermost of the swarm".
    """
    return _by_centroid_distance(ids, pos, n, True, center, exclude)


def sel_nearest_to_centroid(ids, pos, n=4, center=None,
                            exclude=None) -> Set[int]:
    """
    The n robots closest to the centroid -- the core of the formation.

    The complement of sel_farthest_from_centroid against the same reference
    point, so the pair can drive an interior/exterior contrast: give the two
    roles different commands and the inner core and the outer shell run
    different gaits. Note they are complementary only in intent, not by
    construction -- with n_inner + n_outer < the robot count some robots fall
    through to leave_command, and past that the two sets would start to overlap
    in the middle, where the earlier role in `roles` wins.

    To give the core and the shell the *same* command, use sel_radial_shells
    instead -- one role, one command, no ordering to get right.
    """
    return _by_centroid_distance(ids, pos, n, False, center, exclude)


def sel_radial_shells(ids, pos, n_inner=4, n_outer=4, center=None,
                      exclude=None) -> Set[int]:
    """
    n_inner robots from the core plus n_outer robots from the rim, as one set.

    The radial counterpart of n_per_end: _ends_by_projection takes n robots
    from each end of an axis and returns the union, this takes n from each end
    of the *radial* ranking. The robots in the annulus between the two shells
    are left to the next layer, so one role covers "centre and rim behave
    alike, the middle does not" without needing two role entries and without
    depending on which of them comes first in `roles`.

        n_inner   how many to take from the inside out  (0 disables that shell)
        n_outer   how many to take from the outside in  (0 disables that shell)
        center    reference point; see _resolve_center. The default (the
                  swarm's own centroid) tracks the formation; "arena" pins the
                  rings to the arena instead, which is what you want if the
                  shells are meant to line up with the wall.

    With n_inner + n_outer >= the number of unclaimed robots the two shells
    meet and the union is simply the whole pool -- the same saturation
    _ends_by_projection has, and the same remedy: keep the counts well under
    the expected number of unclaimed robots.
    """
    ids = np.asarray(ids)
    if len(ids) == 0:
        return set()
    order = _radial_order(ids, pos, center, exclude)
    if len(order) == 0:
        return set()
    k_in  = max(0, int(n_inner))
    k_out = max(0, int(n_outer))
    out = {int(ids[i]) for i in order[:k_in]}
    if k_out:
        out |= {int(ids[i]) for i in order[-k_out:]}
    return out


def sel_midline(ids, pos, axis="x", n=4, band=None, center="arena",
                exclude=None) -> Set[int]:
    """
    The robots lying along a coordinate-axis centre line of the arena.

    Unlike the PCA family this line is fixed in **world** coordinates: axis="x"
    is the horizontal line y = y_ref, so the selection is "the robots in the
    middle band, left to right", and it does not rotate with the formation.
    That makes it the selector to use when the experiment is about an imposed
    direction (a left-right split, a seeded chain across the arena) rather than
    about whatever direction the swarm has arranged itself along -- and, being
    frame-fixed, it never abstains the way the PCA selectors do on a round
    formation.

        axis    "x" -> the line parallel to the x axis (constant y)
                "y" -> the line parallel to the y axis (constant x)
        n       take the n robots closest to the line. Fixed count, so the role
                is always filled; the band it spans widens as the swarm spreads
        band    half-width in pixels. When given it replaces n: every robot
                within +-band of the line is taken and the count varies with
                how many happen to be there -- including none, which the
                tracker reads as "no information this frame" and holds the
                current members (see SpatialRoleTracker.update)
        center  reference point; defaults to "arena" (the geometric centre,
                fixed for the whole trial) because an absolute midline is the
                point of this selector. Pass None for the swarm's centroid if
                you want the line to follow the formation instead; see
                _resolve_center

    Distance is measured off the line, not along it, so with axis="x" two
    robots at the far left and far right of the arena are equally "on" the
    midline as long as their y matches.
    """
    ids = np.asarray(ids)
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    if len(ids) == 0:
        return set()
    # axis="x" is the line of constant y, so the offset to minimise is in y
    col = {"x": 1, "y": 0}[axis]
    off = np.abs(pos[:, col] - _resolve_center(pos, center)[col])
    idx = np.flatnonzero(_pool_mask(ids, exclude))
    if len(idx) == 0:
        return set()
    if band is not None:
        return {int(ids[i]) for i in idx if off[i] <= float(band)}
    order = idx[np.argsort(off[idx], kind="mergesort")]
    return {int(ids[i]) for i in order[:max(1, int(n))]}


@delegates_to(sel_midline, pinned=("axis",))
def sel_x_midline(ids, pos, **kw) -> Set[int]:
    """The arena's horizontal centre line (constant y) -- a left-to-right middle band"""
    return sel_midline(ids, pos, axis="x", **kw)


@delegates_to(sel_midline, pinned=("axis",))
def sel_y_midline(ids, pos, **kw) -> Set[int]:
    """The arena's vertical centre line (constant x) -- a top-to-bottom middle band"""
    return sel_midline(ids, pos, axis="y", **kw)


def _ends_by_projection(ids, pos, V, cols, n_per_end, exclude=None):
    """Project points onto the given axes, taking n robots from each end of each axis.

    The origin used for projection doesn't affect the result: a translation
    just adds the same constant to every projection, and argsort is
    unaffected. So there's no need to fuss over whether to use the group's
    centroid or the whole field's centroid.

    exclude drops robots from the ordering **after** it is built, so the ends
    are the n outermost robots that are still available -- the rank a claimed
    robot used to occupy is filled by the next one in, rather than being lost.
    That is what keeps the count at n_per_end per end when a higher layer has
    already taken the true extremes.

    The count only falls short when fewer than 2*n_per_end candidates remain on
    an axis: the two ends then overlap and the union is the whole pool. With
    n_per_end=18 and 70 of 100 robots claimed by the group layer, 36 > 30, so
    30 is all you get -- keep n_per_end well under half the expected number of
    unclaimed robots.
    """
    ids = np.asarray(ids)
    proj = np.asarray(pos, dtype=float) @ V
    keep = _pool_mask(ids, exclude)
    out = set()
    k = max(1, int(n_per_end))
    for c in cols:
        order = np.argsort(proj[:, c])
        order = order[keep[order]]
        if len(order) == 0:
            continue
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


def sel_principal_ends(ids, pos, axis="both", n_per_end=1, min_anisotropy=1.5,
                       exclude=None):
    """
    Fit the principal axes (PCA) from every robot's position, and take the
    robots at the ends of the major and/or minor axis.

    exclude never enters the PCA -- the axis is the formation's own
    orientation and must not shift just because a higher layer claimed a few
    robots. It only removes them from the ends.

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
        return {int(i) for i in ids[_pool_mask(ids, exclude)]}

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

    return _ends_by_projection(ids, pos, V, cols, n_per_end, exclude)


@delegates_to(sel_principal_ends, pinned=("axis",))
def sel_major_ends(ids, pos, **kw):
    """Major-axis ends -- the leading and trailing robots along the formation's most-extended direction"""
    return sel_principal_ends(ids, pos, axis="major", **kw)


@delegates_to(sel_principal_ends, pinned=("axis",))
def sel_minor_ends(ids, pos, **kw):
    """Minor-axis ends -- the two side edges along the formation's narrowest direction"""
    return sel_principal_ends(ids, pos, axis="minor", **kw)


@needs_group_record
def sel_largest_group_axis_ends(ids, pos, rec=None, axis="both", n_per_end=1,
                                min_anisotropy=1.5, min_group_size=3,
                                max_group_size=None, min_lead=0,
                                select_from="all", exclude=None):
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
        exclude         robot ids a higher layer already claimed. They still
                        define the group and its axis -- only the endpoint pool
                        drops them. Note that with select_from="group" the pool
                        is the group itself, so if every member is claimed the
                        result is empty; that combination only makes sense when
                        the group rules leave part of the group unclaimed.

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
        return _ends_by_projection(all_ids[sel], all_pos[sel], V, cols,
                                   n_per_end, exclude)
    return _ends_by_projection(all_ids, all_pos, V, cols, n_per_end, exclude)


@needs_group_record
@delegates_to(sel_largest_group_axis_ends, pinned=("axis",))
def sel_largest_group_major_ends(ids, pos, rec=None, **kw):
    """Take the robots at the ends along the largest group's major-axis direction"""
    return sel_largest_group_axis_ends(ids, pos, rec, axis="major", **kw)


@needs_group_record
@delegates_to(sel_largest_group_axis_ends, pinned=("axis",))
def sel_largest_group_minor_ends(ids, pos, rec=None, **kw):
    """Take the robots at the ends along the largest group's minor-axis direction"""
    return sel_largest_group_axis_ends(ids, pos, rec, axis="minor", **kw)


# -----------------------------------------------------------------------------
# Every group above a size threshold, not just the largest one
# -----------------------------------------------------------------------------

def _qualifying_groups(rec, min_group_size, max_group_size=None) -> List[int]:
    """Indices (into rec["sizes"]) of the groups with min_group_size <= size <= max_group_size."""
    if rec is None or not len(rec.get("sizes", ())):
        return []
    sizes = np.asarray(rec["sizes"])
    ok = sizes >= int(min_group_size)
    if max_group_size is not None:
        ok &= sizes <= int(max_group_size)
    return np.flatnonzero(ok).tolist()


@needs_group_record
def sel_large_group_ends(ids, pos, rec=None, min_group_size=6,
                         max_group_size=None, axis="major", n_per_end=1,
                         min_anisotropy=1.5, exclude=None) -> Set[int]:
    """
    The two ends of **every** group whose size is >= min_group_size.

    Each qualifying group is handled on its own: PCA on that group's members
    gives its axis, and the n_per_end members farthest out along it at each
    end are taken. The result is the union over all qualifying groups, so it
    holds up to 2*n_per_end robots per group (4*n_per_end with axis="both").

    Differs from group_major_ends in two ways: that one looks only at the
    single largest group, and by default picks endpoints from the whole field;
    this one covers every group past the threshold and always picks **from
    within the group** -- the ends of a group are its own members.

        min_group_size  size threshold, inclusive. Groups below it are ignored.
                        Groups smaller than 3 never contribute: two points
                        have no meaningful axis to fit.
        max_group_size  optional upper bound, inclusive. None = no cap.
        axis            "major" (default, the group's long-axis ends) /
                        "minor" / "both"
        n_per_end       how many members to take from each end of each group.
                        A group with <= 2*n_per_end members is taken whole.
        min_anisotropy  a group rounder than this aspect ratio has no defined
                        axis and is **skipped** (the other groups still count).
                        Set to 1.0 to never skip.
        exclude         robots a higher layer claimed. They still shape each
                        group's axis but are dropped from its endpoint pool.

    The pool is the group itself, so if the group layer already claims every
    member (the usual case: group_rules cover all sizes >= 3), use this with
    "override_group": True -- otherwise every candidate is excluded and the
    role never fires.

    Returns an empty set when no group qualifies (or every qualifying group is
    too round). SpatialRoleTracker reads that as "no information" and holds the
    current members; add "enabled_when": {"largest_group_size": (min_group_size,
    None)} to the role if members should instead leave once no group is large
    enough.
    """
    gis = _qualifying_groups(rec, min_group_size, max_group_size)
    if not gis:
        return set()
    req_cols = {"minor": [0], "major": [1], "both": [0, 1]}[axis]
    all_ids = np.asarray(rec["ids"])
    all_pos = np.asarray(rec["pos"], dtype=float).reshape(-1, 2)
    labels = np.asarray(rec["labels"])

    out: Set[int] = set()
    for gi in gis:
        sel = labels == gi
        if sel.sum() < 3:
            continue
        _ctr, sigma, V, ratio = principal_axes(all_pos[sel])
        if sigma[1] < 1e-9 or ratio < min_anisotropy:
            continue                      # too round: this group's axis is noise
        cols = req_cols
        if sigma[0] < 1e-6 * sigma[1]:    # nearly collinear: minor axis is meaningless
            cols = [c for c in cols if c != 0]
        if not cols:
            continue
        out |= _ends_by_projection(all_ids[sel], all_pos[sel], V, cols,
                                   n_per_end, exclude)
    return out


@needs_group_record
def sel_large_group_centers(ids, pos, rec=None, min_group_size=6,
                            max_group_size=None, n=1,
                            exclude=None) -> Set[int]:
    """
    The centre of **every** group whose size is >= min_group_size: the n
    members of each group nearest that group's own centroid.

    The interior counterpart of sel_large_group_ends, with the same threshold
    and the same pool (the group's own members). Shape-agnostic, so unlike the
    ends it never skips a round group. The result is the union over all
    qualifying groups: up to n robots per group.

        min_group_size  size threshold, inclusive
        max_group_size  optional upper bound, inclusive. None = no cap.
        n               how many members to take from each group's centre
        exclude         robots a higher layer claimed. The centroid is still
                        computed from every member; claimed ones are only
                        dropped from the pool, so the next-nearest fills in.

    Same caveats as sel_large_group_ends: needs "override_group": True when the
    group layer claims these robots, and an empty result (no group qualifies)
    holds the current members unless enabled_when is set.
    """
    gis = _qualifying_groups(rec, min_group_size, max_group_size)
    if not gis:
        return set()
    all_ids = np.asarray(rec["ids"])
    all_pos = np.asarray(rec["pos"], dtype=float).reshape(-1, 2)
    labels = np.asarray(rec["labels"])
    k = max(1, int(n))

    out: Set[int] = set()
    for gi in gis:
        idx = np.flatnonzero(labels == gi)
        if len(idx) == 0:
            continue
        gpos = all_pos[idx]
        d = np.linalg.norm(gpos - gpos.mean(axis=0), axis=1)
        order = np.argsort(d, kind="mergesort")
        order = order[_pool_mask(all_ids[idx], exclude)[order]]
        out |= {int(all_ids[idx[i]]) for i in order[:k]}
    return out


SELECTORS: Dict[str, Callable] = {
    # world-axis ends, n_per_end from each: frame-fixed, so unlike the PCA
    # family below they never abstain on a round formation
    "extremes": sel_extremes,                 # both axes, up to 4*n_per_end
    "x_extremes": sel_x_extremes,             # leftmost + rightmost
    "y_extremes": sel_y_extremes,             # topmost + bottommost
    "convex_hull": sel_convex_hull,
    "farthest": sel_farthest_from_centroid,   # outermost n, by distance to the centroid
    "nearest": sel_nearest_to_centroid,       # innermost n, same reference point
    "radial_shells": sel_radial_shells,       # core + rim in one set, the radial n_per_end
    # frame-fixed centre lines: unlike the PCA family these do not rotate with
    # the formation, and never abstain on a round one
    "x_midline": sel_x_midline,               # horizontal line (constant y)
    "y_midline": sel_y_midline,               # vertical line (constant x)
    "principal_ends": sel_principal_ends,     # major + minor axis, four ends total
    "major_ends": sel_major_ends,             # major-axis ends only
    "minor_ends": sel_minor_ends,             # minor-axis ends only
    # the three below fit the principal axes using only "the current largest group", unaffected by individuals outside it or small groups
    "group_axis_ends": sel_largest_group_axis_ends,
    "group_major_ends": sel_largest_group_major_ends,
    "group_minor_ends": sel_largest_group_minor_ends,
    # every group with size >= min_group_size, each handled on its own, members
    # picked from within that group (usually wants override_group=True)
    "large_group_ends": sel_large_group_ends,         # each group's axis ends
    "large_group_centers": sel_large_group_centers,   # each group's centre
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
        # Every selector in SELECTORS takes exclude; a hand-written one may not,
        # and for those the excluded robots are filtered out of the result
        # instead (correct, but the count can come up short). Resolved once here
        # rather than per frame.
        self._takes_exclude = _accepts_kwarg(self.selector, "exclude")
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

    def update(self, ts, ids, pos, rec=None, enabled=True,
               exclude=None) -> Set[int]:
        """
        Eats one frame's (ids, pos), returns the current set of role members.
        rec is the whole frame's grouping result, passed through only to
        selectors marked with @needs_group_record.

        exclude is the set of robots a higher-priority layer already claimed
        this frame; the selector skips them and picks further in instead. A
        robot that is currently a member and then gets claimed simply stops
        being selected, so it leaves through the normal n_frames_leave
        hysteresis while its replacement joins through n_frames_join -- the
        handover is not instant, by design.

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
            if exclude and self._takes_exclude:
                kw["exclude"] = exclude
            chosen = self.selector(ids, pos, **kw) if len(ids) else set()
            if exclude and not self._takes_exclude:
                chosen = {int(c) for c in chosen} - {int(x) for x in exclude}
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
        # With exclude given there is a third way to land here: every candidate
        # was claimed by a higher layer. Holding membership is still the right
        # call -- those members are exactly the claimed robots, and the higher
        # layer overwrites their command anyway, so nothing leaks through.
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

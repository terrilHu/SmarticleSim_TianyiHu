"""
sweep_coverage.py  ─  arm sweep coverage ratio k = A' / A_total.

Definition
----------
Each robot's **footprint** consists of three pieces: the central body's rectangle,
plus one sector swept by each arm. The left arm is hinged at body-local coords
(-main_len/2, 0) and points -x when the joint angle is 0; the right arm is hinged
at (+main_len/2, 0), pointing +x. The joint angle swings between [-A, +A], so each
arm sweeps a sector centered at its shoulder point, radius arm_len, half-angle 2A.

    A_0 = main_len * main_w + (A1 + A2) * arm_len^2

The area of a sector with half-angle A is (1/2)*r^2*(2A) = A*r^2, so the two arms
together contribute (A1+A2)*arm_len^2. A_0 depends only on amplitude (and the
fixed geometry) — position and heading don't affect it.

    A'      = area of the **union** of all n footprints in the plane (overlap
              counted once, A' <= n*A_0)
    A_total = interior area of the ring
    k       = A' / A_total

Larger k means footprints fill more of the arena and overlap more, i.e. more
intense interaction and collisions.

The union is **not clipped to the ring's edge**: for a robot near the wall, the
part of its sector extending past the wall still counts toward A'. So k can
slightly exceed 1 — the denominator is the ring's area, but the numerator can
cover up to one footprint_radius beyond the ring. The canvas is sized larger than
the ring by that same margin for this reason, otherwise the union would get
silently truncated at the canvas boundary (which would just move the clipping
problem elsewhere).

Why A_0 is exact when A <= pi/2
--------------------------------
The sector is centered at a body endpoint and opens outward along ±x; when
A <= 90 degrees the whole sector lies in the half-plane x <= -main_len/2 (or
x >= +main_len/2), while the body rectangle occupies |x| <= main_len/2, so the
three pieces are pairwise non-overlapping and areas can simply be summed.
JOINT_LIMIT_DEG is 85 degrees and the amplitude table maxes out at pi/2, both
within this range. For A > pi/2 the formula overestimates and the function emits
a warning.

How the union is computed
--------------------------
An exact union of n "rectangle + two sectors" shapes needs a geometry library like
shapely, which isn't available here, so it's done by **rasterization**: lay a
cell x cell boolean canvas over the ring's bounding box, OR each robot's footprint
into it, and count covered cells at the end. Error is O(perimeter * cell); smaller
cell is more accurate — calibrate measures the convergence empirically, see the
comment on COVERAGE_CELL.

Testing is only done within each robot's own bounding box, so per-frame cost is
n * (2*L_s/cell)^2 grid points, not the whole canvas x n.
"""

import math

import numpy as np


# =============================================================================
# Single robot: analytical area and grid test
# =============================================================================

def footprint_area(sm) -> float:
    """
    A_0 — analytical area of a single robot's footprint (body rectangle + two sectors).

    Reads the instance's **current** A1/A2, so if gait amplitude changes at
    runtime, A_0 changes with it.
    """
    return float(sm.main_len * sm.main_w
                 + (sm.A1 + sm.A2) * sm.arm_len ** 2)


def footprint_radius(sm) -> float:
    """Circumscribing radius of the footprint (origin at body center), used for the bounding box."""
    # The farthest point is the arm tip: shoulder to center is main_len/2, plus arm length.
    return 0.5 * sm.main_len + sm.arm_len


def _mask_local(u, v, main_len, main_w, arm_len, A1, A2):
    """
    Test whether grid points fall inside the footprint, in the robot's body frame.

    u, v are arrays of the same shape (body coordinates, u along the body's long axis).
    Returns a boolean array of the same shape.

    Sector test avoids a square root: point (ru, rv) relative to the shoulder point
    falls inside a sector opening toward -x with half-angle A iff
    ru <= 0  and  rv^2 * cos^2(A) <= ru^2 * sin^2(A)  and  ru^2+rv^2 <= r^2.
    When A = pi/2, cos = 0 and the condition degenerates to a half-disk; when A = 0
    it degenerates to a line segment (zero area). Both extremes hold naturally, no
    special-casing needed.
    """
    half_len = 0.5 * main_len
    inside = (np.abs(u) <= half_len) & (np.abs(v) <= 0.5 * main_w)

    r2 = arm_len * arm_len

    # Left arm: shoulder at (-half_len, 0), pointing -x
    ru = u + half_len
    c2, s2 = math.cos(A1) ** 2, math.sin(A1) ** 2
    inside |= ((ru <= 0.0) & (ru * ru + v * v <= r2)
               & (v * v * c2 <= ru * ru * s2))

    # Right arm: shoulder at (+half_len, 0), pointing +x
    qu = u - half_len
    c2, s2 = math.cos(A2) ** 2, math.sin(A2) ** 2
    inside |= ((qu >= 0.0) & (qu * qu + v * v <= r2)
               & (v * v * c2 <= qu * qu * s2))

    return inside


# =============================================================================
# Ring
# =============================================================================

def ring_area(inner_r, shape="circle", n_sides=None) -> float:
    """A_total — interior area of the ring. Circle is pi*R^2; regular n-gon (circumradius R) is (n/2)R^2 sin(2pi/n)."""
    if (shape or "circle").lower() == "polygon":
        n = max(3, int(n_sides))
        return 0.5 * n * inner_r * inner_r * math.sin(2.0 * math.pi / n)
    return math.pi * inner_r * inner_r


# =============================================================================
# Coverage ratio
# =============================================================================

class CoverageMeter:
    """
    An object for computing k repeatedly: the canvas and ring mask are built once
    and reused every frame after that.

    Usage:
        meter = CoverageMeter(center, INNER_R, RING_SHAPE, RING_N_SIDES, cell=2.0)
        k, info = meter.measure(smarticles)
    """

    def __init__(self, center, inner_r, ring_shape="circle", n_sides=None,
                 cell=2.0):
        """
        cell  grid cell size (pixels). Smaller is more accurate, cost grows as 1/cell^2.

        The canvas is sized in measure() to the **actual** footprints' bounding
        box: it only grows, never shrinks, so at steady state each trial resizes
        it at most a few times. Grid points are always anchored to the (cx-R, cy-R)
        lattice, and pad is rounded up to whole cells, so growing the canvas never
        shifts grid-point phase — the k sequence won't show a step on the frame
        where the canvas resizes.

        Auto-sizing is what makes "no clipping" actually true: if the canvas were
        sized from the ring radius plus a fixed margin, robots that stray past
        that margin would get silently cut off at the canvas boundary — which
        would just move the clipping problem elsewhere.
        """
        self.cx, self.cy = float(center[0]), float(center[1])
        self.inner_r = float(inner_r)
        self.shape   = ring_shape
        self.n_sides = n_sides
        self.cell    = float(cell)

        self.A_total = ring_area(self.inner_r, self.shape, self.n_sides)

        self._pad    = -1.0
        self._canvas = None
        self.x0 = self.y0 = 0.0
        self.nx = self.ny = 0
        self._xs = self._ys = None

    def _ensure_canvas(self, need_pad):
        """
        Ensure the canvas covers the box of inner_r + need_pad around (cx, cy).

        pad is rounded up to a whole number of cells: x0 = cx - R - pad; when pad
        grows by whole cells the original grid points remain grid points (just an
        index shift), so the same footprint lands on the same cells before and
        after growth, and A' won't jump when the canvas resizes.
        """
        if self._canvas is not None and need_pad <= self._pad:
            return
        cell = self.cell
        pad = math.ceil(max(0.0, need_pad) / cell) * cell
        self._pad = pad
        half = self.inner_r + pad
        self.x0 = self.cx - half
        self.y0 = self.cy - half
        self.nx = int(math.ceil(2.0 * half / cell)) + 1
        self.ny = self.nx
        # Grid point center coordinates (1D); combine into 2D via broadcasting to
        # avoid allocating a full nx*ny float array
        self._xs = self.x0 + (np.arange(self.nx) + 0.5) * cell
        self._ys = self.y0 + (np.arange(self.ny) + 0.5) * cell
        self._canvas = np.zeros((self.ny, self.nx), dtype=bool)

    # ── Main interface ───────────────────────────────────────────────────────

    def measure(self, smarticles):
        """
        Returns (k, info). info contains:
            A_union       union area (rasterized estimate, not clipped to the ring)
            A0_sum        sum(A_0), the analytical value by definition
            A0_sum_raster rasterized value of sum(A_0) — same discretization as A_union
            A_total       ring area (analytical value)
            overlap       1 - A_union / A0_sum_raster, fraction lost to overlap
            k_naive       A0_sum / A_total, the upper bound ignoring overlap

        overlap uses A0_sum_raster rather than the analytical A0_sum: both carry
        the same rasterization error, which cancels out in the ratio. Comparing
        the rasterized A_union against the analytical A0_sum would leave a 1~2%
        systematic bias — empirically, two fully separated robots would report
        1.8% "overlap".
        """
        if not len(smarticles):
            return 0.0, {"A_union": 0.0, "A0_sum": 0.0, "A0_sum_raster": 0.0,
                         "A_total": self.A_total, "overlap": 0.0,
                         "k_naive": 0.0}

        # Measure all footprints' bounding boxes first, then decide the canvas size
        plan = []
        need = 0.0
        for sm in smarticles:
            pp  = sm.main_body.position
            rad = footprint_radius(sm)
            plan.append((sm, float(pp.x), float(pp.y),
                         float(sm.main_body.angle), rad))
            need = max(need,
                       abs(pp.x - self.cx) + rad - self.inner_r,
                       abs(pp.y - self.cy) + rad - self.inner_r)
        self._ensure_canvas(need)

        canvas = self._canvas
        canvas.fill(False)

        cell, x0, y0 = self.cell, self.x0, self.y0
        nx, ny = self.nx, self.ny
        A0_sum = 0.0
        A0_cells = 0

        for sm, px, py, psi, rad in plan:
            A0_sum += footprint_area(sm)

            # This robot's index window on the canvas (with one cell of margin)
            i0 = int(math.floor((px - rad - x0) / cell))
            i1 = int(math.ceil((px + rad - x0) / cell)) + 1
            j0 = int(math.floor((py - rad - y0) / cell))
            j1 = int(math.ceil((py + rad - y0) / cell)) + 1
            i0 = 0 if i0 < 0 else i0
            j0 = 0 if j0 < 0 else j0
            i1 = nx if i1 > nx else i1
            j1 = ny if j1 > ny else j1
            if i0 >= i1 or j0 >= j1:
                continue                      # entirely outside the canvas

            dx = self._xs[i0:i1][None, :] - px
            dy = self._ys[j0:j1][:, None] - py

            # Convert to body coordinates: rotate by -psi
            c, s = math.cos(psi), math.sin(psi)
            u =  dx * c + dy * s
            v = -dx * s + dy * c

            mask = _mask_local(u, v, sm.main_len, sm.main_w,
                               sm.arm_len, sm.A1, sm.A2)
            # Count this robot's own cell count while we're at it: mask is already
            # computed, and using it keeps the overlap calc on the same
            # discretization as A_union.
            A0_cells += int(mask.sum())
            canvas[j0:j1, i0:i1] |= mask

        cell2 = cell * cell
        A_union = float(canvas.sum()) * cell2
        A0_raster = float(A0_cells) * cell2
        k = A_union / self.A_total if self.A_total > 0 else float("nan")
        return k, {
            "A_union":       A_union,
            "A0_sum":        A0_sum,
            "A0_sum_raster": A0_raster,
            "A_total":       self.A_total,
            "overlap": (1.0 - A_union / A0_raster) if A0_raster > 0 else 0.0,
            "k_naive": A0_sum / self.A_total if self.A_total > 0 else float("nan"),
        }


def coverage_ratio(smarticles, center, inner_r, ring_shape="circle",
                   n_sides=None, cell=2.0):
    """Compute k for a single frame (internally just builds a CoverageMeter)."""
    return CoverageMeter(center, inner_r, ring_shape, n_sides,
                         cell=cell).measure(smarticles)

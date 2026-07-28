"""
analysis.py  ─  Data analysis, order-parameter calculation, result I/O,
                video recording, and rendering helpers.
"""

import csv
import math
import os
import sys
from functools import lru_cache

import numpy as np
import pygame
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import ConvexHull

from config import (INNER_R, N_SMARTICLES, VIDEO_FPS, VIDEO_STRIDE,
                    VIDEO_DOWNSCALE, VIDEO_CODEC)

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# =============================================================================
# Math / physics helpers used in data collection
# =============================================================================

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-10.0 * x))


def wrap_pi(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def dist_to_inner_wall(pos, center, inner_r: float):
    """Returns (gap, radial_distance).  gap > 0 means inside the ring."""
    r = (pos - center).length
    return inner_r - r, r


def wall_strength(d: float, Ls: float, k: float = 1.0, d0: float = 0.0):
    x = (d0 - d) / (Ls + 1e-9)
    sv = sigmoid(x)
    return k * sv, k * sv * (1.0 - sv)


# =============================================================================
# Order parameters
# =============================================================================

def calculate_orientational_order(dPsi):
    # np.asarray avoids copying when the input is already an ndarray row.
    arr = np.asarray(dPsi)
    op     = np.abs(np.mean(np.exp(1j * arr)))
    op_abs = np.abs(np.mean(np.exp(1j * 2.0 * arr)))
    return op, op_abs


def rotation_order_parameters(Psis, positions, center):
    thetas  = [np.arctan2(pos.y - center.y, pos.x - center.x) for pos in positions]
    Lambda3 = wrap_pi(Psis - np.array(thetas))
    return np.abs(np.mean(np.exp(1j * Lambda3)))


def calculate_macro_shape(positions):
    """Return (Rg, kappa_squared, convex_hull_area)."""
    com        = np.mean(positions, axis=0)
    pos_s      = positions - com
    T          = np.dot(pos_s.T, pos_s) / len(positions)
    eigenvalues = np.linalg.eigvals(T)
    l1, l2     = np.sort(eigenvalues)[::-1]
    rg_sq      = l1 + l2
    rg         = np.sqrt(rg_sq)
    kappa      = (1.0 - 4.0 * (l1 * l2) / (rg_sq ** 2)) if rg_sq > 0 else 0.0
    try:
        area = ConvexHull(positions).volume
    except Exception:
        area = 0.0
    return rg, kappa, area


def compute_gr(dxy_matrix, r_max: float = 2 * INNER_R, bins: int = 50):
    dists       = np.ravel(dxy_matrix)   # view, not a copy
    hist, edges = np.histogram(dists, bins=bins, range=(0, r_max))
    r           = 0.5 * (edges[:-1] + edges[1:])
    gr          = hist / (np.sum(hist) + 1e-8)
    return r, gr


# =============================================================================
# Interaction / actuation metrics
# =============================================================================

@lru_cache(maxsize=4096)
def _actuation_impact_cached(freq1, freq2, Amp1, Amp2):
    """
    Memoised kernel of :func:`actuationimpactCalculation`.

    The quantity is a *pure* function of the four actuation parameters (no RNG,
    no global state), and those parameters are fixed for the whole trial, so the
    result can be cached.  The original code re-evaluated this integral once per
    robot *pair* per frame — O(n^2) scipy calls per frame — which dominated the
    runtime.  Caching returns bit-identical values.
    """
    sample_time = max(1 / freq1, 1 / freq2)
    t      = np.arange(0, sample_time, 0.01)
    theta1 = Amp1 * np.sin(2 * math.pi * freq1 * t)
    theta2 = Amp2 * np.sin(2 * math.pi * freq2 * t)
    d1 = cumulative_trapezoid(np.abs(theta1 - theta2), t, initial=0)[-1]
    d2 = cumulative_trapezoid(np.abs(theta1 - theta1), t, initial=0)[-1]
    return (d1, d2)


def actuationimpactCalculation(freq1, freq2, Amp1, Amp2):
    """Same signature / return value as before; now memoised (see above)."""
    try:
        d1, d2 = _actuation_impact_cached(float(freq1), float(freq2),
                                          float(Amp1), float(Amp2))
    except TypeError:                      # unhashable / exotic input -> no cache
        return list(_actuation_impact_cached.__wrapped__(freq1, freq2, Amp1, Amp2))
    return [d1, d2]


def actutaionDetermination(dx, dy, dPsi, inter1, inter2):
    sign1 = np.sign(math.tan(dy / dx))
    sign2 = np.sign(dPsi)
    sign  = sign1 * sign2
    return 0.5 * max((1 - sign) * inter1, (1 + sign) * inter2)


# =============================================================================
# Vectorised pair-interaction kernel  (O(n^2) work, but done in NumPy)
# =============================================================================

def build_pair_matrices(px, py, psis, acts0, acts1,
                        Ls_i, Lw_i, Smain_i,
                        a0, a1, g0, WC):
    """
    Vectorised replacement for the nested ``for i / for j`` pair loop that used
    to live in ``simulation.run_trial``.

    Parameters
    ----------
    px, py   : (n,) float64 — main-body positions
    psis     : (n,) float64 — main-body angles
    acts0/1  : (n,) float64 — per-robot ``actuationimpactCalculation`` results
    Ls_i     : (n,) float64 — per-robot half-span  (L_s)
    Lw_i     : (n,) float64 — per-robot body width (L)
    Smain_i  : (n,) float64 — per-robot body length (S)

    Returns
    -------
    dx_m, dy_m, dPsi_m, dxy_m, Aij_m : (n, n) float64

    Equivalence
    -----------
    Reproduces the original element-for-element, including the antisymmetry
    conventions the scalar code used:

      * ``dx_m[i][j] = px[i]-px[j]`` and ``dx_m[j][i] = -dx_m[i][j]``
      * ``dPsi_m[i][j] = wrap_pi(psis[j]-psis[i])``, lower triangle set to the
        exact negation (NOT recomputed, because ``wrap_pi`` is not odd at +-pi)
      * ``Aij_m`` is symmetric and built from the ``i < j`` orientation, so the
        per-robot actuation term always comes from the *lower-indexed* robot
      * diagonals stay 0, exactly as ``np.zeros`` left them
    """
    n = px.shape[0]

    # NOTE on np.float_power: the scalar code raised Python/NumPy *scalars* to
    # the power 2, which goes through C pow().  Array ``x ** 2`` instead lowers
    # to x*x, which differs from pow(x, 2.0) in the last bit for some inputs.
    # np.float_power always uses pow() in double precision, so it reproduces
    # the original element-for-element.  It is slower than x*x, but it is only
    # used where the original really did exponentiate.
    dx_m = px[:, None] - px[None, :]
    dy_m = py[:, None] - py[None, :]

    # Upper triangle of wrap_pi(psis[j] - psis[i]); lower triangle = -upper.
    dpsi_full = wrap_pi(psis[None, :] - psis[:, None])
    dPsi_m = np.triu(dpsi_full, 1)
    dPsi_m = dPsi_m - dPsi_m.T

    # |r_i - r_j| is sign-independent, so the full matrix is already symmetric
    # and bit-identical to the scalar math.sqrt(dx*dx + dy*dy).
    dxy_m = np.sqrt(dx_m * dx_m + dy_m * dy_m)

    with np.errstate(divide="ignore", invalid="ignore"):
        # actutaionDetermination, vectorised.  Only the sign of tan() is used.
        sign = np.sign(np.tan(dy_m / dx_m)) * np.sign(dPsi_m)
        inter = 0.5 * np.maximum((1 - sign) * acts0[:, None],
                                 (1 + sign) * acts1[:, None])

        Ls_ij = 0.5 * (Ls_i[:, None] + Ls_i[None, :])
        L_ij  = 0.5 * (Lw_i[:, None] + Lw_i[None, :])
        R0_ij = Ls_ij + 0.01 * (0.5 * (Smain_i[:, None] + Smain_i[None, :]))

        A_up = (np.float_power(L_ij, 2.0)
                / (np.float_power(dx_m, 2.0) + np.float_power(dy_m, 2.0))
                * ((a0 + a1 * inter)
                   * (np.float_power(np.sin(dPsi_m / 2), 2.0) + g0)
                   + WC * sigmoid((R0_ij - dxy_m) / Ls_ij)))

    # Keep only i<j (the diagonal held 0/0 -> nan) and mirror it.
    Aij_m = np.triu(A_up, 1)
    Aij_m = Aij_m + Aij_m.T
    return dx_m, dy_m, dPsi_m, dxy_m, Aij_m


def calculate_sync(smarticles):
    sync = []
    for idx1 in range(len(smarticles)):
        sync1 = 0.0
        temp  = smarticles[idx1].main_body.angle
        temp1 = np.array(smarticles[idx1].main_body.position)
        for idx2 in range(len(smarticles)):
            if idx2 != idx1:
                sync1 += np.abs(math.sin(smarticles[idx2].main_body.angle - temp))
                temp1 += smarticles[idx2].main_body.angular_velocity
        sync.append(sync1)
    return sum(sync) / len(sync) if sync else 0.0


# =============================================================================
# Result I/O
# =============================================================================

def write_results_csv(csv_path: str, results: list):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    results_sorted = sorted(results, key=lambda d: int(d.get("trial_id", 0)))
    if not results_sorted:
        open(csv_path, "w", encoding="utf-8").close()
        return
    fieldnames = list(results_sorted[0].keys())
    seen = set(fieldnames)
    for r in results_sorted[1:]:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in results_sorted:
            w.writerow({k: r.get(k, "") for k in fieldnames})


def write_results_header(csv_path: str):
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
            ["trial_id", "seed", "jammed", "T_stop", "v_stop", "S_stop",
             "Align_score", "videopath"])


# =============================================================================
# Video recorder
# =============================================================================

class VideoRecorder:
    def __init__(self, out_path: str, width: int, height: int, fps: int,
                 downscale: int = None):
        self.writer  = None
        self._buf    = None          # reusable contiguous BGR frame buffer
        self._direct = None          # None = not probed yet
        self.ds      = max(1, int(VIDEO_DOWNSCALE if downscale is None else downscale))
        self.out_wh  = (width // self.ds, height // self.ds)
        if HAS_CV2:
            fourcc      = cv2.VideoWriter_fourcc(*(VIDEO_CODEC or "mp4v"))
            self.writer = cv2.VideoWriter(out_path, fourcc, fps, self.out_wh)

    @staticmethod
    def _can_use_raw_buffer(surface):
        """
        True when the surface's pixels are laid out in memory as B, G, R, X --
        i.e. exactly OpenCV's BGRA -- so the frame can be produced with one
        cv2.cvtColor call instead of a strided NumPy transpose.
        Pixel values are identical either way; this is purely about layout.
        """
        try:
            return (sys.byteorder == "little"
                    and surface.get_bitsize() == 32
                    and surface.get_shifts()[:3] == (16, 8, 0)
                    and surface.get_pitch() == surface.get_width() * 4)
        except Exception:
            return False

    def add_frame_from_surface(self, surface: pygame.Surface):
        """
        Push one frame.  Pixel content is identical to the previous
        implementation; only the number of intermediate copies changed
        (array3d copies the surface, then the reversed slice forced OpenCV to
        copy again — pixels3d is a zero-copy view and we fill one reusable
        contiguous buffer).
        """
        if self.writer is None:
            return
        if self._direct is None:
            self._direct = HAS_CV2 and self._can_use_raw_buffer(surface)
        try:
            if self._direct:
                # Fast path: the surface buffer is already BGRA, so a single
                # SIMD cvtColor produces the BGR frame (~17x faster than the
                # transpose below on a 2183x1843 surface, identical pixels).
                h, w  = surface.get_height(), surface.get_width()
                bgra  = np.frombuffer(surface.get_buffer(),
                                      dtype=np.uint8).reshape(h, w, 4)
                frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
            else:
                view = pygame.surfarray.pixels3d(surface)     # (w, h, 3) view
                if self._buf is None:
                    self._buf = np.empty((view.shape[1], view.shape[0], 3),
                                         dtype=np.uint8)
                np.copyto(self._buf, view.transpose(1, 0, 2)[:, :, ::-1])
                del view                                      # unlock surface
                frame = self._buf
            if self.ds != 1:
                # Crop to an exact multiple of ds first.  With a ratio that is
                # not exactly 1/ds (2183 -> 1091 is 2.0009) INTER_AREA falls
                # back to its slow general path: 21 ms/frame vs 1.4 ms for the
                # exact-integer path.  The few discarded edge pixels are
                # outside the arena (the ring is centred with a margin).
                frame = cv2.resize(frame[:self.out_wh[1] * self.ds,
                                         :self.out_wh[0] * self.ds],
                                   self.out_wh, interpolation=cv2.INTER_AREA)
            self.writer.write(frame)
        except Exception:
            # Any surface format pixels3d cannot map -> original slow path.
            arr = pygame.surfarray.array3d(surface).swapaxes(0, 1)
            self.writer.write(arr[:, :, ::-1])

    def close(self):
        if self.writer:
            self.writer.release()


# =============================================================================
# Pygame rendering helpers
# =============================================================================

def draw_timeseries(surface, rect, xs, ys,
                    color=(30, 30, 30), bg=(250, 250, 250), border=(0, 0, 0),
                    title="sync", y_margin_frac=0.10):
    """Draw a simple real-time line chart in a pygame surface rect."""
    if rect.width <= 10 or rect.height <= 10:
        return
    pygame.draw.rect(surface, bg, rect)
    pygame.draw.rect(surface, border, rect, 1)
    if len(xs) < 2:
        return

    x0, x1 = xs[0], xs[-1]
    if abs(x1 - x0) < 1e-9:
        return
    y_min, y_max = min(ys), max(ys)
    if abs(y_max - y_min) < 1e-9:
        y_max = y_min + 1e-9
    pad   = (y_max - y_min) * y_margin_frac
    y_min -= pad
    y_max += pad

    def mx(x):  return rect.left   + (x - x0) / (x1 - x0) * rect.width
    def my(y):  return rect.bottom - (y - y_min) / (y_max - y_min) * rect.height

    pygame.draw.lines(surface, color, False, [(mx(x), my(y)) for x, y in zip(xs, ys)], 2)
    try:
        avg = sum(ys) / len(ys)
        pygame.draw.line(surface, (200, 0, 0), (rect.left, my(avg)), (rect.right, my(avg)), 2)
    except Exception:
        pass

    font = pygame.font.SysFont("consolas", 14)
    avg_text = f" (avg={sum(ys)/len(ys):.3f})" if ys else ""
    surface.blit(font.render(f"{title}{avg_text}", True, (0, 0, 0)), (rect.left + 6, rect.top + 4))
    surface.blit(font.render(f"{y_max:.3f}",       True, (0, 0, 0)), (rect.left + 6, rect.top + 22))
    surface.blit(font.render(f"{y_min:.3f}",       True, (0, 0, 0)), (rect.left + 6, rect.bottom - 18))


def average_sync_last_seconds(time_hist, sync_hist, now, duration):
    if not time_hist:
        return None
    t0   = now - duration
    vals = [s for t, s in zip(time_hist, sync_hist) if t >= t0]
    return sum(vals) / len(vals) if vals else None

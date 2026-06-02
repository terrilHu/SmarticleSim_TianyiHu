"""
analysis.py  ─  Data analysis, order-parameter calculation, result I/O,
                video recording, and rendering helpers.
"""

import csv
import math
import os

import numpy as np
import pygame
from scipy.spatial import ConvexHull

from config import INNER_R, N_SMARTICLES, VIDEO_FPS, VIDEO_STRIDE

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
    arr = np.array(dPsi)
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
    dists       = dxy_matrix.flatten()
    hist, edges = np.histogram(dists, bins=bins, range=(0, r_max))
    r           = 0.5 * (edges[:-1] + edges[1:])
    gr          = hist / (np.sum(hist) + 1e-8)
    return r, gr


# =============================================================================
# Interaction / actuation metrics
# =============================================================================

def actuationimpactCalculation(freq1, freq2, Amp1, Amp2):
    from scipy.integrate import cumulative_trapezoid
    sample_time = max(1 / freq1, 1 / freq2)
    t      = np.arange(0, sample_time, 0.01)
    theta1 = Amp1 * np.sin(2 * math.pi * freq1 * t)
    theta2 = Amp2 * np.sin(2 * math.pi * freq2 * t)
    d1 = cumulative_trapezoid(np.abs(theta1 - theta2), t, initial=0)[-1]
    d2 = cumulative_trapezoid(np.abs(theta1 - theta1), t, initial=0)[-1]
    return [d1, d2]


def actutaionDetermination(dx, dy, dPsi, inter1, inter2):
    sign1 = np.sign(math.tan(dy / dx))
    sign2 = np.sign(dPsi)
    sign  = sign1 * sign2
    return 0.5 * max((1 - sign) * inter1, (1 + sign) * inter2)


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
    def __init__(self, out_path: str, width: int, height: int, fps: int):
        self.writer = None
        if HAS_CV2:
            fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    def add_frame_from_surface(self, surface: pygame.Surface):
        if self.writer is None:
            return
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

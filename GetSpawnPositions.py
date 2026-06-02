import json
import random
import numpy as np
import pymunk
import time
import sys
import math
import pygame
import pymunk.pygame_util
import os

from config import (INNER_R, WALL_THICK, N_SMARTICLES, TRIAL_SEED_BASE,
                    MAIN_LEN, MAIN_W, ARM_LEN, ARM_W, W, H,
                    INIT_PHASES, OMEGA_NOM1, OMEGA_NOM2, A_DEG_NOM1, A_DEG_NOM2)
from smarticle import Smarticle3Link, add_ring
from spawn import (spawn_smarticles, spawn_smarticles_norelax,
                   any_penetration, inside_ring, build_from_initial_conditions)
from naming import generate_trial_name

# =========================
# 参数
# =========================
N_TRIALS = 200

_EXP_NAME = generate_trial_name(
    N_SMARTICLES,
    INIT_PHASES,
    omega=(OMEGA_NOM1, OMEGA_NOM2),
    amplitude=(A_DEG_NOM1, A_DEG_NOM2),
)
SAVE_PATH = os.path.join("init_conditions", f"{_EXP_NAME}.json")
IMAGE_DIR = os.path.join("spawn_images",    _EXP_NAME)


# =========================
# 可视化与保存 debug (修改版)
# =========================
def save_layout_image(smarts, center, inner_r, filepath):
    """
    无弹窗离线渲染：将当前机器人的物理位置直接绘制在内存 Surface 上，并保存为 jpg
    """
    if not pygame.get_init():
        pygame.init()
        
    # 创建一个纯内存的 Surface，不需要调用 set_mode 产生弹窗
    surface = pygame.Surface((W, H))
    surface.fill((255, 255, 255))
    
    # 绘制外圈物理墙壁的示意图（浅灰色）
    pygame.draw.circle(surface, (220, 220, 220), (int(center.x), int(center.y)), int(inner_r + WALL_THICK), int(WALL_THICK))
    # 绘制内圈的安全边界（绿色实线）
    pygame.draw.circle(surface, (0, 200, 0), (int(center.x), int(center.y)), int(inner_r), 2)

    # 利用你原本的函数画所有的 smarticles
    for sm in smarts:
        draw_smarticle(surface, sm)
        
    # 保存为本地图片
    pygame.image.save(surface, filepath)


def draw_smarticle(surface, sm):
    def draw_poly(body, shape, color):
        verts = shape.get_vertices()
        pts = [body.local_to_world(v) for v in verts]
        pts = [(int(p.x), int(p.y)) for p in pts]
        pygame.draw.polygon(surface, color, pts)

    draw_poly(sm.main_body, sm.main_shape, (0, 100, 255))
    draw_poly(sm.left_body,  sm.left_shape,  (255, 100, 100))
    draw_poly(sm.right_body, sm.right_shape, (100, 255, 100))


# =========================
# 进度条
# =========================
def print_progress_bar(iteration, total, start_time, bar_length=30):
    percent = iteration / total
    filled_len = int(bar_length * percent)
    bar = "█" * filled_len + "-" * (bar_length - filled_len)
    elapsed = time.time() - start_time
    eta = (elapsed / iteration * (total - iteration)) if iteration > 0 else 0
    eta_str = time.strftime("%M:%S", time.gmtime(eta))
    sys.stdout.write(f"\r[{bar}] {percent*100:5.1f}% ({iteration}/{total}) ETA: {eta_str}")
    sys.stdout.flush()
    if iteration == total:
        print()


# =========================
# 提取状态
# =========================
def extract_smarticle_state(sm: Smarticle3Link):
    return {
        "pos":   [float(sm.main_body.position.x), float(sm.main_body.position.y)],
        "angle": float(sm.main_body.angle),
        "thL":   float(sm.left_body.angle  - sm.main_body.angle),
        "thR":   float(sm.right_body.angle - sm.main_body.angle),
    }


def relax_system(space, steps=300, dt=1/240.0):
    old_damping = space.damping
    space.damping = 0.85
    for _ in range(steps):
        space.step(dt)
    space.damping = old_damping


# =========================
# 主函数
# =========================
def generate_all_initial_conditions():
    all_trials = []
    start_time = time.time()

    # 创建独立的图片保存文件夹
    os.makedirs(IMAGE_DIR, exist_ok=True)

    for trial_id in range(N_TRIALS):
        seed = TRIAL_SEED_BASE + trial_id
        random.seed(seed)
        np.random.seed(seed)

        # ── 建 space ──────────────────────────────────────
        space = pymunk.Space()
        center = pymunk.Vec2d(W / 2, H / 2)
        add_ring(space, center, INNER_R, WALL_THICK)

        # ── 调用新 spawn ──────────────────────────────────
        smarts = spawn_smarticles(space, center, INNER_R, N_SMARTICLES)

        # ==============================================================
        # 新增：无论当前生成成功还是失败，都立马把它们的样子拍下来保存
        # ==============================================================
        # 根据要求生成文件名：trail_0001_seed_#####.jpg (trial_id 从 0 开始，所以 +1)
        img_filename = f"trial_{trial_id:04d}.jpg"
        img_path = os.path.join(IMAGE_DIR, img_filename)
        save_layout_image(smarts, center, INNER_R, img_path)

        # ── 放不满：记录日志并跳过 ────────────────────────────────
        if len(smarts) != N_SMARTICLES:
            print(f"\n[DEBUG] Trial {trial_id + 1} failed: only placed {len(smarts)}/{N_SMARTICLES}. Image saved.")
            # 移除已有的 smarticles 释放物理空间内存，跳过这次进入下一个 seed
            for sm in smarts:
                sm.remove_from_space()
            continue

        # ── Relax 后验证 (如果你以后需要可以取消注释) ──────────────
        # relax_system(space, steps=300)
        # valid = all(
        #    not any_penetration(space, sm) and inside_ring(sm, center, INNER_R)
        #    for sm in smarts
        # )
        # if not valid:
        #    print(f"\n[DEBUG] Trial {trial_id + 1}: relax 后验证失败，跳过")
        #    for sm in smarts:
        #        sm.remove_from_space()
        #    continue

        # ── 保存成功的 Trial 到 JSON ────────────────────────────────
        trial_data = {
            "trial_id": trial_id,
            "seed":     seed,
            "smarticles": [extract_smarticle_state(sm) for sm in smarts],
        }
        for sm in smarts:
            sm.remove_from_space()

        all_trials.append(trial_data)
        print_progress_bar(trial_id + 1, N_TRIALS, start_time)

    # 写入最终合集的 JSON 文件
    with open(SAVE_PATH, "w") as f:
        json.dump(all_trials, f, indent=2)
    print(f"\n[spawn] Experiment: {_EXP_NAME}")
    print(f"[spawn] Saved {len(all_trials)}/{N_TRIALS} initial conditions → {SAVE_PATH}")
    print(f"[spawn] Spawn images → {IMAGE_DIR}")


if __name__ == "__main__":
    area_circle = math.pi * INNER_R * INNER_R
    area_sm = MAIN_LEN * MAIN_W + 2 * ARM_LEN * ARM_W
    print("Packing ratio =", N_SMARTICLES * area_sm / area_circle)
    generate_all_initial_conditions()
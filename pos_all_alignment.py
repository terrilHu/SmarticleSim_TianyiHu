"""
pos_all_alignment.py
---------------------
处理仿真输出的 _POS_ALL.csv，对每帧做 Voronoi 邻接分析，
计算每个机器人与其邻居的 alignment，最终输出全局 alignment 时间序列。

用法:
    python pos_all_alignment.py --input trial_0000_seed_0_POS_ALL.csv \
                                 --max_dist 60.0 \
                                 --output alignment_output.csv
"""

import argparse
import math
import numpy as np
import pandas as pd
from scipy.spatial import Delaunay


# =============================================================================
# Voronoi 邻接矩阵（通过 Delaunay 三角剖分，与 MATLAB 版本等价）
# =============================================================================

def voronoi_adjacency(positions: np.ndarray, max_dist: float):
    """
    输入:
        positions  - (N, 2) 当前帧所有机器人的 (x, y) 坐标，可含 NaN
        max_dist   - 距离阈值，超过则邻接矩阵置 False

    输出:
        adj  - (N, N) bool 对称邻接矩阵
    """
    N = len(positions)
    adj = np.zeros((N, N), dtype=bool)

    # 过滤 NaN
    valid_mask = np.isfinite(positions).all(axis=1)
    valid_idx  = np.where(valid_mask)[0]
    pts        = positions[valid_idx]   # (M, 2)

    if len(pts) < 2:
        return adj

    try:
        tri = Delaunay(pts)
    except Exception:
        return adj

    # 从三角形提取无重复边
    simplices = tri.simplices          # (T, 3)
    edge_set  = set()
    for s in simplices:
        for a, b in [(s[0],s[1]), (s[1],s[2]), (s[0],s[2])]:
            edge_set.add((min(a,b), max(a,b)))

    for li, lj in edge_set:
        i = valid_idx[li]
        j = valid_idx[lj]
        d = np.linalg.norm(pts[li] - pts[lj])
        if d <= max_dist:
            adj[i, j] = True
            adj[j, i] = True

    return adj


# =============================================================================
# 单个 alignment 计算（与 MATLAB singleAlignment 等价）
# 输入角度单位：弧度（theta 已是弧度，与 MATLAB 版接收角度不同，注意区分）
# =============================================================================

def single_alignment(angles_rad: np.ndarray) -> float:
    """
    计算一组朝向角的 alignment（向量序参量）。
    对应 MATLAB singleAlignment，但输入已是弧度。

    alignment = |mean(exp(2i * theta))| ∈ [0, 1]
    1 表示完全对齐，0 表示完全随机。
    """
    if len(angles_rad) == 0:
        return 0.0
    V = np.sum(np.exp(2j * angles_rad))
    return abs(V) / len(angles_rad)


# =============================================================================
# 主处理函数
# =============================================================================

def process_pos_all(csv_path: str, max_dist: float) -> pd.DataFrame:
    """
    读取 _POS_ALL.csv，逐帧计算全局 alignment。

    CSV 格式（修复 bug 后）:
        Step, Agent_ID, X, Y, Theta
        0, 1, 312.14, 256.78, 1.2341
        ...

    返回:
        DataFrame，列: [Step, mean_alignment]
    """
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    steps      = sorted(df["Step"].unique())
    n_agents   = df["Agent_ID"].nunique()
    results    = []

    for step in steps:
        frame = df[df["Step"] == step].sort_values("Agent_ID")

        positions = frame[["X", "Y"]].to_numpy(dtype=float)   # (N, 2)
        thetas    = frame["Theta"].to_numpy(dtype=float)       # (N,)  弧度

        # 1. Voronoi 邻接矩阵
        adj = voronoi_adjacency(positions, max_dist)
        # if step == 1:
        #     print(adj)

        # 2. 每个机器人与自身 + 邻居的 alignment
        agent_alignments = []
        for i in range(n_agents):
            neighbor_idx = np.where(adj[i])[0]
            angles = np.concatenate([[thetas[i]], thetas[neighbor_idx]])
            a = single_alignment(angles)
            agent_alignments.append(a)

        # 3. 所有机器人平均
        mean_align = float(np.mean(agent_alignments))
        results.append({"Step": step, "mean_alignment": mean_align})

    return pd.DataFrame(results)


# =============================================================================
# 入口
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="计算 POS_ALL 数据的 alignment 时间序列")
    parser.add_argument("--input",    required=True,        help="_POS_ALL.csv 路径")
    parser.add_argument("--max_dist", type=float, default=60.0, help="Voronoi 邻接距离阈值")
    parser.add_argument("--output",   default="alignment_timeseries.csv", help="输出 CSV 路径")
    args = parser.parse_args()

    print(f"读取: {args.input}")
    result_df = process_pos_all(args.input, args.max_dist)

    result_df.to_csv(args.output, index=False)
    print(f"完成，共 {len(result_df)} 帧，已保存至: {args.output}")
    print(result_df.describe())


if __name__ == "__main__":
    main()

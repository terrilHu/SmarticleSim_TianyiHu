"""
batch_group_plots.py
--------------------
批量画"分组随时间演化"的图，并对每组实验做 trial 之间的平均统计。

输入是**一串实验目录**，每个实验目录下面是一系列 trial 文件夹：

    datafile/exp_A/                 <- 命令行给这一层
        config_snapshot.json
        trial_0000/trial_0000_POS_ALL.csv
        trial_0001/trial_0001_POS_ALL.csv
        ...
    datafile/exp_B/
        ...

用法:
    python batch_group_plots.py datafile/exp_A datafile/exp_B
    python batch_group_plots.py "datafile/*"                  # 通配符
    python batch_group_plots.py --dirs-from list.txt          # 每行一个实验目录
    python batch_group_plots.py datafile/exp_A --figs composition mean_groupsize

输出分两层。**每个 trial** 一套（默认放回该 trial 文件夹）：

    <trial>_composition.png   各规模档位占了多少机器人？(堆叠面积)
    <trial>_groupsize.png     最大组、"随手抓一台它在多大的组里"怎么变？
    <trial>_kymograph.png     谁和谁并在一起，什么时候？
    <trial>_counts.png        组的数量、最大团占比
    <trial>_snapshots.png     空间上长什么样？

**每组实验**一套（放在实验目录下），把该实验所有 trial 平均起来：

    <exp>_mean_composition.png   平均后的堆叠面积
    <exp>_mean_groupsize.png     细线是各 trial，粗线是均值，带是 ±1 std
    <exp>_mean_counts.png        组数、最大团占比的均值与离散度
    <exp>_trial_stats.csv        每个 trial 一行的标量统计
    <exp>_mean_timeseries.csv    对齐到公共时间轴后各量的 mean/std

时间轴的 t=0 是**第一个被记录的帧**，不是仿真的 t=0：RECORD_AFTER_WARMUP
会跳过 WARMUP_STEPS 那段暖机，所以 12 秒的 trial 落盘的是约 11 秒数据。
这样各 trial 的 t=0 对应同一个物理时刻（暖机刚结束），跨 trial 平均才对得齐。

跨 trial 平均前会插值到公共时间轴（取最短那条 trial 的时长；外推没有意义，
短的那条之后本就没有数据），时长不一致会在日志里说明截断到了多长。

分组结果缓存成 trial_XXXX_groups_d<max_dist>.csv（列与 pos_all_grouping 的
process_pos_all_groups 完全一致），下次重画直接读缓存，--force 强制重算。
不同 max_dist 的缓存互不覆盖，文件名里带着阈值。

堆叠面积图画的是**滑动平均后**的占比（默认 2 秒，--smooth-s 可改，给 0 关掉）。
所以一条带可能比它自己的档位下限还薄：某帧存在一个 16 台的组，瞬时占比就是
16%，但若这个组在 2 秒窗口里只存在 30% 的时间，画出来就是 0.3*16% ≈ 5%。
换句话说带的厚度是"这段时间里有多少机器人-时间落在该档位"，不是某一瞬间的占比。
想看瞬时值就 --smooth-s 0。

绘图沿用实机侧 plot_group_evolution.py 的配色与平滑方式；数据入口不同：
实机读带 Time 列的 group<stamp>.csv，这里时间由 Step / fps 换算，fps 优先从
实验目录的 config_snapshot.json 读 RENDER_FPS_HEADLESS。

分组用的是 pos_all_grouping.compute_frame_groups —— 和 strategy.py 实时决策
调的是同一个函数，已逐帧核对过两者划分完全一致（前提是 max_dist 相同）。
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pos_all_grouping import compute_frame_groups

# 图上标注一律用英文：投稿和汇报都用得上，也省掉了跨机器的中文字体依赖
# (Windows 有 SimHei、Linux 常常什么都没有，同一份脚本在两边出图会不一样)。
plt.rcParams["axes.unicode_minus"] = False

FIG_TRIAL = ("composition", "groupsize", "kymograph", "counts", "snapshots")
FIG_EXP = ("mean_composition", "mean_groupsize", "mean_counts")
FIG_ALIASES = {
    "all": FIG_TRIAL + FIG_EXP,
    "trial": FIG_TRIAL,
    "exp": FIG_EXP,
    "aggregation": ("composition", "groupsize"),   # 原来那张图拆开后的两半
}

SIZE_COLORS = ["#d9d9d9", "#9ecae1", "#4292c6", "#2171b5", "#08306b"]
C_LARGEST, C_MEANSZ, C_ALIGN = "#08306b", "#e6550d", "#31a354"

EPILOG = """\
位置参数是实验目录，里面装着 trial_XXXX/ 子文件夹。
--figs 可选:
    每个 trial : composition groupsize kymograph counts snapshots
    每组实验   : mean_composition mean_groupsize mean_counts
    别名       : all / trial / exp / aggregation(=composition+groupsize)
"""


def size_bins(n):
    """
    分档随种群规模走。实机那份写死 9+ 封顶，对 17 台正合适；
    到 100 台时"9 台以上"会把绝大多数机器人塞进同一档，图就没信息了。
    """
    if n <= 20:
        edges = [1, 3, 5, 8]
    elif n <= 60:
        edges = [1, 3, 6, 12]
    else:
        edges = [1, 5, 10, 15]
    bins, labels, lo = [], [], 1
    for e in edges:
        bins.append((lo, e))
        labels.append(str(lo) if lo == e else f"{lo}-{e}")
        lo = e + 1
    bins.append((lo, max(lo, n)))
    labels.append(f"{lo}+")
    return bins, labels


# =============================================================================
# 读取：POS_ALL -> 每帧分组
# =============================================================================

def find_pos_all(folder):
    hits = sorted(glob.glob(os.path.join(folder, "*_POS_ALL.csv")))
    return hits[0] if hits else None


def find_trials(exp_dir):
    """
    实验目录下所有含 *_POS_ALL.csv 的直接子目录，按名字排序。

    如果实验目录自己就直接放着 POS_ALL（有人图省事把单个 trial 目录传进来），
    就把它当成"只有一个 trial 的实验"，免得静默地什么都不做。
    """
    out = [sub for sub in sorted(glob.glob(os.path.join(exp_dir, "*")))
           if os.path.isdir(sub) and find_pos_all(sub)]
    if not out and find_pos_all(exp_dir):
        out = [exp_dir]
    return out


def detect_fps(folder, fallback=60.0):
    """
    每秒记录多少帧 POS_ALL。优先用该实验自己的 config_snapshot.json
    （run_trial 把它写在实验目录里），退回到当前 config.py。
    """
    here = os.path.abspath(folder)
    for d in (here, os.path.dirname(here)):
        path = os.path.join(d, "config_snapshot.json")
        if os.path.isfile(path):
            try:
                with open(path, encoding="utf-8") as f:
                    v = json.load(f).get("RENDER_FPS_HEADLESS")
                if v:
                    return float(v), "config_snapshot.json"
            except Exception:
                pass
    try:
        from config import RENDER_FPS_HEADLESS
        return float(RENDER_FPS_HEADLESS), "config.py"
    except Exception:
        return float(fallback), "fallback"


def group_table(pos_all_csv, max_dist, include_singletons=True, force=False):
    """
    每帧每组一行的表，列与 pos_all_grouping.process_pos_all_groups 相同。

    结果缓存在 <prefix>_groups_d<max_dist>.csv 旁边。计算本身调的就是
    compute_frame_groups，只是把 process_pos_all_groups 里
    "每帧重扫一遍整张表"(O(帧数^2)) 换成排序后切片 —— 分组逻辑一字未改。
    """
    tag = f"{max_dist:g}".replace(".", "p")
    cache = pos_all_csv.replace("_POS_ALL.csv", f"_groups_d{tag}.csv")
    if not include_singletons:
        cache = cache.replace(".csv", "_nosing.csv")
    if os.path.isfile(cache) and not force:
        return pd.read_csv(cache), cache, True

    df = pd.read_csv(pos_all_csv)
    df.columns = df.columns.str.strip()
    df = df.sort_values(["Step", "Agent_ID"], kind="mergesort")

    step_col = df["Step"].to_numpy()
    xy = df[["X", "Y"]].to_numpy(dtype=float)
    th = df["Theta"].to_numpy(dtype=float)
    aid = df["Agent_ID"].to_numpy()

    steps, starts = np.unique(step_col, return_index=True)
    bounds = np.append(starts, len(step_col))

    rows = []
    for k, step in enumerate(steps):
        sl = slice(bounds[k], bounds[k + 1])
        for gi, g in enumerate(compute_frame_groups(
                xy[sl], th[sl], aid[sl], max_dist, include_singletons)):
            rows.append({
                "Step": int(step),
                "group_local_id": gi,
                "size": g["size"],
                "alignment": g["alignment"],
                "centroid_x": g["centroid"][0],
                "centroid_y": g["centroid"][1],
                "member_ids": " ".join(str(int(a)) for a in g["agent_ids"]),
            })
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    return out, cache, False


def load_frames(groups_df, fps):
    """-> [(step, seconds, [ {members, size, alignment, centroid}, ... ]), ...]"""
    frames = []
    for step, g in groups_df.groupby("Step", sort=True):
        frames.append((int(step), float(step) / fps, [{
            "members": frozenset(int(x) for x in str(r["member_ids"]).split()),
            "size": int(r["size"]),
            "alignment": float(r["alignment"]),
            "centroid": (float(r["centroid_x"]), float(r["centroid_y"])),
        } for _, r in g.iterrows()]))
    return frames


def track_groups(frames, min_size=2, min_jaccard=0.3):
    """
    给每帧的分组配一个跨帧稳定的编号：按成员集合的 Jaccard 重叠度做贪心
    一对一匹配。和 pos_all_grouping.track_groups 同样的思路，这里独立实现是
    因为画图想追踪所有 size>=2 的组，而那份是围绕 DataFrame 组织的。

    -> [ {gid: members}, ... ] 每帧一个 dict
    """
    prev = {}
    next_gid = 1
    out = []
    for _step, _sec, groups in frames:
        cur = [g["members"] for g in groups if g["size"] >= min_size]
        pairs = []
        for gid, pm in prev.items():
            for ci, m in enumerate(cur):
                inter = len(pm & m)
                if inter:
                    j = inter / len(pm | m)
                    if j >= min_jaccard:
                        pairs.append((-j, -inter, gid, ci))
        pairs.sort()
        used_g, used_c, now = set(), set(), {}
        for _j, _i, gid, ci in pairs:
            if gid in used_g or ci in used_c:
                continue
            used_g.add(gid); used_c.add(ci)
            now[gid] = cur[ci]
        for ci, m in enumerate(cur):
            if ci not in used_c:
                now[next_gid] = m
                next_gid += 1
        out.append(now)
        prev = now
    return out


# =============================================================================
# 每个 trial 的时间序列（画图和跨 trial 平均都吃这个）
# =============================================================================

def trial_series(frames, bins):
    """把逐帧的分组结果压成几条等长的时间序列。"""
    m = len(frames)
    secs = np.array([f[1] for f in frames], dtype=float)
    frac = np.zeros((m, len(bins)))
    largest = np.empty(m); mean_sz = np.empty(m); align = np.empty(m)
    n_all = np.empty(m); n_multi = np.empty(m); n_seen = np.empty(m)

    for i, (_s, _t, groups) in enumerate(frames):
        sizes = [g["size"] for g in groups]
        total = sum(sizes)
        for g in groups:
            for b, (lo, hi) in enumerate(bins):
                if lo <= g["size"] <= hi:
                    frac[i, b] += g["size"] / max(total, 1)
                    break
        largest[i] = max(sizes)
        # 机器人视角的平均组规模: 随便抓一台机器人, 它所在的组多大
        mean_sz[i] = sum(s * s for s in sizes) / max(total, 1)
        align[i] = sum(g["size"] * g["alignment"] for g in groups) / max(total, 1)
        n_all[i] = len(groups)
        n_multi[i] = sum(1 for s in sizes if s >= 2)
        n_seen[i] = total

    return {"secs": secs, "frac": frac, "largest": largest, "mean_size": mean_sz,
            "alignment": align, "n_groups": n_all, "n_multi": n_multi,
            "n_seen": n_seen, "largest_frac": largest / np.maximum(n_seen, 1)}


def smooth_window(secs, seconds):
    """把"多少秒"换算成滑动窗口的帧数；<=0 表示不平滑。"""
    if seconds is None or seconds <= 0:
        return 1
    span = max(float(secs[-1]) - float(secs[0]), 1e-9)
    return max(3, int(round(seconds * len(secs) / span)))


def roll_mean(v, win):
    if win <= 1:
        return np.asarray(v, dtype=float)
    return pd.Series(v).rolling(win, center=True, min_periods=1).mean().to_numpy()


def roll_median(v, win):
    if win <= 1:
        return np.asarray(v, dtype=float)
    return pd.Series(v).rolling(win, center=True, min_periods=1).median().to_numpy()


# =============================================================================
# 图 1a: 组成（各规模档位占多少机器人）
# =============================================================================

def fig_composition(sr, labels, path, title="", smooth_s=2.0, note=None):
    secs = sr["secs"]
    # 逐帧占比抖动很大(单帧的一条 Voronoi 边就能让两个组合并又分开)，
    # 直接画会被噪声淹没。用约 2 秒的滑动平均，趋势才看得出来。
    win = smooth_window(secs, smooth_s)
    frac_s = np.column_stack([roll_mean(sr["frac"][:, b], win)
                              for b in range(sr["frac"].shape[1])])

    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.stackplot(secs, frac_s.T * 100, labels=labels,
                 colors=SIZE_COLORS[:len(labels)], edgecolor="none")
    ax.set_ylim(0, 100)
    ax.set_xlim(secs[0], secs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("share of robots (%)")
    ax.set_title(f"Group size composition{title}")
    ax.legend(title="group size", loc="upper center", bbox_to_anchor=(0.5, -0.16),
              ncol=len(labels), frameon=False, fontsize=9)
    if note is None:
        note = (f"({smooth_s:g}s moving average — a band can read below its own "
                f"size when it is occupied only part of the window)"
                if win > 1 else "(per-frame share, no smoothing)")
    ax.text(0.995, 0.03, note, transform=ax.transAxes, ha="right",
            fontsize=8, color="#555")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 图 1b: 组规模
# =============================================================================

def fig_groupsize(sr, path, title="", smooth_s=2.0):
    secs = sr["secs"]
    win = smooth_window(secs, smooth_s)

    fig, ax = plt.subplots(figsize=(11, 4.4))
    ax.plot(secs, sr["largest"], lw=0.8, color=C_LARGEST, alpha=0.30)
    ax.plot(secs, roll_mean(sr["largest"], win), lw=2.2, color=C_LARGEST,
            label="largest group")
    ax.plot(secs, roll_mean(sr["mean_size"], win), lw=2.2, color=C_MEANSZ,
            label="mean group size (per robot)")
    ax.set_xlim(secs[0], secs[-1])
    ax.set_xlabel("time (s)")
    ax.set_ylabel("group size")
    ax.set_title(f"Group size over time{title}")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(alpha=0.25)

    ax2 = ax.twinx()
    ax2.plot(secs, roll_mean(sr["alignment"], win), lw=1.2, color=C_ALIGN, ls="--")
    ax2.set_ylim(0, 1.05)
    ax2.set_ylabel("size-weighted alignment", color=C_ALIGN)
    ax2.tick_params(axis="y", colors=C_ALIGN)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 图 2: kymograph (时间 x 机器人)
# =============================================================================

def fig_kymograph(frames, tracked, path, min_life=6, title=""):
    ids = sorted({m for _s, _t, gs in frames for g in gs for m in g["members"]})
    row = {m: i for i, m in enumerate(ids)}
    secs = [f[1] for f in frames]

    # 每台机器人每帧所属的那个组编号，落单为 -1
    M = np.full((len(ids), len(frames)), -1, dtype=int)
    for j, now in enumerate(tracked):
        for gid, members in now.items():
            for m in members:
                if m in row:
                    M[row[m], j] = gid

    # 只给"活得够久"的组分颜色。存在一两帧就散掉的组统统一个中性淡紫，
    # 否则上百个瞬生瞬灭的组各占一个色号，图会变成彩色噪声。
    life = {g: int((M == g).any(axis=0).sum()) for g in np.unique(M) if g >= 0}
    stable = sorted([g for g, n in life.items() if n >= min_life])
    cmap = plt.get_cmap("tab20")
    color = {g: cmap(i % 20) for i, g in enumerate(stable)}
    rgb = np.ones(M.shape + (3,))
    rgb[M >= 0] = (0.72, 0.72, 0.78)         # 短暂成组
    for g, c in color.items():
        rgb[M == g] = c[:3]
    rgb[M < 0] = (0.95, 0.95, 0.95)          # 落单

    # 机器人多的时候逐行标号会糊成一片，改成稀疏刻度
    step = max(1, len(ids) // 40)
    fig, ax = plt.subplots(figsize=(12, min(24, 0.32 * len(ids) + 2)))
    ax.imshow(rgb, aspect="auto", interpolation="nearest", origin="lower",
              extent=[secs[0], secs[-1], -0.5, len(ids) - 0.5])
    ax.set_yticks(range(0, len(ids), step))
    ax.set_yticklabels([ids[i] for i in range(0, len(ids), step)], fontsize=8)
    ax.set_ylabel("robot ID")
    ax.set_xlabel("time (s)")
    # 图例说明放到第二行：机器人多的时候一行标题会被画布切掉右半截
    ax.set_title(f"Group membership per robot{title}\n"
                 f"colour = groups lasting >{min_life} frames   |   "
                 f"light purple = transient   |   grey = alone",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 图 3: 数量
# =============================================================================

def fig_counts(sr, path, title="", smooth_s=2.0):
    secs = sr["secs"]
    w = smooth_window(secs, smooth_s)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    a1.plot(secs, sr["n_groups"], lw=0.7, alpha=0.3, color="#666")
    a1.plot(secs, roll_median(sr["n_groups"], w), lw=2, color="#666",
            label="all groups")
    a1.plot(secs, roll_median(sr["n_multi"], w), lw=2, color="#2171b5",
            label="groups with >=2")
    a1.plot(secs, roll_median(sr["n_seen"], w), lw=1.2, ls=":", color="#999",
            label="robots detected")
    a1.set_ylabel("count")
    a1.set_title(f"Group counts{title}", fontsize=11)
    a1.legend(frameon=False, ncol=3, fontsize=9)
    a1.grid(alpha=0.25)

    a2.plot(secs, sr["largest_frac"], lw=0.7, alpha=0.3, color=C_LARGEST)
    a2.plot(secs, roll_median(sr["largest_frac"], w), lw=2.2, color=C_LARGEST)
    a2.set_ylim(0, 1)
    a2.set_xlim(secs[0], secs[-1])
    a2.set_ylabel("largest group / all")
    a2.set_xlabel("time (s)")
    a2.grid(alpha=0.25)
    a2.set_title("Largest cluster fraction", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 图 4: 空间快照
# =============================================================================

def fig_snapshots(frames, tracked, path, n=5, title=""):
    idx = np.linspace(0, len(frames) - 1, n).astype(int)
    cmap = plt.get_cmap("tab20")
    xs = [g["centroid"][0] for _s, _t, gs in frames for g in gs]
    ys = [g["centroid"][1] for _s, _t, gs in frames for g in gs]
    pad = 0.05 * max(max(xs) - min(xs), max(ys) - min(ys), 1.0)

    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4))
    for ax, i in zip(np.atleast_1d(axes), idx):
        _s, sec, groups = frames[i]
        for g in groups:
            gid = next((k for k, v in tracked[i].items() if v == g["members"]), None)
            c = cmap(gid % 20) if gid is not None else (0.8, 0.8, 0.8)
            ax.scatter(*g["centroid"], s=25 + 45 * g["size"], color=c,
                       edgecolor="k", linewidth=0.4, alpha=0.85)
            if g["size"] >= 2:
                ax.annotate(str(g["size"]), g["centroid"], fontsize=7,
                            ha="center", va="center")
        ax.set_title(f"t = {sec:.0f}s", fontsize=10)
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)   # 仿真世界坐标 y 向下，与视频一致
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle(f"Spatial snapshots{title}\n"
                 "marker size ~ group size, same colour = same group",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 跨 trial 平均
# =============================================================================

def align_series(all_sr, n_points=None):
    """
    把若干条 trial 的序列插到公共时间轴上。

    时间轴取 0 .. min(各 trial 时长)：外推没有意义，最短那条之后并没有数据，
    硬补会凭空造出一段"所有 trial 都还在"的假象。
    """
    t_end = min(float(sr["secs"][-1]) for sr in all_sr)
    n_points = n_points or max(len(sr["secs"]) for sr in all_sr)
    grid = np.linspace(0.0, t_end, int(n_points))

    keys = ("largest", "mean_size", "alignment", "n_groups", "n_multi",
            "n_seen", "largest_frac")
    stacked = {k: np.stack([np.interp(grid, sr["secs"], sr[k]) for sr in all_sr])
               for k in keys}
    nb = all_sr[0]["frac"].shape[1]
    stacked["frac"] = np.stack([                              # (trials, T, bins)
        np.stack([np.interp(grid, sr["secs"], sr["frac"][:, b])
                  for b in range(nb)], axis=1)
        for sr in all_sr])
    return grid, stacked, t_end


def fig_mean_composition(grid, stacked, labels, path, title="", smooth_s=2.0):
    n = stacked["frac"].shape[0]
    note = (f"(mean of {n} trials, {smooth_s:g}s moving average — a band can "
            f"read below its own size)" if smooth_s and smooth_s > 0
            else f"(mean of {n} trials, no smoothing)")
    fig_composition({"secs": grid, "frac": stacked["frac"].mean(axis=0)},
                    labels, path, title, smooth_s, note=note)


def _band(ax, grid, arr, color, label, win, show_trials=True):
    """细线 = 各 trial，粗线 = 均值，带 = ±1 std。"""
    if show_trials:
        for r in arr:
            ax.plot(grid, roll_mean(r, win), lw=0.7, color=color, alpha=0.25)
    m = roll_mean(arr.mean(axis=0), win)
    s = roll_mean(arr.std(axis=0), win)
    ax.fill_between(grid, m - s, m + s, color=color, alpha=0.18, linewidth=0)
    ax.plot(grid, m, lw=2.4, color=color, label=label)


def fig_mean_groupsize(grid, stacked, path, title="", smooth_s=2.0):
    win = smooth_window(grid, smooth_s)
    n = stacked["largest"].shape[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)

    _band(a1, grid, stacked["largest"], C_LARGEST, "largest group", win)
    a1.set_ylabel("largest group size")
    a1.set_title(f"Group size across trials{title}\n"
                 f"thin = individual trials (n={n}), bold = mean, band = ±1 std",
                 fontsize=11)
    a1.legend(frameon=False, loc="upper left")
    a1.grid(alpha=0.25)

    _band(a2, grid, stacked["mean_size"], C_MEANSZ,
          "mean group size (per robot)", win)
    a2.set_ylabel("group size")
    a2.set_xlabel("time (s)")
    a2.set_xlim(grid[0], grid[-1])
    a2.legend(frameon=False, loc="upper left")
    a2.grid(alpha=0.25)

    ax = a2.twinx()
    ax.plot(grid, roll_mean(stacked["alignment"].mean(axis=0), win),
            lw=1.2, color=C_ALIGN, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("size-weighted alignment", color=C_ALIGN)
    ax.tick_params(axis="y", colors=C_ALIGN)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_mean_counts(grid, stacked, path, title="", smooth_s=2.0):
    win = smooth_window(grid, smooth_s)
    n = stacked["n_groups"].shape[0]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 6.4), sharex=True)

    _band(a1, grid, stacked["n_groups"], "#666", "all groups", win)
    _band(a1, grid, stacked["n_multi"], "#2171b5", "groups with >=2", win,
          show_trials=False)
    a1.set_ylabel("count")
    a1.set_title(f"Group counts across trials{title}\n"
                 f"thin = individual trials (n={n}), bold = mean, band = ±1 std",
                 fontsize=11)
    a1.legend(frameon=False, ncol=2, fontsize=9)
    a1.grid(alpha=0.25)

    _band(a2, grid, stacked["largest_frac"], C_LARGEST, "largest / all", win)
    a2.set_ylim(0, 1)
    a2.set_xlim(grid[0], grid[-1])
    a2.set_ylabel("largest group / all")
    a2.set_xlabel("time (s)")
    a2.legend(frameon=False, loc="upper left")
    a2.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# =============================================================================
# 处理一个 trial / 一组实验
# =============================================================================

def process_trial(folder, exp_name, bins, labels, args, fps):
    pos_all = find_pos_all(folder)
    name = os.path.basename(os.path.normpath(folder))

    gdf, cache, cached = group_table(pos_all, args.max_dist,
                                     not args.no_singletons, args.force)
    if gdf.empty:
        print(f"    [skip] {name}: 分组表为空")
        return None

    frames = load_frames(gdf, fps)
    sr = trial_series(frames, bins)

    out_dir = args.out or folder
    os.makedirs(out_dir, exist_ok=True)
    prefix = f"{exp_name}_{name}" if args.out else name
    title = f"  [{exp_name}/{name}, max_dist={args.max_dist:g}]" if args.title else ""

    made = []

    def p(fig_name):
        fn = f"{prefix}_{fig_name}.png"
        made.append(fn)
        return os.path.join(out_dir, fn)

    if "composition" in args.figs:
        fig_composition(sr, labels, p("composition"), title, args.smooth_s)
    if "groupsize" in args.figs:
        fig_groupsize(sr, p("groupsize"), title, args.smooth_s)
    if "counts" in args.figs:
        fig_counts(sr, p("counts"), title, args.smooth_s)
    # 跨帧追踪只有这两张图要用，别的组合就不必付这份代价
    if {"kymograph", "snapshots"} & args.figs:
        tracked = track_groups(frames, min_size=args.min_size)
        if "kymograph" in args.figs:
            fig_kymograph(frames, tracked, p("kymograph"),
                          min_life=max(3, int(round(args.min_life_s * fps))),
                          title=title)
        if "snapshots" in args.figs:
            fig_snapshots(frames, tracked, p("snapshots"), args.snapshots, title)

    tail = sr["secs"] >= sr["secs"][-1] - args.steady_s
    stats = {
        "exp": exp_name, "trial": name,
        "n_robots": int(sr["n_seen"].max()),
        "frames": len(frames), "duration_s": float(sr["secs"][-1]),
        "largest_mean": float(sr["largest"].mean()),
        "largest_steady": float(sr["largest"][tail].mean()),
        "largest_peak": int(sr["largest"].max()),
        "mean_size_steady": float(sr["mean_size"][tail].mean()),
        "n_groups_steady": float(sr["n_groups"][tail].mean()),
        "largest_frac_steady": float(sr["largest_frac"][tail].mean()),
        "alignment_steady": float(sr["alignment"][tail].mean()),
    }
    print(f"    {name}: {len(frames)} 帧 / {stats['duration_s']:.1f}s, "
          f"最大组 均值 {stats['largest_mean']:.1f} / 末{args.steady_s:g}s "
          f"{stats['largest_steady']:.1f} / 峰值 {stats['largest_peak']}"
          f"   [{'缓存' if cached else '新算'}]")
    for fn in made:
        print(f"      -> {fn}")
    return sr, stats


STEADY_COLS = ("largest_mean", "largest_steady", "largest_peak",
               "mean_size_steady", "n_groups_steady",
               "largest_frac_steady", "alignment_steady")


def process_experiment(exp_dir, args):
    exp_name = os.path.basename(os.path.normpath(exp_dir))
    trials = find_trials(exp_dir)
    if not trials:
        print(f"  [skip] 没有含 *_POS_ALL.csv 的 trial 子目录")
        return None

    fps, fps_src = (args.fps, "--fps") if args.fps else detect_fps(exp_dir)
    print(f"  {len(trials)} 个 trial, fps={fps:g} ({fps_src})")

    # 档位边界要在整组实验内统一，否则各 trial 的堆叠面积图不可比、
    # 也没法把它们平均起来。先扫一眼各 trial 的机器人数，取最大的定档。
    n_max = 0
    for t in trials:
        n_max = max(n_max, int(pd.read_csv(find_pos_all(t),
                                           usecols=["Agent_ID"])
                               ["Agent_ID"].nunique()))
    bins, labels = size_bins(n_max)

    all_sr, all_stats = [], []
    for t in trials:
        r = process_trial(t, exp_name, bins, labels, args, fps)
        if r:
            all_sr.append(r[0])
            all_stats.append(r[1])
    if not all_sr:
        return None

    out_dir = args.out or exp_dir
    os.makedirs(out_dir, exist_ok=True)
    stats_df = pd.DataFrame(all_stats)
    stats_path = os.path.join(out_dir, f"{exp_name}_trial_stats.csv")
    stats_df.to_csv(stats_path, index=False)
    print(f"    -> {os.path.basename(stats_path)}  ({len(stats_df)} 行)")

    grid, stacked, t_end = align_series(all_sr)
    durations = [float(sr["secs"][-1]) for sr in all_sr]
    if max(durations) - min(durations) > 1e-6:
        print(f"    [note] trial 时长不一致 "
              f"({min(durations):.1f}~{max(durations):.1f}s)，"
              f"跨 trial 平均截断到 {t_end:.1f}s")

    ts = pd.DataFrame({"time": grid})
    for k in ("largest", "mean_size", "alignment", "n_groups", "n_multi",
              "largest_frac"):
        ts[f"{k}_mean"] = stacked[k].mean(axis=0)
        ts[f"{k}_std"] = stacked[k].std(axis=0)
    for b, lab in enumerate(labels):
        ts[f"share_{lab}_mean"] = stacked["frac"][:, :, b].mean(axis=0)
        ts[f"share_{lab}_std"] = stacked["frac"][:, :, b].std(axis=0)
    ts_path = os.path.join(out_dir, f"{exp_name}_mean_timeseries.csv")
    ts.to_csv(ts_path, index=False)
    print(f"    -> {os.path.basename(ts_path)}")

    title = f"  [{exp_name}, max_dist={args.max_dist:g}]" if args.title else ""
    sm = args.smooth_s
    for fig_name, fn in (("mean_composition",
                          lambda p: fig_mean_composition(grid, stacked, labels,
                                                         p, title, sm)),
                         ("mean_groupsize",
                          lambda p: fig_mean_groupsize(grid, stacked, p, title, sm)),
                         ("mean_counts",
                          lambda p: fig_mean_counts(grid, stacked, p, title, sm))):
        if fig_name in args.figs:
            fp = os.path.join(out_dir, f"{exp_name}_{fig_name}.png")
            fn(fp)
            print(f"    -> {os.path.basename(fp)}")

    row = {"exp": exp_name, "n_trials": len(stats_df),
           "n_robots": int(stats_df["n_robots"].max()),
           "duration_s": t_end}
    for c in STEADY_COLS:
        row[f"{c}_mean"] = float(stats_df[c].mean())
        row[f"{c}_std"] = float(stats_df[c].std(ddof=0))
    print(f"    实验平均 (n={len(stats_df)}): 最大组末{args.steady_s:g}s "
          f"{row['largest_steady_mean']:.2f} ± {row['largest_steady_std']:.2f}, "
          f"组数 {row['n_groups_steady_mean']:.2f} ± "
          f"{row['n_groups_steady_std']:.2f}, "
          f"最大团占比 {row['largest_frac_steady_mean']:.3f} ± "
          f"{row['largest_frac_steady_std']:.3f}")
    return row


# =============================================================================
# 入口
# =============================================================================

def expand_experiments(patterns):
    out, seen = [], set()
    for pat in patterns:
        hits = glob.glob(pat) or ([pat] if os.path.isdir(pat) else [])
        for h in sorted(hits):
            d = os.path.normpath(h)
            if os.path.isdir(h) and d not in seen:
                seen.add(d)
                out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="批量画分组演化图，并对每组实验做 trial 之间的平均",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=EPILOG)
    ap.add_argument("experiments", nargs="*",
                    help="实验目录（里面装着 trial_XXXX/ 子文件夹），可用通配符")
    ap.add_argument("--dirs-from", default=None,
                    help="从文件读实验目录列表，每行一个（# 开头为注释）")
    ap.add_argument("--max-dist", type=float, default=65.0,
                    help="Voronoi 邻接距离门限，像素 (默认 65)")
    ap.add_argument("--no-singletons", action="store_true",
                    help="不把落单机器人算作 size=1 的组")
    ap.add_argument("--figs", nargs="+",
                    choices=list(FIG_TRIAL) + list(FIG_EXP) + list(FIG_ALIASES),
                    default=["all"], help="只画其中几张 (默认全画)")
    ap.add_argument("--out", default=None,
                    help="所有图和 csv 输出到这个目录 (默认放回各自的目录)")
    ap.add_argument("--fps", type=float, default=None,
                    help="每秒记录帧数 (默认读 config_snapshot.json)")
    ap.add_argument("--smooth-s", type=float, default=2.0,
                    help="曲线/堆叠图的滑动平均窗口，秒；0 = 不平滑 (默认 2)")
    ap.add_argument("--steady-s", type=float, default=10.0,
                    help="末尾多少秒算作稳态，用于标量统计 (默认 10)")
    ap.add_argument("--min-size", type=int, default=2,
                    help="kymograph 里算作'成组'的最小规模")
    ap.add_argument("--min-life-s", type=float, default=1.5,
                    help="kymograph 里能拿到独立颜色的组至少要存活多少秒")
    ap.add_argument("--snapshots", type=int, default=5,
                    help="空间快照画几个时刻")
    ap.add_argument("--title", action="store_true",
                    help="在图标题里标出实验/trial 名和 max_dist")
    ap.add_argument("--force", action="store_true",
                    help="忽略已有的分组缓存，重新计算")
    ap.add_argument("--summary", default=None,
                    help="把每组实验的汇总写成一个 csv")
    args = ap.parse_args()

    figs = set()
    for f in args.figs:
        figs |= set(FIG_ALIASES.get(f, (f,)))
    args.figs = figs

    patterns = list(args.experiments)
    if args.dirs_from:
        with open(args.dirs_from, encoding="utf-8") as f:
            patterns += [ln.strip() for ln in f
                         if ln.strip() and not ln.lstrip().startswith("#")]
    if not patterns:
        ap.error("至少要给一个实验目录（或用 --dirs-from）")

    exps = expand_experiments(patterns)
    if not exps:
        print("没找到任何目录。位置参数应当是实验目录，"
              "里面装着 trial_XXXX/ 子文件夹。")
        return 1

    print(f"共 {len(exps)} 组实验，max_dist={args.max_dist:g}，"
          f"要画 {', '.join(sorted(args.figs))}")
    rows = []
    for i, exp in enumerate(exps, 1):
        print(f"[{i}/{len(exps)}] {exp}")
        try:
            r = process_experiment(exp, args)
            if r:
                rows.append(r)
        except Exception as e:
            print(f"  [FAIL] {exp}: {type(e).__name__}: {e}")

    if args.summary and rows:
        pd.DataFrame(rows).to_csv(args.summary, index=False)
        print(f"\n汇总已写入 {args.summary}")
    print(f"\n完成 {len(rows)}/{len(exps)} 组实验")
    return 0 if rows else 1


if __name__ == "__main__":
    sys.exit(main())

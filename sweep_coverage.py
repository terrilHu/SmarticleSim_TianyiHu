"""
sweep_coverage.py  ─  手臂扫掠覆盖率 k = A' / A_total。

定义
----
每台机器人的**活动范围**由三块组成：中心 body 的矩形，加上两端手臂各自扫出的
一个扇形。左臂铰接在 body 局部坐标 (-main_len/2, 0)，关节角为 0 时指向 -x；
右臂铰接在 (+main_len/2, 0)，指向 +x。关节角在 [-A, +A] 之间摆动，所以每条臂
扫出一个以肩点为圆心、半径 arm_len、张角 2A 的扇形。

    A_0 = main_len * main_w + (A1 + A2) * arm_len^2

半角 A 的扇形面积是 (1/2)*r^2*(2A) = A*r^2，所以两臂合起来是 (A1+A2)*arm_len^2。
A_0 只由振幅(和固定的几何)决定 —— 位置和朝向都不影响它。

    A'      = 全部 n 个活动范围在平面上的**并集**面积（重叠只算一次，A' <= n*A_0）
    A_total = 环的内部面积
    k       = A' / A_total

k 越大表示活动范围把场地占得越满、彼此重叠越多，交互和碰撞越剧烈。

并集**不按环的边缘裁剪**：靠墙的机器人扇形伸到墙外的部分照样计入 A'。
因此 k 可以略大于 1 —— 分母是环的面积，分子却能覆盖到环外最多一个
footprint_radius 的范围。画布也据此比环大出这一圈，否则并集会在画布边界上
被悄悄截断（那等于换了个地方裁剪）。

为什么 A <= pi/2 时 A_0 是精确的
--------------------------------
扇形以 body 端点为圆心、绕 ±x 向外张开，A <= 90 度时整个扇形落在 x <= -main_len/2
(或 x >= +main_len/2) 的半平面里，而 body 矩形占的是 |x| <= main_len/2，
所以三块两两不重叠，面积可以直接相加。JOINT_LIMIT_DEG 是 85 度，
振幅表最大 pi/2，都在这个范围内。A > pi/2 时上式会高估，函数会给出警告。

并集怎么算
----------
n 个「矩形 + 两扇形」的精确并集需要 shapely 一类的几何库，环境里没有，
所以用**栅格化**：在环的包围盒上铺一张 cell x cell 的布尔画布，逐个机器人把
自己的活动范围 OR 进去，最后数被覆盖的格子。误差是 O(周长 * cell)，
cell 越小越准；calibrate 里实测了收敛情况，见 COVERAGE_CELL 的注释。

只在每个机器人自己的包围盒范围内做判定，所以每帧代价是
n * (2*L_s/cell)^2 个格点，而不是整张画布 x n。
"""

import math

import numpy as np


# =============================================================================
# 单台机器人：解析面积与栅格判定
# =============================================================================

def footprint_area(sm) -> float:
    """
    A_0 —— 一台机器人活动范围的解析面积（body 矩形 + 两个扇形）。

    读的是实例上**当前**的 A1/A2，所以运行时改了步态振幅，A_0 会跟着变。
    """
    return float(sm.main_len * sm.main_w
                 + (sm.A1 + sm.A2) * sm.arm_len ** 2)


def footprint_radius(sm) -> float:
    """活动范围的外接半径（以 body 中心为原点），用来算包围盒。"""
    # 最远的点是臂尖：肩点到中心 main_len/2，再加臂长。
    return 0.5 * sm.main_len + sm.arm_len


def _mask_local(u, v, main_len, main_w, arm_len, A1, A2):
    """
    在机器人体坐标系里判定格点是否落在活动范围内。

    u, v 是同形状的数组（体坐标，u 沿 body 长轴）。返回同形状的布尔数组。

    扇形判定不开方：点 (ru, rv) 相对肩点，落在朝 -x、半角 A 的扇形内
    等价于  ru <= 0  且  rv^2 * cos^2(A) <= ru^2 * sin^2(A)  且  ru^2+rv^2 <= r^2。
    A = pi/2 时 cos = 0，条件退化成半圆盘；A = 0 时退化成一条线段（零面积）。
    两种极端都自然成立，不用特判。
    """
    half_len = 0.5 * main_len
    inside = (np.abs(u) <= half_len) & (np.abs(v) <= 0.5 * main_w)

    r2 = arm_len * arm_len

    # 左臂：肩点 (-half_len, 0)，朝 -x
    ru = u + half_len
    c2, s2 = math.cos(A1) ** 2, math.sin(A1) ** 2
    inside |= ((ru <= 0.0) & (ru * ru + v * v <= r2)
               & (v * v * c2 <= ru * ru * s2))

    # 右臂：肩点 (+half_len, 0)，朝 +x
    qu = u - half_len
    c2, s2 = math.cos(A2) ** 2, math.sin(A2) ** 2
    inside |= ((qu >= 0.0) & (qu * qu + v * v <= r2)
               & (v * v * c2 <= qu * qu * s2))

    return inside


# =============================================================================
# 环
# =============================================================================

def ring_area(inner_r, shape="circle", n_sides=None) -> float:
    """A_total —— 环内部面积。圆是 pi*R^2；正 n 边形(外接半径 R)是 (n/2)R^2 sin(2pi/n)。"""
    if (shape or "circle").lower() == "polygon":
        n = max(3, int(n_sides))
        return 0.5 * n * inner_r * inner_r * math.sin(2.0 * math.pi / n)
    return math.pi * inner_r * inner_r


# =============================================================================
# 覆盖率
# =============================================================================

class CoverageMeter:
    """
    反复计算 k 的对象：画布和环掩膜只建一次，之后每帧复用。

    用法：
        meter = CoverageMeter(center, INNER_R, RING_SHAPE, RING_N_SIDES, cell=2.0)
        k, info = meter.measure(smarticles)
    """

    def __init__(self, center, inner_r, ring_shape="circle", n_sides=None,
                 cell=2.0):
        """
        cell  栅格边长(像素)。越小越准，代价按 1/cell^2 增长。

        画布在 measure() 里按**实际** footprint 的包围盒开：只会长大，不会缩，
        所以稳态下每个 trial 至多重开几次。格点始终锚在 (cx-R, cy-R) 这条
        栅格上、pad 取整格数，因此扩张不会改变格点相位 —— k 序列不会在
        重开的那一帧出现台阶。

        自动定尺寸是为了保证「不裁剪」是真的：按环半径加一个固定余量开画布的话，
        跑到余量之外的机器人会被画布边界悄悄切掉，那等于换了个地方裁剪。
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
        保证画布覆盖 (cx, cy) 周围 inner_r + need_pad 的方框。

        pad 向上取到整格数：x0 = cx - R - pad，pad 增加整格数时原来的格点
        仍然是格点(下标平移而已)，所以扩张前后同一个 footprint 落在相同的
        格子上，A' 不会因为重开画布而跳变。
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
        # 格点中心坐标（一维），二维用广播拼出来，省一张 nx*ny 的浮点数组
        self._xs = self.x0 + (np.arange(self.nx) + 0.5) * cell
        self._ys = self.y0 + (np.arange(self.ny) + 0.5) * cell
        self._canvas = np.zeros((self.ny, self.nx), dtype=bool)

    # ── 主接口 ───────────────────────────────────────────────────────────────

    def measure(self, smarticles):
        """
        返回 (k, info)。info 里有：
            A_union       并集面积（栅格估计，不按环裁剪）
            A0_sum        sum(A_0)，按定义的解析值
            A0_sum_raster sum(A_0) 的栅格值 —— 和 A_union 同一套离散化
            A_total       环面积（解析值）
            overlap       1 - A_union / A0_sum_raster，重叠掉的比例
            k_naive       A0_sum / A_total，忽略重叠时的上界

        overlap 用的是 A0_sum_raster 而不是解析的 A0_sum：两者都带同样的
        栅格误差，相除时抵消。拿栅格的 A_union 去比解析的 A0_sum 会留下
        1~2% 的系统偏差 —— 实测两台完全分开的机器人会报出 1.8% 的"重叠"。
        """
        if not len(smarticles):
            return 0.0, {"A_union": 0.0, "A0_sum": 0.0, "A0_sum_raster": 0.0,
                         "A_total": self.A_total, "overlap": 0.0,
                         "k_naive": 0.0}

        # 先量出所有 footprint 的包围盒，再决定画布要多大
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

            # 该机器人在画布上的索引窗口（含一格余量）
            i0 = int(math.floor((px - rad - x0) / cell))
            i1 = int(math.ceil((px + rad - x0) / cell)) + 1
            j0 = int(math.floor((py - rad - y0) / cell))
            j1 = int(math.ceil((py + rad - y0) / cell)) + 1
            i0 = 0 if i0 < 0 else i0
            j0 = 0 if j0 < 0 else j0
            i1 = nx if i1 > nx else i1
            j1 = ny if j1 > ny else j1
            if i0 >= i1 or j0 >= j1:
                continue                      # 完全在画布外

            dx = self._xs[i0:i1][None, :] - px
            dy = self._ys[j0:j1][:, None] - py

            # 转到体坐标：绕 -psi 旋转
            c, s = math.cos(psi), math.sin(psi)
            u =  dx * c + dy * s
            v = -dx * s + dy * c

            mask = _mask_local(u, v, sm.main_len, sm.main_w,
                               sm.arm_len, sm.A1, sm.A2)
            # 这一台单独占的格数，顺手数掉：mask 本来就算出来了，
            # 用它算 overlap 才和 A_union 同一套离散化。
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
    """一次性算一帧的 k（内部就是建一个 CoverageMeter）。"""
    return CoverageMeter(center, inner_r, ring_shape, n_sides,
                         cell=cell).measure(smarticles)

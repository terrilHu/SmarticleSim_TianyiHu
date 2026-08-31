"""
spatial_roles.py  ─  按空间关系挑出一组"角色"机器人。

例如"x 最小/最大、y 最小/最大的那台"(阵形的四个极值点)，
或"主轴(PCA)长轴/短轴两端的机器人"(与坐标系无关，跟着阵形自身的朝向走)，
主轴既可以用全场所有机器人拟合，也可以只用当前最大的那个分组来定方向，
再把全场机器人投影到该方向上取两端。
这些机器人固定执行一条单独的指令，优先级低于"处于大分组内"。

与分组控制的关系是分层的，在 strategy.py 里合成最终指令再发：

    第 1 层(最高)  已确认分组的成员  -> 该组的指令
    第 2 层        空间角色          -> 角色指令
    第 3 层(兜底)  其它              -> leave_command

所以一台机器人既是 x 最小又在大分组里时，执行的是分组指令 ── 角色不会盖掉它。

选择器全是"吃一帧 (ids, pos) 吐一组 marker id"的纯函数，
所以加新规则只要写一个函数并登记到 SELECTORS，不用碰控制器。

本文件与实机侧同名文件保持一致的函数名与语义，这样同一套角色定义可以
在仿真和实机上得到可比的结果；仿真里机器人不会丢失，miss_tolerance 相关
的分支恒不触发，保留只是为了两边同构。
"""

from typing import Callable, Dict, List, Set

import numpy as np


def needs_group_record(fn):
    """
    标记该选择器除了 (ids, pos) 还需要整帧的分组结果 rec。
    用显式标记而不是靠签名推断: 自己写的选择器加不加这个装饰器一目了然，
    没加的照旧只收 (ids, pos)，不会因为参数名手滑而被意外传入 rec。
    """
    fn.needs_rec = True
    return fn


# =============================================================================
# 选择器：(ids, pos) -> set(marker id)
#   ids (k,)  marker id
#   pos (k,2) 像素坐标，注意图像坐标系 y 轴向下
# =============================================================================

def sel_extremes(ids, pos) -> Set[int]:
    """x 最小/最大、y 最小/最大 ── 阵形的四个极值点。

    并列时 argmin/argmax 取第一个，所以同一台机器人可能同时占两个角色
    (比如它既最靠左又最靠上)，此时返回集自然只有一个它，数量少于 4。
    """
    if len(ids) == 0:
        return set()
    return {int(ids[i]) for i in (np.argmin(pos[:, 0]), np.argmax(pos[:, 0]),
                                  np.argmin(pos[:, 1]), np.argmax(pos[:, 1]))}


def sel_x_extremes(ids, pos) -> Set[int]:
    """只取 x 最小/最大"""
    if len(ids) == 0:
        return set()
    return {int(ids[np.argmin(pos[:, 0])]), int(ids[np.argmax(pos[:, 0])])}


def sel_y_extremes(ids, pos) -> Set[int]:
    """只取 y 最小/最大"""
    if len(ids) == 0:
        return set()
    return {int(ids[np.argmin(pos[:, 1])]), int(ids[np.argmax(pos[:, 1])])}


def sel_convex_hull(ids, pos) -> Set[int]:
    """整个阵形凸包上的机器人 ── 比四个极值点更完整的"边缘"定义"""
    if len(ids) < 3:
        return {int(i) for i in ids}
    try:
        from scipy.spatial import ConvexHull
        return {int(ids[i]) for i in ConvexHull(pos).vertices}
    except Exception:
        return sel_extremes(ids, pos)


def sel_farthest_from_centroid(ids, pos, n=4) -> Set[int]:
    """离质心最远的 n 台"""
    if len(ids) == 0:
        return set()
    d = np.linalg.norm(pos - pos.mean(axis=0), axis=1)
    return {int(ids[i]) for i in np.argsort(d)[::-1][:n]}


def _ends_by_projection(ids, pos, V, cols, n_per_end):
    """把点投影到给定的轴上，取每条轴两端各 n 台。

    投影用的原点不影响结果：平移只是给所有投影加同一个常数，argsort 不变。
    所以这里不必纠结该用组的质心还是全场质心。
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
    对点云做主成分分析(PCA)，返回 (中心, 轴向标准差, 特征向量, 长短比)。

        sigma    升序 [σ_短, σ_长]，即 sqrt(特征值)，单位与坐标相同(像素)
        eigvecs  列向量, eigvecs[:,0]=短轴方向, eigvecs[:,1]=长轴方向, 两者正交
        ratio    σ_长/σ_短, 即"长轴比短轴长多少倍"

    这里返回标准差而不是特征值: 特征值是方差, 比值是长度比的**平方**,
    拿它当阈值很容易看走眼(比值 1.5 其实只对应 1.22:1 的形状)。
    开根号之后 ratio 就是直观的长宽比。

    用 eigh 而不是 eig: 协方差矩阵是对称的, eigh 保证实数解且特征值升序。
    """
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    ctr = pos.mean(axis=0)
    cov = np.cov((pos - ctr).T)
    w, V = np.linalg.eigh(cov)
    sigma = np.sqrt(np.maximum(w, 0.0))          # 数值误差可能给出极小的负值
    ratio = float(sigma[1] / sigma[0]) if sigma[0] > 1e-9 else float("inf")
    return ctr, sigma, V, ratio


def sel_principal_ends(ids, pos, axis="both", n_per_end=1, min_anisotropy=1.5):
    """
    用全部机器人的位置拟合主轴(PCA)，取长轴和/或短轴两端的机器人。

        axis           "major" 只取长轴两端 / "minor" 只取短轴两端 / "both" 两条轴都取
        n_per_end      每端取几台(按投影最靠外的 n 台)
        min_anisotropy 长短比(σ长/σ短)低于此值时判定"阵形太圆、主轴方向没意义"，
                       返回空集。这一条很重要: 接近圆形时特征向量方向由噪声决定,
                       实测长宽比 1.1 时长轴方向每帧乱跳 40 度以上,
                       不设门槛的话选出来的就是随机两台。默认 1.5 对应 1.5:1 的形状。

    注意主轴方向有正负号歧义(V 和 -V 都是特征向量), 但这里两端一起取,
    所以选出的集合与符号无关, 不会因为符号翻转而在两端之间跳。

    返回空集表示"本帧无法定义该角色"; SpatialRoleTracker 会把它当作
    "本帧无信息"而保持现任角色不变, 不会把大家撤任。
    """
    ids = np.asarray(ids)
    pos = np.asarray(pos, dtype=float).reshape(-1, 2)
    if len(ids) < 3:
        return {int(i) for i in ids}

    _ctr, sigma, V, ratio = principal_axes(pos)
    if sigma[1] < 1e-9:            # 所有点重合, 连长轴都没有
        return set()
    if ratio < min_anisotropy:     # 太圆, 轴向由噪声决定
        return set()

    cols = {"minor": [0], "major": [1], "both": [0, 1]}[axis]
    # 短轴长度相对长轴可忽略(阵形几乎共线)时, 短轴两端同样是噪声, 跳过该轴
    if sigma[0] < 1e-6 * sigma[1]:
        cols = [c for c in cols if c != 0]
        if not cols:
            return set()

    return _ends_by_projection(ids, pos, V, cols, n_per_end)


def sel_major_ends(ids, pos, **kw):
    """长轴两端 ── 阵形最伸展方向上最靠前和最靠后的机器人"""
    return sel_principal_ends(ids, pos, axis="major", **kw)


def sel_minor_ends(ids, pos, **kw):
    """短轴两端 ── 阵形最窄方向上的两侧边缘"""
    return sel_principal_ends(ids, pos, axis="minor", **kw)


@needs_group_record
def sel_largest_group_axis_ends(ids, pos, rec=None, axis="both", n_per_end=1,
                                min_anisotropy=1.5, min_group_size=3,
                                max_group_size=None, min_lead=0,
                                select_from="all"):
    """
    用**当前最大的分组**拟合主轴，只取它的**方向**，
    再把机器人投影到这两个方向上，选出两端的个体。

    拆成两步是有意的：
      - 方向由那个团决定 ── 团外个体和小团不会把轴带偏，量的是团自己的朝向；
      - 端点默认在**全场所有机器人**里挑(select_from="all") ── 选出来的可能
        是团外那些沿该方向更靠外的机器人，这正是"沿团的朝向最前/最后"的含义。
        想只在团内部选就把 select_from 设成 "group"。

        axis            "major" / "minor" / "both"
        n_per_end       每端取几台
        min_anisotropy  该组长短比低于此值 -> 方向由噪声决定, 返回空集
        min_group_size  最大组规模低于此门槛 -> 返回空集(还没形成值得追踪的团)
        max_group_size  最大组规模高于此上限 -> 返回空集。None = 不设上限。
                        通常用角色层的 group_size_range 来控制更方便，
                        这里保留是为了让选择器单独使用时也能锁定范围
        min_lead        最大组要比第二大的组多出至少这么多台才作数。两组规模接近时
                        "哪个最大"会逐帧易主，轴向会整个跳到另一群机器人身上；
                        设 1~2 可把这种情况判为不确定。默认 0 = 不要求。
        select_from     "all" 在全场机器人里选端点(默认) / "group" 只在该组内选

    返回空集表示"本帧无法定义该角色"，SpatialRoleTracker 会保持现任角色不变。
    """
    if rec is None or not len(rec.get("sizes", ())):
        return set()
    sizes = np.asarray(rec["sizes"])
    order = np.argsort(-sizes, kind="mergesort")   # 稳定排序: 同规模时保持原有顺序
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
        return set()                      # 点太少，拟合不出方向

    # 第 1 步：只用该组的成员定方向
    _ctr, sigma, V, ratio = principal_axes(all_pos[sel])
    if sigma[1] < 1e-9 or ratio < min_anisotropy:
        return set()
    cols = {"minor": [0], "major": [1], "both": [0, 1]}[axis]
    if sigma[0] < 1e-6 * sigma[1]:        # 该组几乎共线，短轴方向没意义
        cols = [c for c in cols if c != 0]
        if not cols:
            return set()

    # 第 2 步：把机器人投影到这两个方向上取两端
    if select_from == "group":
        return _ends_by_projection(all_ids[sel], all_pos[sel], V, cols, n_per_end)
    return _ends_by_projection(all_ids, all_pos, V, cols, n_per_end)


@needs_group_record
def sel_largest_group_major_ends(ids, pos, rec=None, **kw):
    """按最大分组的长轴方向，取两端的机器人"""
    return sel_largest_group_axis_ends(ids, pos, rec, axis="major", **kw)


@needs_group_record
def sel_largest_group_minor_ends(ids, pos, rec=None, **kw):
    """按最大分组的短轴方向，取两端的机器人"""
    return sel_largest_group_axis_ends(ids, pos, rec, axis="minor", **kw)


SELECTORS: Dict[str, Callable] = {
    "extremes": sel_extremes,
    "x_extremes": sel_x_extremes,
    "y_extremes": sel_y_extremes,
    "convex_hull": sel_convex_hull,
    "farthest": sel_farthest_from_centroid,
    "principal_ends": sel_principal_ends,     # 长轴+短轴共四端
    "major_ends": sel_major_ends,             # 只要长轴两端
    "minor_ends": sel_minor_ends,             # 只要短轴两端
    # 下面三个只用"当前最大的分组"拟合主轴，不受团外个体和小团影响
    "group_axis_ends": sel_largest_group_axis_ends,
    "group_major_ends": sel_largest_group_major_ends,
    "group_minor_ends": sel_largest_group_minor_ends,
}


# =============================================================================
# 带迟滞的角色追踪
# =============================================================================

class SpatialRoleTracker:
    """
    维护"当前担任空间角色"的机器人数组。

    极值点每帧都可能因为一两个像素的抖动而易主 ── 两台机器人 x 坐标只差 3px 时，
    谁是最小值纯粹是噪声。所以和分组一样用连续帧数做迟滞：

        连续 n_frames_join  帧被选中 -> 加入数组
        连续 n_frames_leave 帧未被选中 -> 移出数组

    代价是成员数不再恒等于选择器返回的个数，两个方向都可能偏：
      - 多了：交接期间旧的还没退、新的已经进；
      - 少了：两台严格交替领先时谁都攒不满连续帧，于是谁都不担任该角色
        (这种情况下"谁是最左"本来就无法判定，空着比每帧换人好)。
    需要角色数严格固定就把 n_frames_join 设为 1，代价是又变回每帧易主。
    """

    def __init__(self, selector="extremes", n_frames_join=6, n_frames_leave=6,
                 miss_tolerance=2, verbose=True, label=None, **kwargs):
        """
        label   角色层的名字，多层并存时用来区分日志和事件记录
        kwargs  原样转交给选择器，例如 PCA 系列的 axis / n_per_end / min_anisotropy
        """
        if callable(selector):
            self.selector = selector
            self.name = getattr(selector, "__name__", "custom")
        else:
            if selector not in SELECTORS:
                raise ValueError(f"未知的空间角色选择器: {selector!r}; "
                                 f"可选: {sorted(SELECTORS)}")
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
        吃一帧的 (ids, pos)，返回当前的角色成员集合。
        rec 是整帧的分组结果，只转交给标了 @needs_group_record 的选择器。

        enabled=False 表示"本帧该层不适用"(比如最大组规模跑出了配置的范围)。
        此时按"明确没被选中"处理，让现任成员走正常的 n_frames_leave 迟滞退出，
        而不是立刻全体撤任 ── 规模在范围边界上抖动时才不会来回刷指令。
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

        # 选择器返回空集有两种含义：本帧没数据，或者判定"此刻无法定义该角色"
        # (PCA 选择器在阵形接近圆形时就会这样)。两种情况都不该立刻把现任角色
        # 全部撤任 ── 那会造成指令抖动。这里当作"本帧无信息"，计数原地保持。
        # 但 enabled=False 是明确的"不适用"，要走退出流程，所以排除在外。
        if enabled and not chosen and len(ids):
            return set(self.members)

        # 本帧没看到的，容忍若干帧再按"未被选中"计，避免漏检直接把角色踢掉
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
                    print(f"[role] {mid} 加入角色层 {self.label}, "
                          f"当前 {sorted(self.members)}")
            elif mid in self.members and self.streak_out.get(mid, 0) >= self.n_leave:
                self.members.discard(mid)
                self._log(ts, "role_leave", mid)
                if self.verbose:
                    print(f"[role] {mid} 退出角色层 {self.label}, "
                          f"当前 {sorted(self.members)}")

        # 长期看不见的直接移出，它可能已经出局面了
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

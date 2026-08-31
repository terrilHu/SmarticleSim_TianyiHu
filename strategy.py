"""
strategy.py  ─  分组 + 空间角色 -> 分层指令的运行时策略。

流程（每个策略 tick 走一遍）：

    位置/朝向
      │
      ├─ Voronoi(Delaunay) 邻接 + max_dist 门限   pos_all_alignment.voronoi_adjacency
      ├─ 连通分量 -> 分组                          pos_all_grouping.compute_frame_groups
      │      -> rec = {ids, pos, theta, labels, sizes, alignment, centroid}
      │
      ├─ 第 1 层  按分组规模匹配规则  -> 组指令      GroupRuleLayer(带迟滞)
      ├─ 第 2 层  空间角色            -> 角色指令    spatial_roles.SpatialRoleTracker
      └─ 第 3 层  兜底                -> leave_command
                     │
                     └─ 合成 {robot_id: command} 交给 gait.GaitController

分层优先级默认和实机侧一致：**分组指令盖过角色指令**。一台机器人既是长轴端点
又在大分组里时，执行的是分组指令 —— 角色只管那些没被任何分组规则认领的机器人。
给某条角色加 "override_group": True 可以把它单独提到分组层之上。
同一层内，roles 列表**靠前**的优先级更高。

分组用的是 pos_all_alignment / pos_all_grouping 里那两个函数本身，不是另写一份：
离线分析(_alignment.csv、A(n) 曲线)看到的分组，和策略当场依据的分组是同一个定义，
所以"当时为什么发这条指令"事后可以在分析结果里对上。

指令一律是 config.py 那套 ±XYZ 整数。发出去的只是**请求**：
gait.GaitController 会把它排到该机器人下一次相位过零点再生效，
所以角色交接不会把手臂猛拽一下。

调参
====

max_dist —— 这里最要紧的一个数，而且它会**渗流**
------------------------------------------------
在真实 100 机器人 trial 上实测(6 个快照, t = 5~17.5 s)，门限以体长
L_s = (MAIN_LEN + 2*ARM_LEN)/2 = 105 px 为单位：

    max_dist/L_s   组数    最大组   前几大
       0.70        46.7     7.5    [7, 6, 4, 4, 4, 3]
       1.00        26.8    19.0    [17, 13, 11, 5, 5, 5]
       1.10        17.2    29.5    [21, 19, 18, 6, 5, 5]    <- 默认
       1.20         4.8    84.5    [89, 10, 1]
       1.30         1.8    99.0    [100]
       1.43         1.0   100.0    [100]     (= 离线分析用的 150 px)

超过 ~1.2*L_s 整群塌成一个组：所有机器人匹配同一条规则，PCA 角色选择器又因为
满场是圆的而弃权，策略直接退化。150.0 对离线 alignment 分析没问题(那里邻域宽松
正是目的)，但拿来区分分组没用。默认写成体长的倍数，这样 N_SMARTICLES(以及跟着
变的场地)改了之后含义不变。

两个坑，改层之前先看
--------------------
(a) **第 1 层的规则如果无上界，会把第 2 层饿死。** 第 1 层优先级最高，所以
    开放上界的 "size_range": (8, None) 会在群一旦渗流成单个团时认领全场，
    角色从此再也不会生效。实测：用那条规则跑完是一个 94 台的组 + 全场统一
    -851。把上界收住，渗流态就留给了角色层。
(b) **PCA 系列选择器在圆形聚集体上会弃权**，这是设计使然 —— 圆盘的主轴方向
    是噪声(见 spatial_roles 的 min_anisotropy)。group_major_ends 适合**拉长**
    的阵形；紧凑的群要用形状无关的选择器，例如 convex_hull(边界)或 farthest。
"""

import inspect
import math

import numpy as np

from spatial_roles import (SELECTORS, SpatialRoleTracker,
                           sel_largest_group_axis_ends)


# =============================================================================
# 一帧的分组结果
# =============================================================================

def frame_record(ids, pos, theta, max_dist, include_singletons=True):
    """
    跑一帧 Voronoi+门限分组，返回选择器要的 rec。

    键名是 spatial_roles 那边定死的契约：
        ids     (n,)   机器人编号
        pos     (n,2)  位置
        theta   (n,)   朝向(弧度)
        labels  (n,)   每台机器人所属的组序号，索引进 sizes/alignment/centroid
        sizes   (g,)   每组的成员数
        alignment (g,) 每组的向列序参量 |<exp(2i*theta)>| ∈ [0,1]
        centroid  (g,2) 每组质心
        groups  compute_frame_groups() 的原始返回，需要更多信息时可直接取用
    """
    # 惰性导入：pos_all_grouping 会拉进 pandas，策略没启用时不该付这个代价。
    from pos_all_grouping import compute_frame_groups

    ids   = np.asarray(ids)
    pos   = np.asarray(pos, dtype=float).reshape(-1, 2)
    theta = np.asarray(theta, dtype=float)

    groups = compute_frame_groups(pos, theta, ids, max_dist,
                                  include_singletons=include_singletons)

    labels = np.full(len(ids), -1, dtype=int)      # -1 = 不属于任何组
    for gi, g in enumerate(groups):
        labels[g["local_idx"]] = gi

    return {
        "ids":       ids,
        "pos":       pos,
        "theta":     theta,
        "labels":    labels,
        "sizes":     np.array([g["size"] for g in groups], dtype=int),
        "alignment": np.array([g["alignment"] for g in groups], dtype=float),
        "centroid":  np.array([g["centroid"] for g in groups],
                              dtype=float).reshape(-1, 2),
        "groups":    groups,
    }


def _check_selector_kwargs(selector, kwargs, where):
    """
    选择器多余参数的早期校验。写错参数名(比如给 convex_hull 传 n_per_end)
    本来会在运行时抛 TypeError，而策略是在 gait 回调里构造的，异常会被
    GaitController.poll 吞成一行警告，之后每帧静默无事发生。这里当场报错。
    """
    if not kwargs:
        return
    fn = SELECTORS[selector] if isinstance(selector, str) else selector
    if isinstance(selector, str) and selector not in SELECTORS:
        raise ValueError(f"{where}: 未知的 selector {selector!r}; "
                         f"可选 {sorted(SELECTORS)}")
    def _named(f):
        # 只要具名参数：**kw / *args 这两个形参名本身不是可传的参数名
        return {name for name, prm in inspect.signature(f).parameters.items()
                if prm.kind not in (inspect.Parameter.VAR_KEYWORD,
                                    inspect.Parameter.VAR_POSITIONAL)}

    accepted = _named(fn)
    if any(prm.kind is inspect.Parameter.VAR_KEYWORD
           for prm in inspect.signature(fn).parameters.values()):
        # 形如 sel_largest_group_major_ends(ids, pos, rec=None, **kw)，
        # 参数最终落到它转调的那个函数上
        accepted |= _named(sel_largest_group_axis_ends)
    accepted -= {"ids", "pos", "rec"}
    unknown = sorted(set(kwargs) - accepted)
    if unknown:
        raise ValueError(
            f"{where}: selector {selector!r} 不接受参数 {unknown}; "
            f"它能接受的是 {sorted(accepted)}")


def _in_range(value, rng):
    """rng = (lo, hi)，任一端为 None 表示不设限；两端都含。"""
    if rng is None:
        return True
    lo, hi = rng
    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


# =============================================================================
# 第 1 层：按分组规模发指令（带迟滞）
# =============================================================================

class GroupRuleLayer:
    """
    把每台机器人所在组的规模(以及可选的向列序)匹配到一条规则上，发该规则的指令。

    和角色层一样要迟滞。分组是逐帧重算的，一台机器人在团边缘晃动时，
    它和团的 Voronoi 边可能这帧短于 max_dist、下帧又超出，组规模就在
    门槛两侧来回跳。直接按当帧结果发指令会让它每隔几帧换一次步态 ——
    这既不是实机能做到的，也会污染"分组规模 vs 步态"的因果关系。
    所以：连续 n_ticks 个 tick 匹配到同一条规则，才认为这台机器人
    **已确认**属于该规则，指令才发出去。

    规则按列表顺序匹配，第一条命中的生效，所以把窄的范围写在前面。
    """

    def __init__(self, rules, n_ticks=6, verbose=False, label="group"):
        """
        rules   [{"size_range": (lo, hi), "command": int,
                  "alignment_range": (lo, hi) 可选,
                  "name": str 可选}, ...]
        n_ticks 连续命中多少个 tick 才确认（1 = 不迟滞）
        """
        self.rules = []
        for i, r in enumerate(rules):
            rule = dict(r)
            if "command" not in rule:
                raise ValueError(f"分组规则 #{i} 缺少 'command'")
            rule.setdefault("size_range", None)
            rule.setdefault("alignment_range", None)
            rule.setdefault("name", f"rule{i}")
            self.rules.append(rule)

        self.n_ticks = max(1, int(n_ticks))
        self.verbose = verbose
        self.label   = label

        self.assigned = {}      # robot_id -> 已确认的规则序号
        self._streak  = {}      # robot_id -> (规则序号, 连续命中数)
        self.events   = []
        self.ticks    = 0

    def _match(self, size, alignment):
        for i, rule in enumerate(self.rules):
            if not _in_range(size, rule["size_range"]):
                continue
            if not _in_range(alignment, rule["alignment_range"]):
                continue
            return i
        return None

    def update(self, ts, rec) -> dict:
        """返回 {robot_id: command}，只含已确认的机器人。"""
        self.ticks += 1
        ids    = rec["ids"]
        labels = rec["labels"]
        sizes  = rec["sizes"]
        align  = rec["alignment"]

        out = {}
        for k, rid in enumerate(ids):
            rid = int(rid)
            gi  = int(labels[k])
            hit = None if gi < 0 else self._match(int(sizes[gi]), float(align[gi]))

            prev_rule, streak = self._streak.get(rid, (None, 0))
            streak = streak + 1 if hit == prev_rule else 1
            self._streak[rid] = (hit, streak)

            if streak < self.n_ticks:
                # 还没确认：保持上一次确认的结果，别让边界抖动漏下来
                if rid in self.assigned:
                    out[rid] = self.rules[self.assigned[rid]]["command"]
                continue

            if hit is None:
                if rid in self.assigned:
                    old = self.assigned.pop(rid)
                    self._log(ts, "group_leave", rid, self.rules[old]["name"], gi)
                    if self.verbose:
                        print(f"[group] {rid} 离开 {self.rules[old]['name']}")
                continue

            if self.assigned.get(rid) != hit:
                self.assigned[rid] = hit
                self._log(ts, "group_join", rid, self.rules[hit]["name"], gi)
                if self.verbose:
                    print(f"[group] {rid} -> {self.rules[hit]['name']} "
                          f"(组规模 {int(sizes[gi])}) cmd={self.rules[hit]['command']}")
            out[rid] = self.rules[hit]["command"]

        return out

    def _log(self, ts, event, rid, rule_name, gi):
        self.events.append({"Time": ts, "tick": self.ticks, "event": event,
                            "robot_id": rid, "rule": rule_name, "group": gi,
                            "layer": self.label})


# =============================================================================
# 合成：分组 > 角色 > 兜底
# =============================================================================

class LayeredStrategy:
    """
    可以直接当 runtime gait 回调用的策略对象：

        config.RUNTIME_GAIT_CONTROLLER = "gait_control:strategy"

    每个 tick 重算分组、更新各层，再把三层合成成 {robot_id: command}。
    只有目标指令与现状不同、且该机器人没有排队中的切换时才发请求，
    所以稳态下返回空字典，不会每个 tick 都刷一遍队列。
    """

    def __init__(self, max_dist, leave_command,
                 group_rules=(), roles=(),
                 period=0.25, group_n_ticks=6,
                 include_singletons=True, verbose=False):
        """
        max_dist       Voronoi 邻接的距离门限(像素)。两台机器人是 Delaunay 邻居
                       且距离 <= 该值才算连通。离线分析用的是同一个参数。
        leave_command  第 3 层兜底：没被任何分组规则或角色认领的机器人执行它
        group_rules    见 GroupRuleLayer
        roles          [{"selector": str, "command": int,
                         "n_frames_join": int, "n_frames_leave": int,
                         "override_group": bool 可选(默认 False),
                         "enabled_when": {"largest_group_size": (lo, hi)} 可选,
                         "label": str 可选,
                         其余键原样转交选择器，如 axis / n_per_end /
                         min_anisotropy / min_group_size / select_from}, ...]

                       列表靠前的角色优先级更高（两个角色都选中同一台机器人时，
                       靠前那条的指令生效）。

                       override_group=False（默认）时该角色低于分组层：机器人
                       只要被某条 group_rule 认领，执行的就是分组指令。
                       override_group=True 时反过来，角色指令盖过分组指令 ——
                       想让"团的两端"无论团多大都执行特定步态时用这个。
        period         每隔多少秒重算一次(秒)。迟滞用的"帧数"数的是 tick，
                       所以 period=0.25 + n_frames_join=6 ≈ 1.5 秒才确认一个角色。
        """
        self.max_dist      = float(max_dist)
        self.leave_command = int(leave_command)
        self.period        = float(period)
        self.include_singletons = bool(include_singletons)
        self.verbose       = bool(verbose)

        self.group_layer = (GroupRuleLayer(group_rules, n_ticks=group_n_ticks,
                                           verbose=verbose)
                            if group_rules else None)

        self.roles = []
        for i, spec in enumerate(roles):
            spec = dict(spec)
            selector     = spec.pop("selector")
            command      = int(spec.pop("command"))
            enabled_when = spec.pop("enabled_when", None)
            override     = bool(spec.pop("override_group", False))
            label        = spec.pop("label", None)
            n_join       = spec.pop("n_frames_join", 6)
            n_leave      = spec.pop("n_frames_leave", 6)
            miss_tol     = spec.pop("miss_tolerance", 2)
            vb           = spec.pop("verbose", verbose)
            # 剩下的原样转交选择器；先按签名校验一遍，否则写错一个参数名会在
            # 回调里抛 TypeError 被 GaitController.poll 吞掉，只剩一行警告，
            # 之后每帧静默什么都不做 —— 很难查。
            _check_selector_kwargs(selector, spec, f"roles[{i}]")
            tracker = SpatialRoleTracker(
                selector=selector, n_frames_join=n_join,
                n_frames_leave=n_leave, miss_tolerance=miss_tol,
                verbose=vb, label=label, **spec)
            self.roles.append({"tracker": tracker, "command": command,
                               "enabled_when": enabled_when,
                               "override_group": override})

        self._next_t = 0.0
        self.rec     = None       # 最后一帧的分组结果，调试时可直接看
        self.ticks   = 0

    # ── 回调接口 ─────────────────────────────────────────────────────────────

    def __call__(self, t, robots, ctx):
        if t < self._next_t:
            return None
        self._next_t = t + self.period
        self.ticks += 1

        ids   = np.fromiter((r.id for r in robots), dtype=int, count=len(robots))
        pos   = np.array([(r.x, r.y) for r in robots], dtype=float).reshape(-1, 2)
        theta = np.fromiter((r.angle for r in robots), dtype=float,
                            count=len(robots))

        rec = frame_record(ids, pos, theta, self.max_dist,
                           include_singletons=self.include_singletons)
        self.rec = rec

        target = self.decide(t, rec, ids)

        # 只发真正需要改的：稳态下这里是空的
        req = {}
        for r in robots:
            want = target.get(r.id)
            if want is None or want == r.command or r.pending_cmd is not None:
                continue
            req[r.id] = want
        return req or None

    def decide(self, t, rec, ids) -> dict:
        """
        三层合成，返回每台机器人**应该**执行的指令（不管它现在执行的是什么）。
        单独调用它可以在不跑仿真的情况下测策略。
        """
        target = {int(i): self.leave_command for i in ids}          # 第 3 层

        # 先把所有角色层都 update 一遍(迟滞计数必须每 tick 都走)，
        # 记下成员，稍后再按优先级写入 target。
        largest = int(rec["sizes"].max()) if len(rec["sizes"]) else 0
        members = []
        for role in self.roles:
            enabled = True
            cond = role["enabled_when"]
            if cond is not None:
                enabled = _in_range(largest, cond.get("largest_group_size"))
            members.append(role["tracker"].update(t, rec["ids"], rec["pos"],
                                                  rec=rec, enabled=enabled))

        def _apply_roles(want_override):
            # 倒序写入，于是列表**靠前**的角色最后落笔、覆盖靠后的 ——
            # 这才是文档里说的"靠前的优先级更高"。
            for role, mids in zip(reversed(self.roles), reversed(members)):
                if role["override_group"] != want_override:
                    continue
                for mid in mids:
                    target[int(mid)] = role["command"]

        _apply_roles(False)                                         # 第 2 层
        if self.group_layer is not None:                            # 第 1 层
            for rid, cmd in self.group_layer.update(t, rec).items():
                target[int(rid)] = cmd
        _apply_roles(True)      # override_group 的角色盖在分组指令之上

        return target

    # ── 调试 ─────────────────────────────────────────────────────────────────

    def summary(self) -> str:
        """当前分组和各角色成员的一行速览。"""
        if self.rec is None:
            return "(还没跑过 tick)"
        sizes = sorted(self.rec["sizes"].tolist(), reverse=True)
        parts = [f"groups {len(sizes)} sizes {sizes[:8]}"
                 f"{'...' if len(sizes) > 8 else ''}"]
        for role in self.roles:
            tr = role["tracker"]
            parts.append(f"{tr.label}={sorted(tr.members)}->{role['command']:+d}")
        if self.group_layer is not None:
            n_assigned = len(self.group_layer.assigned)
            parts.append(f"grouped {n_assigned}")
        return "  |  ".join(parts)

    def events(self) -> list:
        """所有层的事件流，按时间排序。纯内存，不落盘。"""
        out = list(self.group_layer.events) if self.group_layer else []
        for role in self.roles:
            out.extend(role["tracker"].events)
        return sorted(out, key=lambda e: (e["Time"], e.get("event", "")))


# =============================================================================
# 从声明式配置构造
# =============================================================================

def build_strategy(spec) -> LayeredStrategy:
    """
    把 config.STRATEGY_SPEC 那样的 dict 变成 LayeredStrategy。
    spec 里未给出的键用 LayeredStrategy 的默认值。
    """
    spec = dict(spec or {})
    if "max_dist" not in spec or "leave_command" not in spec:
        raise ValueError("STRATEGY_SPEC 至少要有 'max_dist' 和 'leave_command'")
    return LayeredStrategy(**spec)

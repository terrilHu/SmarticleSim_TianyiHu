"""
experiments.py  ─  成组的对比实验：多种 strategy × 多种 N_SMARTICLES。

每个条件 = 一份 config 覆盖参数。驱动器把它写成一个覆盖文件，然后
**起一个新的 Python 进程**跑 simulation.main()，最后把各条件的 summary 汇总。

为什么必须换进程
----------------
smarticle.py / spawn.py / analysis.py 都是 `from config import MAIN_LEN, ...`
这种按值绑定，而这些量全部由 N_SMARTICLES 派生。进程内改 N_SMARTICLES 只会改
config 模块里的那个名字，已经绑好的几何常量不会跟着变 —— 于是会拿 N=17 的机器人
尺寸去跑 100 台的场地。每个条件一个新解释器是唯一可靠的做法（也顺带隔离了
config.py 在 import 时消耗的全局 RNG）。

用法
----
    python experiments.py --list            # 只列出会跑哪些条件
    python experiments.py --dry-run         # 生成覆盖文件但不跑
    python experiments.py                   # 全跑
    python experiments.py --only n17_edge n50_edge
    python experiments.py --out-root datafile/compare0901

也可以不用内置矩阵，直接给一串写好的 config 文件，依次每个跑一次实验：

    python experiments.py --configs conditions/a.py conditions/b.py
    python experiments.py --configs "conditions/*.py"      # 通配符
    python experiments.py --configs-from list.txt          # 每行一个路径

条件名默认取文件里的 EXP_NAME，没有就用文件名（去掉扩展名）。
SMARTICLE_CONFIG 本身只能指一个文件 —— config.py 在 import 时执行一次，
一个进程就是一次实验；"依次跑多个"这件事发生在驱动器这一层。

条件矩阵在下面的 STRATEGIES / POPULATIONS 里改；跑出来的目录结构是

    <out_root>/<condition>/            <- 每个条件一组实验
        config_snapshot.json           <- 这次实际用的全部参数
        trial_0000/ trial_0001/ ...
    <out_root>/_conditions/<name>.py   <- 该条件的覆盖文件(可复现)
    <out_root>/summary.csv             <- 所有条件的 trial 汇总拼在一起

跑完直接接分析：
    python batch_group_plots.py "<out_root>/*" --max-dist 65 --summary groups.csv
"""

import argparse
import glob
import itertools
import os
import pprint
import subprocess
import sys
import time

# =============================================================================
# 条件矩阵 —— 改这里
# =============================================================================

# 每种 N 对应的初始条件文件（IC 文件里的机器人数必须等于 N，run_trial 会检查）
INIT_FILES = {
    17:  "init_conditions/init_conditions_200_p_N17.json",
    50:  "init_conditions/init_conditions_200_p_N50.json",
    100: "init_conditions/init_conditions_200_p.json",
}

POPULATIONS = [17, 50, 100]

# 每个策略给出它自己那份 config 覆盖。名字会进目录名，别带空格。
#
# "baseline" 关掉运行时控制，全场同一条 COMMAND_ARRAY —— 对照组。
# 其余几个都开 gait_control:strategy，区别只在 STRATEGY_SPEC。
_MAX_DIST = 65.0        # 与实机一致；换 N 时想按体长缩放就在下面算
_LEAVE, _SMALL, _MID, _EDGE = 462, 862, -851, -426

STRATEGIES = {
    "baseline": {
        "ENABLE_RUNTIME_GAIT_CONTROL": False,
        # spec 用不到，但也一并清空：否则 config_snapshot.json 里会留着
        # config.py 的默认 roles/group_rules，事后翻记录容易以为它生效了。
        "STRATEGY_SPEC": {"max_dist": _MAX_DIST, "leave_command": _LEAVE,
                          "group_rules": [], "roles": []},
    },
    "groupsize": {                      # 只按分组规模发指令
        "ENABLE_RUNTIME_GAIT_CONTROL": True,
        "RUNTIME_GAIT_CONTROLLER": "gait_control:strategy",
        "STRATEGY_SPEC": {
            "max_dist": _MAX_DIST, "period": 0.25, "leave_command": _LEAVE,
            "group_rules": [
                {"name": "small", "size_range": (3, 7),  "command": _SMALL},
                {"name": "mid",   "size_range": (8, 30), "command": _MID},
            ],
            "group_n_ticks": 6,
            "roles": [],
        },
    },
    "edge": {                           # 只按空间角色：边界 vs 内部
        "ENABLE_RUNTIME_GAIT_CONTROL": True,
        "RUNTIME_GAIT_CONTROLLER": "gait_control:strategy",
        "STRATEGY_SPEC": {
            "max_dist": _MAX_DIST, "period": 0.25, "leave_command": _LEAVE,
            "group_rules": [],
            "roles": [
                {"selector": "convex_hull", "command": _EDGE,
                 "n_frames_join": 6, "n_frames_leave": 6},
            ],
        },
    },
    "ends2": {                          # 最大团长轴两端各 2 台，盖过分组层
        "ENABLE_RUNTIME_GAIT_CONTROL": True,
        "RUNTIME_GAIT_CONTROLLER": "gait_control:strategy",
        "STRATEGY_SPEC": {
            "max_dist": _MAX_DIST, "period": 0.25, "leave_command": _LEAVE,
            "group_rules": [
                {"name": "mid", "size_range": (8, 30), "command": _MID},
            ],
            "group_n_ticks": 6,
            "roles": [
                {"selector": "group_major_ends", "command": _EDGE,
                 "n_per_end": 2, "min_group_size": 4, "override_group": True,
                 "n_frames_join": 6, "n_frames_leave": 6},
            ],
        },
    },
}

# 所有条件共用的设置
COMMON = {
    "N_TRIALS_GLOBAL": 5,
    "MAX_RUNTIME": 120.0,
    # 必须是 "random" 或 "sequential"：INIT_SELECTION="explicit" 时 main() 会用
    # len(INIT_INDICES) 覆盖掉 N_TRIALS_GLOBAL，每个条件的 trial 数就不是 5 了。
    "INIT_SELECTION": "random",
    "RECORD_VIDEO": False,       # 成组实验先别录像，太占地方
    "COVERAGE_ENABLED": True,
    "PARALLEL_WORKERS": 1,
}

OUT_ROOT = os.path.join("datafile", "compare")


# =============================================================================
# 条件生成
# =============================================================================

def build_conditions(strategies=STRATEGIES, populations=POPULATIONS,
                     common=COMMON):
    """-> [(name, overrides), ...]，strategy × N 的笛卡尔积。"""
    out = []
    for n, (sname, sover) in itertools.product(populations,
                                               strategies.items()):
        if n not in INIT_FILES:
            raise ValueError(f"N={n} 没有对应的初始条件文件；"
                             f"请在 INIT_FILES 里加一条")
        name = f"n{n}_{sname}"
        ov = dict(common)
        ov.update(sover)
        ov["N_SMARTICLES"] = n
        ov["INIT_FILE"] = INIT_FILES[n]
        ov["EXP_NAME"] = name
        out.append((name, ov))
    return out


def conditions_from_files(paths):
    """
    把一串 config 文件变成 [(name, overrides), ...]。

    用的是 config.py 自己那个加载函数，所以驱动器读到的覆盖项和 config.py
    在子进程里读到的完全一致（.py / .json、BOM、注释的处理都不会分叉）。
    """
    from config import _load_override_file      # 与 config.py 共用同一份实现

    out, seen = [], {}
    for path in paths:
        settings, meta = _load_override_file(path)
        ov = dict(settings)
        if "_snapshot" in meta:
            # config_snapshot.json：派生量由输入参数重算，别写进条件文件
            ov = {k: v for k, v in ov.items() if k not in DERIVED_KEYS}
        name = ov.get("EXP_NAME") or os.path.splitext(os.path.basename(path))[0]
        if name in seen:
            raise ValueError(
                f"条件名 {name!r} 重复：{seen[name]} 和 {path}。"
                f"给其中一个加上 EXP_NAME，否则输出会互相覆盖。")
        seen[name] = path
        ov["EXP_NAME"] = name
        ov["_source"] = os.path.abspath(path)
        out.append((name, ov))
    return out


# config.py 末尾拒绝直接覆盖的那些量；快照里有，条件文件里不该有
DERIVED_KEYS = {
    "W", "H", "SCALE", "INNER_R", "INNER_R_UNSCALED", "WALL_THICK",
    "WALL_SEGMENTS", "MAIN_LEN", "MAIN_W", "ARM_LEN", "ARM_W",
    "RING_N_SIDES", "RING_MASS", "L", "L_s", "S", "MASS_MAIN", "MASS_ARM",
}

# 逐条写死会跟着 config.py 漂移，所以直接问 config.py 要
def _derived_keys():
    try:
        import config
        return set(getattr(config, "_DERIVED", DERIVED_KEYS))
    except Exception:
        return DERIVED_KEYS


def minimal_from_snapshot(snapshot_path, keep_all=False):
    """
    把一次实验的 config_snapshot.json 变成一份**精简、好手改**的覆盖参数。

    快照是完整记录（100 多个键，还含派生量），直接拿来当条件文件能跑，但没法读
    也没法改。这里只留下真正把这次实验和 config.py 默认值区分开的那些键：

      - 去掉派生量（会由输入参数重算）
      - 去掉和 config.py 当前默认值相同的键（keep_all=True 可保留）
      - COMMAND_ARRAY 如果全场一致，压成简写字符串 "a-462"

    剩下的通常只有十来行，改一改就是下一个条件。
    """
    from config import _load_override_file

    settings, meta = _load_override_file(snapshot_path)
    if "_snapshot" not in meta:
        print(f"[warn] {snapshot_path} 没有 _snapshot 标记，"
              f"可能不是 config_snapshot.json（照样按快照处理）")

    derived = _derived_keys()
    out = {k: v for k, v in settings.items() if k not in derived}

    if not keep_all:
        # 和默认值相同的键没必要留：config.py 本来就是这个值
        import subprocess
        import sys as _sys
        code = ("import json,sys,types\n"
                "sys.path.insert(0, r'%s')\n"
                "import config as c\n"
                "print('@@'+json.dumps({k: v for k, v in vars(c).items() "
                "if not k.startswith('_') and not isinstance(v, types.ModuleType) "
                "and isinstance(v, (int, float, bool, str, list, dict, type(None)))}, "
                "default=str))\n" % os.path.dirname(os.path.abspath(__file__)))
        env = dict(os.environ)
        env.pop("SMARTICLE_CONFIG", None)      # 要的是纯默认值
        env["PYTHONIOENCODING"] = "utf-8"
        r = subprocess.run([_sys.executable, "-c", code], capture_output=True,
                           text=True, encoding="utf-8", errors="replace", env=env,
                           cwd=os.path.dirname(os.path.abspath(__file__)))
        line = next((l for l in (r.stdout or "").splitlines()
                     if l.startswith("@@")), None)
        if line:
            import json as _json
            defaults = _json.loads(line[2:])
            out = {k: v for k, v in out.items()
                   if k not in defaults or defaults[k] != v}
        else:
            print("[warn] 读不到 config.py 默认值，保留全部非派生键")

    # 全场同一条指令时，压成简写；config.py 会把它展开回长度 N 的列表
    cmds = settings.get("COMMAND_ARRAY")
    if isinstance(cmds, list) and cmds and len(set(cmds)) == 1:
        out["COMMAND_ARRAY"] = f"a{cmds[0]:+d}"
    return out


def expand_config_paths(patterns):
    """展开通配符并保序去重；顺序就是实验执行顺序。"""
    out, seen = [], set()
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or ([pat] if os.path.isfile(pat) else [])
        if not hits:
            raise ValueError(f"找不到 config 文件: {pat}")
        for h in hits:
            h = os.path.normpath(h)
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def write_condition(path, name, overrides):
    """把覆盖参数写成一个可直接 SMARTICLE_CONFIG= 使用的 .py 文件。"""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # 文档字符串里的路径一律换成正斜杠：Windows 路径原样写进普通字符串会
    # 被当成转义序列（C:\Users 里的 \U 直接让生成的文件语法错误），
    # 正斜杠既安全，Windows 也照样认。
    shown = os.path.abspath(path).replace('\\', '/')
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'"""对比实验条件 {name} —— 由 experiments.py 生成。\n\n'
                f'单独重跑这个条件:\n'
                f'    SMARTICLE_CONFIG={shown} python simulation.py\n'
                f'"""\n\n')
        src = overrides.get("_source")
        if src:
            f.write(f"# 来自 {src.replace(chr(92), '/')}\n\n")
        for k in sorted(overrides):
            if k.startswith("_"):
                continue          # _source 只是驱动器的记账，不是 config 参数
            f.write(f"{k} = {pprint.pformat(overrides[k], width=76, indent=4)}\n")
    return path


# =============================================================================
# 运行
# =============================================================================

def run_condition(name, cfg_path, out_root, env=None, timeout=None):
    """新起一个解释器跑这个条件。返回 (returncode, 用时秒)。"""
    env = dict(os.environ if env is None else env)
    env["SMARTICLE_CONFIG"] = os.path.abspath(cfg_path)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    t0 = time.time()
    r = subprocess.run([sys.executable, "simulation.py"], env=env,
                       cwd=os.path.dirname(os.path.abspath(__file__)),
                       timeout=timeout)
    return r.returncode, time.time() - t0


def collect_summaries(out_root, conditions, path):
    """把各条件的 *_summary.csv 拼成一张表，加上 exp / n / strategy 三列。"""
    import pandas as pd
    rows = []
    for name, ov in conditions:
        csv = os.path.join(out_root, name + "_summary.csv")
        if not os.path.isfile(csv):
            continue
        df = pd.read_csv(csv)
        df.insert(0, "strategy",
                  name.split("_", 1)[1] if "_" in name else name)
        df.insert(0, "n_smarticles", ov.get("N_SMARTICLES", ""))
        df.insert(0, "condition", name)
        rows.append(df)
    if not rows:
        return None
    out = pd.concat(rows, ignore_index=True)
    out.to_csv(path, index=False)
    return out


def main():
    ap = argparse.ArgumentParser(
        description="成组对比实验：strategy × N_SMARTICLES",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=OUT_ROOT,
                    help=f"输出根目录 (默认 {OUT_ROOT})")
    ap.add_argument("--from-snapshot", default=None,
                    help="把一次实验的 config_snapshot.json 转成精简的条件文件")
    ap.add_argument("-o", "--output", default=None,
                    help="--from-snapshot 的输出路径 (默认打印到屏幕)")
    ap.add_argument("--keep-all", action="store_true",
                    help="--from-snapshot 时保留所有非派生键，不只留与默认值不同的")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="不用内置矩阵，依次跑这些 config 文件（可用通配符）")
    ap.add_argument("--configs-from", default=None,
                    help="从文件读 config 路径列表，每行一个（# 开头为注释）")
    ap.add_argument("--only", nargs="+", default=None,
                    help="只跑这些条件名")
    ap.add_argument("--skip-existing", action="store_true",
                    help="已经有输出目录的条件直接跳过")
    ap.add_argument("--list", action="store_true", help="列出条件后退出")
    ap.add_argument("--dry-run", action="store_true",
                    help="生成覆盖文件但不真的跑")
    ap.add_argument("--timeout", type=float, default=None,
                    help="单个条件的超时秒数")
    args = ap.parse_args()

    if args.from_snapshot:
        ov = minimal_from_snapshot(args.from_snapshot, args.keep_all)
        name = (ov.get("EXP_NAME")
                or os.path.basename(os.path.dirname(
                    os.path.abspath(args.from_snapshot))) or "from_snapshot")
        if args.output:
            write_condition(args.output, name, ov)
            print(f"{len(ov)} 个参数 -> {args.output}")
            print(f"改完直接跑: python experiments.py --configs {args.output}")
        else:
            print(f"# 由 {args.from_snapshot} 精简而来（{len(ov)} 个参数）")
            for k in sorted(ov):
                print(f"{k} = {pprint.pformat(ov[k], width=76, indent=4)}")
        return 0

    patterns = list(args.configs or [])
    if args.configs_from:
        with open(args.configs_from, encoding="utf-8-sig") as f:
            patterns += [ln.strip() for ln in f
                         if ln.strip() and not ln.lstrip().startswith("#")]
    if patterns:
        paths = expand_config_paths(patterns)
        conditions = conditions_from_files(paths)
        print(f"来自 {len(paths)} 个 config 文件")
    else:
        conditions = build_conditions()

    if args.only:
        want = set(args.only)
        unknown = want - {n for n, _ in conditions}
        if unknown:
            ap.error(f"未知条件 {sorted(unknown)}; "
                     f"可选 {[n for n, _ in conditions]}")
        conditions = [(n, o) for n, o in conditions if n in want]

    if args.list:
        print(f"{len(conditions)} 个条件:")
        for n, o in conditions:
            # 用户给的 config 文件不一定写全这些键，缺的就显示 config.py 的默认
            print(f"  {n:<20} "
                  f"N={o.get('N_SMARTICLES', '默认')!s:<6} "
                  f"trials={o.get('N_TRIALS_GLOBAL', '默认')!s:<6} "
                  f"runtime={o.get('MAX_RUNTIME', '默认')!s:<7} "
                  f"{'strategy' if o.get('ENABLE_RUNTIME_GAIT_CONTROL') else 'baseline'}"
                  f"{'  <- ' + os.path.basename(o['_source']) if o.get('_source') else ''}")
        return 0

    cfg_dir = os.path.join(args.out_root, "_conditions")
    os.makedirs(cfg_dir, exist_ok=True)
    print(f"{len(conditions)} 个条件 -> {args.out_root}")

    ok, failed, skipped = [], [], []
    t_start = time.time()
    for i, (name, ov) in enumerate(conditions, 1):
        ov = dict(ov)
        ov["OUT_ROOT"] = os.path.abspath(args.out_root)
        cfg_path = write_condition(os.path.join(cfg_dir, name + ".py"), name, ov)

        done = os.path.join(args.out_root, name)
        if args.skip_existing and os.path.isdir(done):
            print(f"[{i}/{len(conditions)}] {name}: 已存在，跳过")
            skipped.append(name)
            continue
        if args.dry_run:
            print(f"[{i}/{len(conditions)}] {name}: 覆盖文件已写 {cfg_path}")
            continue

        print(f"\n[{i}/{len(conditions)}] {name}  "
              f"(N={ov['N_SMARTICLES']}, {ov['N_TRIALS_GLOBAL']} trials)")
        try:
            rc, secs = run_condition(name, cfg_path, args.out_root,
                                     timeout=args.timeout)
        except subprocess.TimeoutExpired:
            print(f"  [TIMEOUT] {name}")
            failed.append(name)
            continue
        if rc == 0:
            print(f"  完成，用时 {secs/60:.1f} 分钟")
            ok.append(name)
        else:
            print(f"  [FAIL] 退出码 {rc}")
            failed.append(name)

    if args.dry_run:
        print(f"\n覆盖文件都写在 {cfg_dir}")
        return 0

    print(f"\n成功 {len(ok)}/{len(conditions)}，失败 {len(failed)}，"
          f"跳过 {len(skipped)}，总用时 {(time.time()-t_start)/60:.1f} 分钟")
    if failed:
        print(f"失败的条件: {failed}")

    try:
        df = collect_summaries(args.out_root, conditions,
                               os.path.join(args.out_root, "summary.csv"))
        if df is not None:
            print(f"汇总: {os.path.join(args.out_root, 'summary.csv')} "
                  f"({len(df)} 行)")
            cols = [c for c in ("condition", "n_smarticles", "strategy",
                                "k_steady", "final_rg") if c in df.columns]
            if cols:
                print(df.groupby(["condition"])[cols[3:]].mean().to_string())
    except Exception as e:
        print(f"[WARN] 汇总失败: {type(e).__name__}: {e}")

    print(f"\n接着画分组图:\n"
          f"    python batch_group_plots.py \"{args.out_root}/*\" "
          f"--max-dist 65 --summary {args.out_root}/groups.csv")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

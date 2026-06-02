"""
naming.py  ─  根据仿真参数自动生成输出文件名

编码规则（对应下位机 C 代码）:
    三位整数 XYZ:
        百位 X (1-8): initial phase → [pi/4, pi/2, pi*3/4, pi, pi*5/4, pi*3/2, pi*7/4, 2*pi]
        十位 Y (1-6): amplitude     → [pi/12, pi/6, pi/4, pi/3, pi*5/12, pi/2]
        个位 Z (1-9): frequency     → [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5]
        0 表示"未在表中匹配"或"各机器人不一致"
"""

import math

# ── 与 C 代码完全对应的字典 ──────────────────────────────────────────────────
_INITIAL_DICT = [
    math.pi/4, math.pi/2, math.pi*3/4, math.pi,
    math.pi*5/4, math.pi*3/2, math.pi*7/4, 2*math.pi,
]

_AMPLI_DICT = [
    math.pi/12, math.pi/6, math.pi/4,
    math.pi/3,  math.pi*5/12, math.pi/2,
]

_FREQ_DICT = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5]

_TOL = 1e-4   # 浮点比较容差


def _match(value: float, table: list) -> int:
    """在 table 中查找与 value 最接近的项，返回 1-based 索引；未找到返回 0。"""
    for idx, v in enumerate(table):
        if abs(value - v) < _TOL:
            return idx + 1
    return 0


def _encode_phase(phase_rad: float) -> int:
    """将弧度值编码为百位数字 (1-8)，匹配失败返回 0。"""
    return _match(phase_rad, _INITIAL_DICT)


def _encode_omega(omega_rad_per_s: float) -> int:
    """将 omega（rad/s）转换为频率（Hz）后查表，返回 1-based 索引，匹配失败返回 0。"""
    freq = omega_rad_per_s / (2 * math.pi)
    return _match(freq, _FREQ_DICT)


def _encode_ampli(ampli_deg: float) -> int:
    """将振幅（度）转换为弧度后查 _AMPLI_DICT，返回 1-based 索引，匹配失败返回 0。"""
    return _match(math.radians(ampli_deg), _AMPLI_DICT)


def _all_same(values: list) -> bool:
    """判断列表中所有值是否相同（浮点容差比较）。"""
    if not values:
        return True
    return all(abs(v - values[0]) < _TOL for v in values)


def generate_trial_name(
    n_robots: int,
    init_phases: list,          # [(j1_ph, j2_ph), ...]，长度 = n_robots，单位弧度
    omega: tuple,               # (omega1, omega2)，单位 rad/s，全局适用
    amplitude: tuple,           # (A_deg1, A_deg2)，单位度，全局适用
    prefix: str = "trial",
) -> str:
    """
    根据机器人数量、初始相位、频率和振幅生成文件名字符串。

    参数:
        n_robots    - 机器人数量
        init_phases - 每个机器人的 (joint1_phase, joint2_phase) 元组列表，单位弧度
        omega       - (omega1, omega2)，两个关节的角频率，单位 rad/s
        amplitude   - (A_deg1, A_deg2)，两个关节的振幅，单位度
        prefix      - 文件名前缀，默认 "trial"

    返回:
        命名字符串，例如: "trial_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6"
        其中 J1p5  = 关节1初相第5档（pi*5/4）
             W1f2  = 关节1频率第2档（1.0 Hz）
             A1a6  = 关节1振幅第6档（pi/2 = 90°）
             *0    = 未在表中匹配

    示例:
        init_phases = [(math.pi*5/4, math.pi*5/4)] * 17
        omega       = (2*math.pi, 2*math.pi)    # 1 Hz
        amplitude   = (90.0, 90.0)              # pi/2
        → "trial_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6"
    """
    assert len(init_phases) == n_robots, \
        f"init_phases 长度 {len(init_phases)} 与 n_robots {n_robots} 不符"

    j1_phases = [ph[0] for ph in init_phases]
    j2_phases = [ph[1] for ph in init_phases]

    j1_code = _encode_phase(j1_phases[0]) if _all_same(j1_phases) else 0
    j2_code = _encode_phase(j2_phases[0]) if _all_same(j2_phases) else 0

    w1_code = _encode_omega(omega[0])
    w2_code = _encode_omega(omega[1])

    a1_code = _encode_ampli(amplitude[0])
    a2_code = _encode_ampli(amplitude[1])

    return (f"{prefix}_N{n_robots}"
            f"_J1p{j1_code}_J2p{j2_code}"
            f"_W1f{w1_code}_W2f{w2_code}"
            f"_A1a{a1_code}_A2a{a2_code}")


# =============================================================================
# 快速测试
# =============================================================================
if __name__ == "__main__":
    N = 17

    phases_uniform = [(math.pi*5/4, math.pi*5/4) for _ in range(N)]
    phases_mixed   = [(math.pi/2 if i % 2 == 0 else math.pi, math.pi) for i in range(N)]

    omega_same = (2 * math.pi, 2 * math.pi)    # 两关节均 1 Hz
    omega_diff = (2 * math.pi, 4 * math.pi)    # 关节1=1Hz, 关节2=2Hz
    ampli_same = (90.0, 90.0)                   # 两关节均 pi/2（第6档）
    ampli_diff = (30.0, 60.0)                   # 关节1=pi/6（第2档），关节2=pi/3（第4档）

    print(generate_trial_name(N, phases_uniform, omega_same, ampli_same))
    # → trial_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6

    print(generate_trial_name(N, phases_mixed, omega_diff, ampli_diff))
    # → trial_N17_J1p0_J2p4_W1f2_W2f4_A1a2_A2a4

    print(generate_trial_name(N, phases_uniform, omega_same, ampli_same, prefix="sim"))
    # → sim_N17_J1p5_J2p5_W1f2_W2f2_A1a6_A2a6

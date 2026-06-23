"""
bodies.py  ─  Per-individual physical body specifications (heterogeneous robots).

This module turns the declarative data in config.py
(ENABLE_HETEROGENEOUS_BODIES / BODY_TYPES / BODY_ASSIGNMENT) into concrete
keyword arguments for Smarticle3Link, applying the SAME SCALE pipeline and
clamps that config.py uses for the homogeneous global geometry.

Design contract
---------------
* When ENABLE_HETEROGENEOUS_BODIES is False, resolve_body_params(i) returns
  values byte-for-byte identical to the global config geometry/mass, so the
  homogeneous experiment is completely unchanged.
* Geometry overrides are given in BASE (unscaled) units to stay consistent with
  the BASE_* -> SCALE -> clamp pipeline. Mass overrides are absolute (scaled).
* If a mass is not overridden it is auto-derived from the area ratio relative to
  the homogeneous default (bigger limb -> proportionally heavier).
"""

import config as cfg


# ── Scaling + clamp helpers (mirror config.py exactly) ────────────────────────
def _scaled_len(base):    return max(18, int(base * cfg.SCALE))   # main_len / arm_len
def _scaled_main_w(base): return max(10, int(base * cfg.SCALE))   # main_w
def _scaled_arm_w(base):  return max(3,  int(base * cfg.SCALE))   # arm_w


# Homogeneous baseline = the BASE_* values that produce the global geometry.
_DEFAULT_BASE = {
    "main_len_base": cfg.BASE_MAIN_LEN,
    "main_w_base":   cfg.BASE_MAIN_W,
    "arm_len_base":  cfg.BASE_ARM_LEN,
    "arm_w_base":    cfg.BASE_ARM_W,
}

_GEOM_KEYS = set(_DEFAULT_BASE.keys())
_MASS_KEYS = {"mass_main", "mass_arm"}

def gait_for_smarticle(sm) -> int:
    """根据机器人的实际几何属性返回对应的 COMMAND int。"""
    if not getattr(cfg, "ENABLE_HETEROGENEOUS_BODIES", False):
        return None  # 让调用方用 COMMAND_ARRAY[i]

    arm_len = sm.arm_len  # 从实例读，和 IC 文件加载还是新 spawn 无关

    # 和 BODY_TYPES 里的阈值对应，而不是查 BODY_ASSIGNMENT
    if arm_len >= _scaled_len(cfg.BASE_ARM_LEN) * 1.3:   # long_arm
        return cfg.GAIT_BY_TYPE.get("long_arm", None)
    elif arm_len <= _scaled_len(cfg.BASE_ARM_LEN) * 0.75: # short_arm
        return cfg.GAIT_BY_TYPE.get("short_arm", None)
    else:
        return cfg.GAIT_BY_TYPE.get("default", None)

def _body_type_for(i: int) -> dict:
    """Return the raw override dict assigned to robot index i (validated)."""
    if not getattr(cfg, "ENABLE_HETEROGENEOUS_BODIES", False):
        return {}

    assignment = cfg.BODY_ASSIGNMENT
    if len(assignment) != cfg.N_SMARTICLES:
        raise ValueError(
            f"BODY_ASSIGNMENT has length {len(assignment)} but N_SMARTICLES="
            f"{cfg.N_SMARTICLES}. They must match.")

    type_name = assignment[i]
    if type_name not in cfg.BODY_TYPES:
        raise KeyError(
            f"BODY_ASSIGNMENT[{i}]={type_name!r} is not a key of BODY_TYPES "
            f"({sorted(cfg.BODY_TYPES)}).")

    override = dict(cfg.BODY_TYPES[type_name])
    unknown = set(override) - _GEOM_KEYS - _MASS_KEYS
    if unknown:
        raise KeyError(
            f"BODY_TYPES[{type_name!r}] has unknown keys {sorted(unknown)}; "
            f"allowed: {sorted(_GEOM_KEYS | _MASS_KEYS)}.")
    return override


def resolve_body_params(i: int) -> dict:
    """
    Resolve robot index i to concrete Smarticle3Link kwargs:
    {main_len, main_w, arm_len, arm_w, mass_main, mass_arm}.
    """
    override = _body_type_for(i)

    base = dict(_DEFAULT_BASE)
    base.update({k: v for k, v in override.items() if k in _GEOM_KEYS})

    main_len = _scaled_len(base["main_len_base"])
    main_w   = _scaled_main_w(base["main_w_base"])
    arm_len  = _scaled_len(base["arm_len_base"])
    arm_w    = _scaled_arm_w(base["arm_w_base"])

    # Mass: auto-derive from area ratio vs. homogeneous default unless overridden.
    ref_main_area = max(1.0, cfg.MAIN_LEN * cfg.MAIN_W)
    ref_arm_area  = max(1.0, cfg.ARM_LEN  * cfg.ARM_W)
    mass_main = override.get(
        "mass_main", cfg.MASS_MAIN * (main_len * main_w) / ref_main_area)
    mass_arm = override.get(
        "mass_arm",  cfg.MASS_ARM  * (arm_len * arm_w)  / ref_arm_area)

    return {
        "main_len": main_len, "main_w": main_w,
        "arm_len":  arm_len,  "arm_w":  arm_w,
        "mass_main": float(mass_main), "mass_arm": float(mass_arm),
    }


def body_span_from_geom(main_len: float, arm_len: float) -> float:
    """Half-span L_s = (main_len + 2*arm_len)/2 used by the coupling model."""
    return (main_len + 2.0 * arm_len) / 2.0


def body_params_from_dict(d: dict) -> dict:
    """
    Build Smarticle3Link geometry kwargs from a persisted body dict (e.g. loaded
    from an initial-conditions JSON). Missing keys fall back to the global
    homogeneous defaults so old IC files without a 'body' block still work.
    """
    return {
        "main_len": int(d.get("main_len", cfg.MAIN_LEN)),
        "main_w":   int(d.get("main_w",   cfg.MAIN_W)),
        "arm_len":  int(d.get("arm_len",  cfg.ARM_LEN)),
        "arm_w":    int(d.get("arm_w",    cfg.ARM_W)),
        "mass_main": float(d.get("mass_main", cfg.MASS_MAIN)),
        "mass_arm":  float(d.get("mass_arm",  cfg.MASS_ARM)),
    }

def arm_colors(sm):
    if sm.arm_len >= 100:        # long_arm
        return (255, 140, 0, 255), (255, 200, 100, 255)   # 橙系
    elif sm.arm_len <= 50:       # short_arm
        return (100, 200, 100, 255), (180, 230, 180, 255)  # 绿系
    else:                        # default
        return (255, 100, 100, 255), (100, 100, 255, 255)  # 原红/蓝

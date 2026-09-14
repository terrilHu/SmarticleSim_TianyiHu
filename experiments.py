"""
experiments.py  ─  Grouped comparison experiments: multiple strategies × multiple N_SMARTICLES.

Each condition = one set of config overrides. The driver writes it out as an
override file, then **spawns a new Python process** to run simulation.main(),
and finally aggregates the summary across all conditions.

Why a new process is required
----------------------------
smarticle.py / spawn.py / analysis.py all use `from config import MAIN_LEN, ...`
style value binding, and these quantities are all derived from N_SMARTICLES.
Changing N_SMARTICLES within a process only changes that name in the config
module — the already-bound geometry constants won't follow, so you'd end up
running a 100-robot arena with N=17 robot dimensions. A fresh interpreter per
condition is the only reliable approach (it also conveniently isolates the
global RNG that config.py consumes at import time).

Usage
----
    python experiments.py --list            # just list which conditions would run
    python experiments.py --dry-run         # generate override files without running
    python experiments.py                   # run everything
    python experiments.py --only n17_edge n50_edge
    python experiments.py --out-root datafile/compare0901

You can also skip the built-in matrix and directly supply a list of pre-written
config files, running each one in turn:

    python experiments.py --configs conditions/a.py conditions/b.py
    python experiments.py --configs "conditions/*.py"      # wildcard
    python experiments.py --configs-from list.txt          # one path per line

The condition name defaults to EXP_NAME from the file, falling back to the
filename (without extension) if absent. SMARTICLE_CONFIG itself can only point
to a single file — config.py executes once at import time, so one process is
one experiment; "running several in sequence" happens at the driver level.

The condition matrix is edited below in STRATEGIES / POPULATIONS; the
resulting directory structure is

    <out_root>/<condition>/            <- one set of experiments per condition
        config_snapshot.json           <- all parameters actually used this run
        trial_0000/ trial_0001/ ...
    <out_root>/_conditions/<name>.py   <- override file for this condition (reproducible)
    <out_root>/summary.csv             <- trial summaries from all conditions concatenated

Feed straight into analysis afterward:
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
# Condition matrix — edit here
# =============================================================================

# Initial condition file for each N (robot count in the IC file must equal N; run_trial checks this)
INIT_FILES = {
    17:  "init_conditions/init_conditions_200_p_N17.json",
    50:  "init_conditions/init_conditions_200_p_N50.json",
    100: "init_conditions/init_conditions_200_p.json",
}

POPULATIONS = [17, 50, 100]

# Each strategy provides its own set of config overrides. The name goes into
# the directory name, so no spaces.
#
# "baseline" disables runtime control, using the same COMMAND_ARRAY across the
# whole arena — the control group. The rest all enable gait_control:strategy,
# differing only in STRATEGY_SPEC.
_MAX_DIST = 65.0        # matches the real robot; scale by body length below if N changes
_LEAVE, _SMALL, _MID, _EDGE = 462, 862, -851, -426

STRATEGIES = {
    "baseline": {
        "ENABLE_RUNTIME_GAIT_CONTROL": False,
        # spec isn't used here, but clear it too: otherwise config_snapshot.json
        # would retain config.py's default roles/group_rules, which could be
        # mistaken for being active when reviewing records later.
        "STRATEGY_SPEC": {"max_dist": _MAX_DIST, "leave_command": _LEAVE,
                          "group_rules": [], "roles": []},
    },
    "groupsize": {                      # commands based only on group size
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
    "edge": {                           # only by spatial role: boundary vs interior
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
    "ends2": {                          # 2 robots at each end of the largest cluster's major axis, overrides the group layer
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

# Settings shared by all conditions
COMMON = {
    "N_TRIALS_GLOBAL": 5,
    "MAX_RUNTIME": 120.0,
    # Must be "random" or "sequential": with INIT_SELECTION="explicit", main()
    # would override N_TRIALS_GLOBAL with len(INIT_INDICES), so each condition
    # would no longer have 5 trials.
    "INIT_SELECTION": "random",
    "RECORD_VIDEO": False,       # skip video recording for grouped experiments, too much disk space
    "COVERAGE_ENABLED": True,
    "PARALLEL_WORKERS": 1,
}

OUT_ROOT = os.path.join("datafile", "compare")


# =============================================================================
# Condition generation
# =============================================================================

def build_conditions(strategies=STRATEGIES, populations=POPULATIONS,
                     common=COMMON):
    """-> [(name, overrides), ...], the Cartesian product of strategy × N."""
    out = []
    for n, (sname, sover) in itertools.product(populations,
                                               strategies.items()):
        if n not in INIT_FILES:
            raise ValueError(f"N={n} has no corresponding initial condition file; "
                             f"please add one in INIT_FILES")
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
    Turn a list of config files into [(name, overrides), ...].

    Uses config.py's own loader function, so the overrides read by the driver
    are exactly consistent with what config.py reads in the subprocess (no
    divergence in .py / .json, BOM, or comment handling).
    """
    from config import _load_override_file      # shares the same implementation as config.py

    out, seen = [], {}
    for path in paths:
        settings, meta = _load_override_file(path)
        ov = dict(settings)
        if "_snapshot" in meta:
            # config_snapshot.json: derived quantities are recomputed from
            # input parameters, so don't write them into the condition file
            ov = {k: v for k, v in ov.items() if k not in DERIVED_KEYS}
        name = ov.get("EXP_NAME") or os.path.splitext(os.path.basename(path))[0]
        if name in seen:
            raise ValueError(
                f"Duplicate condition name {name!r}: {seen[name]} and {path}. "
                f"Add an EXP_NAME to one of them, or the outputs will overwrite each other.")
        seen[name] = path
        ov["EXP_NAME"] = name
        ov["_source"] = os.path.abspath(path)
        out.append((name, ov))
    return out


# Quantities config.py refuses to let be overridden directly at the end of the
# file; present in the snapshot, but shouldn't be in a condition file
DERIVED_KEYS = {
    "W", "H", "SCALE", "INNER_R", "INNER_R_UNSCALED", "WALL_THICK",
    "WALL_SEGMENTS", "MAIN_LEN", "MAIN_W", "ARM_LEN", "ARM_W",
    "RING_N_SIDES", "RING_MASS", "L", "L_s", "S", "MASS_MAIN", "MASS_ARM",
}

# Hardcoding this list would drift out of sync with config.py, so ask config.py directly
def _derived_keys():
    try:
        import config
        return set(getattr(config, "_DERIVED", DERIVED_KEYS))
    except Exception:
        return DERIVED_KEYS


def minimal_from_snapshot(snapshot_path, keep_all=False):
    """
    Turn one experiment's config_snapshot.json into a **minimal, easy-to-hand-edit**
    set of overrides.

    The snapshot is a complete record (100+ keys, including derived quantities);
    it can be used directly as a condition file, but it's hard to read or edit.
    This keeps only the keys that actually distinguish this experiment from
    config.py's defaults:

      - drops derived quantities (recomputed from input parameters)
      - drops keys that match config.py's current defaults (keep_all=True keeps them)
      - if COMMAND_ARRAY is uniform across the arena, compresses it to a short string like "a-462"

    What's left is usually only about a dozen lines — tweak it to make the next condition.
    """
    from config import _load_override_file

    settings, meta = _load_override_file(snapshot_path)
    if "_snapshot" not in meta:
        print(f"[warn] {snapshot_path} has no _snapshot marker, "
              f"might not be a config_snapshot.json (treating it as a snapshot anyway)")

    derived = _derived_keys()
    out = {k: v for k, v in settings.items() if k not in derived}

    if not keep_all:
        # No need to keep keys matching the default: that's already config.py's value
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
        env.pop("SMARTICLE_CONFIG", None)      # we want the plain defaults
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
            print("[warn] could not read config.py defaults, keeping all non-derived keys")

    # When the whole arena shares one command, compress it to shorthand; config.py expands it back to a length-N list
    cmds = settings.get("COMMAND_ARRAY")
    if isinstance(cmds, list) and cmds and len(set(cmds)) == 1:
        out["COMMAND_ARRAY"] = f"a{cmds[0]:+d}"
    return out


def expand_config_paths(patterns):
    """Expand wildcards and deduplicate while preserving order; order is experiment execution order."""
    out, seen = [], set()
    for pat in patterns:
        hits = sorted(glob.glob(pat)) or ([pat] if os.path.isfile(pat) else [])
        if not hits:
            raise ValueError(f"config file not found: {pat}")
        for h in hits:
            h = os.path.normpath(h)
            if h not in seen:
                seen.add(h)
                out.append(h)
    return out


def write_condition(path, name, overrides):
    """Write the overrides out as a .py file usable directly with SMARTICLE_CONFIG=."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Always convert paths in the docstring to forward slashes: a Windows path
    # written as-is into a plain string gets treated as an escape sequence
    # (the \U in C:\Users directly causes a syntax error in the generated file);
    # forward slashes are safe and Windows still accepts them.
    shown = os.path.abspath(path).replace('\\', '/')
    with open(path, "w", encoding="utf-8") as f:
        f.write(f'"""Comparison experiment condition {name} -- generated by experiments.py.\n\n'
                f'To rerun this condition alone:\n'
                f'    SMARTICLE_CONFIG={shown} python simulation.py\n'
                f'"""\n\n')
        src = overrides.get("_source")
        if src:
            f.write(f"# From {src.replace(chr(92), '/')}\n\n")
        for k in sorted(overrides):
            if k.startswith("_"):
                continue          # _source is just driver bookkeeping, not a config parameter
            f.write(f"{k} = {pprint.pformat(overrides[k], width=76, indent=4)}\n")
    return path


# =============================================================================
# Run
# =============================================================================

def run_condition(name, cfg_path, out_root, env=None, timeout=None):
    """Spawn a fresh interpreter to run this condition. Returns (returncode, elapsed seconds)."""
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
    """Concatenate each condition's *_summary.csv into one table, adding exp / n / strategy columns."""
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
        description="Grouped comparison experiments: strategy x N_SMARTICLES",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root", default=OUT_ROOT,
                    help=f"Output root directory (default {OUT_ROOT})")
    ap.add_argument("--from-snapshot", default=None,
                    help="Turn one experiment's config_snapshot.json into a minimal condition file")
    ap.add_argument("-o", "--output", default=None,
                    help="Output path for --from-snapshot (default: print to screen)")
    ap.add_argument("--keep-all", action="store_true",
                    help="With --from-snapshot, keep all non-derived keys instead of only those differing from defaults")
    ap.add_argument("--configs", nargs="+", default=None,
                    help="Skip the built-in matrix and run these config files in turn (wildcards allowed)")
    ap.add_argument("--configs-from", default=None,
                    help="Read a list of config paths from a file, one per line (# prefix for comments)")
    ap.add_argument("--only", nargs="+", default=None,
                    help="Only run these condition names")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip conditions that already have an output directory")
    ap.add_argument("--list", action="store_true", help="List conditions and exit")
    ap.add_argument("--dry-run", action="store_true",
                    help="Generate override files without actually running")
    ap.add_argument("--timeout", type=float, default=None,
                    help="Timeout in seconds for a single condition")
    args = ap.parse_args()

    if args.from_snapshot:
        ov = minimal_from_snapshot(args.from_snapshot, args.keep_all)
        name = (ov.get("EXP_NAME")
                or os.path.basename(os.path.dirname(
                    os.path.abspath(args.from_snapshot))) or "from_snapshot")
        if args.output:
            write_condition(args.output, name, ov)
            print(f"{len(ov)} parameters -> {args.output}")
            print(f"Edit then run directly: python experiments.py --configs {args.output}")
        else:
            print(f"# Reduced from {args.from_snapshot} ({len(ov)} parameters)")
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
        print(f"From {len(paths)} config files")
    else:
        conditions = build_conditions()

    if args.only:
        want = set(args.only)
        unknown = want - {n for n, _ in conditions}
        if unknown:
            ap.error(f"Unknown condition(s) {sorted(unknown)}; "
                     f"available: {[n for n, _ in conditions]}")
        conditions = [(n, o) for n, o in conditions if n in want]

    if args.list:
        print(f"{len(conditions)} conditions:")
        for n, o in conditions:
            # User-supplied config files may not set all these keys; show config.py's default for missing ones
            print(f"  {n:<20} "
                  f"N={o.get('N_SMARTICLES', 'default')!s:<6} "
                  f"trials={o.get('N_TRIALS_GLOBAL', 'default')!s:<6} "
                  f"runtime={o.get('MAX_RUNTIME', 'default')!s:<7} "
                  f"{'strategy' if o.get('ENABLE_RUNTIME_GAIT_CONTROL') else 'baseline'}"
                  f"{'  <- ' + os.path.basename(o['_source']) if o.get('_source') else ''}")
        return 0

    cfg_dir = os.path.join(args.out_root, "_conditions")
    os.makedirs(cfg_dir, exist_ok=True)
    print(f"{len(conditions)} conditions -> {args.out_root}")

    ok, failed, skipped = [], [], []
    t_start = time.time()
    for i, (name, ov) in enumerate(conditions, 1):
        ov = dict(ov)
        ov["OUT_ROOT"] = os.path.abspath(args.out_root)
        cfg_path = write_condition(os.path.join(cfg_dir, name + ".py"), name, ov)

        done = os.path.join(args.out_root, name)
        if args.skip_existing and os.path.isdir(done):
            print(f"[{i}/{len(conditions)}] {name}: already exists, skipping")
            skipped.append(name)
            continue
        if args.dry_run:
            print(f"[{i}/{len(conditions)}] {name}: override file written {cfg_path}")
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
            print(f"  done, took {secs/60:.1f} min")
            ok.append(name)
        else:
            print(f"  [FAIL] exit code {rc}")
            failed.append(name)

    if args.dry_run:
        print(f"\nAll override files written to {cfg_dir}")
        return 0

    print(f"\nSucceeded {len(ok)}/{len(conditions)}, failed {len(failed)}, "
          f"skipped {len(skipped)}, total time {(time.time()-t_start)/60:.1f} min")
    if failed:
        print(f"Failed conditions: {failed}")

    try:
        df = collect_summaries(args.out_root, conditions,
                               os.path.join(args.out_root, "summary.csv"))
        if df is not None:
            print(f"Summary: {os.path.join(args.out_root, 'summary.csv')} "
                  f"({len(df)} rows)")
            cols = [c for c in ("condition", "n_smarticles", "strategy",
                                "k_steady", "final_rg") if c in df.columns]
            if cols:
                print(df.groupby(["condition"])[cols[3:]].mean().to_string())
    except Exception as e:
        print(f"[WARN] summary failed: {type(e).__name__}: {e}")

    print(f"\nNext, plot grouped figures:\n"
          f"    python batch_group_plots.py \"{args.out_root}/*\" "
          f"--max-dist 65 --summary {args.out_root}/groups.csv")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())

"""
branch.py  ─  Resume a recorded trial from any recorded moment with a changed
strategy ("real-time control", done after the fact).

    python branch.py <trial or branch folder> --at 42.3  --spec new_spec.json
    python branch.py <folder> --at-step 2538 --cmd "a-862" --controller none
    python branch.py <folder> --at 120 --controller gait_control:example_stagger
    python branch.py <folder> --at 120 --spec '{"max_dist": 115.5, ...}' --until 300
    python branch.py <folder> --at 100 --cmd-a-fraction 0.12 --cmd-a -852

The folder is a trial folder from a run made with CHECKPOINT_ENABLED = True
(it has a checkpoints/ subfolder), or a branch folder made by this script, so
branches can be branched again. Then stitch.py joins a chain of segments into
one ordinary trial folder.

What happens
============
1. The parent's config is loaded (its experiment's config_snapshot.json, or
   the branch's own), so the replay runs under exactly the recorded settings.
2. The branch point is Step s on the POS_ALL axis (--at is seconds on the same
   recorded axis render_trial and the gait log use, i.e. s = round(at * fps)).
   If s lies before the given branch's own segment, the ancestor that recorded
   it is used instead.
3. The last checkpoint at or before Step s is loaded and replayed, with the
   recorded controller, to the end of frame s -- after Step s is recorded and
   the controller has polled. Checkpoints are the reload points of the
   recording (see checkpoint.py), so this replay is exact; by default the
   replayed frame is checked against the parent's POS_ALL.
4. The intervention is applied:
     --spec      new STRATEGY_SPEC (file .json / .py, or inline JSON); implies
                 --controller gait_control:strategy unless one is given. The
                 running strategy's hysteresis is inherited (see
                 LayeredStrategy.inherit_state); --fresh-state starts it cold.
     --cmd       one-shot command text in the gait debug syntax ("a-862",
                 "0..4-851", ...), queued at t*. A strategy that is still
                 running will override it at its next tick, so this usually
                 goes with --controller none.
     --controller  "module:function", or "none" to switch runtime control off
                 (robots keep their current commands).
     --cmd-a-fraction F
                 change the CMD_A / CMD_B mix: round(N * F) robots on CMD_A,
                 the same count config.py gives at start-up. Incremental by
                 default -- robots already on A stay on A and the difference
                 is drawn at random -- so 0.12 then 0.36 keeps the first 60 of
                 500 and adds 120. --redraw draws the whole set afresh;
                 --cmd-a / --cmd-b override the parent's commands; --seed
                 fixes the draw (otherwise random, recorded in branch.json
                 with the switched ids).
   Applied in that order (fraction, then --cmd), so --cmd wins on overlap.
   Switches take effect when RUNTIME_GAIT_SWITCH_MODE schedules them, as in
   any run. The X (phase) digit of a mid-run command is ignored: the phase
   comes from the switch mode (zero_phase: continuity at the switch;
   phase_locked: the robot's home clock), so a robot switched to -852 mid-run
   is not phase-for-phase the same as one that started on -852.
5. A checkpoint is written at the branch point, and the branch runs on to
   --until with checkpoints every CHECKPOINT_EVERY, as the parent did.

Output: <root trial>/branches/bNNN[_from_bMMM]_t<at>[_name]/ -- all branches
of a trial side by side, whatever they were branched from (branch.json holds
the parent), so paths stay short however long a chain gets -- with
    branch.json             parent, branch point, intervention, what was inherited
    config_snapshot.json    the branch's effective config
    trial_XXXX_POS_ALL.npz  Steps s+1 .. end (npz: see pos_all_io.py)
    trial_XXXX_render.npz, _gait_log.csv, _coverage.csv, _alignment.csv
    checkpoints/

No video is recorded; render the stitched result with analysis/render_trial.py.
"""

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


# =============================================================================
# Folder bookkeeping (no simulator imports: config must not load yet)
# =============================================================================

def read_branch_json(folder):
    p = os.path.join(folder, "branch.json")
    if os.path.isfile(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def parent_of(folder):
    """
    The folder a branch was made from, from its branch.json: the relative
    path first (survives moving the whole tree), the absolute one as fallback.
    """
    folder = os.path.abspath(folder)
    bj = read_branch_json(folder)
    if bj is None:
        return None
    p = os.path.normpath(os.path.join(folder, bj["parent"]))
    if not os.path.isdir(p) and os.path.isdir(bj.get("parent_abs", "")):
        p = bj["parent_abs"]
    return p


def root_of(folder):
    """The recorded trial a chain of branches starts from."""
    f = os.path.abspath(folder)
    while read_branch_json(f) is not None:
        f = parent_of(f)
    return f


def first_step(folder) -> int:
    """First POS_ALL Step this folder recorded itself (0 for a root trial)."""
    bj = read_branch_json(folder)
    return 0 if bj is None else int(bj["first_step"])


def snapshot_path(folder):
    """A branch carries its own config; a root trial uses its experiment's."""
    here = os.path.abspath(folder)
    for d in ((here,) if read_branch_json(here) else (os.path.dirname(here),)):
        p = os.path.join(d, "config_snapshot.json")
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"no config_snapshot.json for {folder}")


def load_spec(text):
    if text is None:
        return None
    s = text.strip()
    if s.startswith("{"):
        return json.loads(s)
    if s.lower().endswith(".json"):
        with open(s, encoding="utf-8-sig") as f:
            d = json.load(f)
        return d.get("STRATEGY_SPEC", d)
    ns = {}
    with open(s, encoding="utf-8-sig") as f:
        exec(compile(f.read(), s, "exec"), ns)
    if "STRATEGY_SPEC" not in ns:
        raise ValueError(f"{s} does not define STRATEGY_SPEC")
    return ns["STRATEGY_SPEC"]


def git_commit():
    try:
        return subprocess.run(["git", "-C", HERE, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


# =============================================================================
# Main
# =============================================================================

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="trial folder (with checkpoints/) or branch folder")
    at = ap.add_mutually_exclusive_group(required=True)
    at.add_argument("--at", type=float, help="branch time, seconds on the recorded axis")
    at.add_argument("--at-step", type=int, help="branch after this POS_ALL Step")
    ap.add_argument("--spec", help="new STRATEGY_SPEC: .json / .py file or inline JSON")
    ap.add_argument("--cmd", help='one-shot command text, e.g. "a-862" or "0..4-851"')
    ap.add_argument("--controller", help='"module:function", or "none"')
    ap.add_argument("--cmd-a-fraction", type=float, default=None,
                    help="new CMD_A_FRACTION: switch robots so round(N * F) run CMD_A "
                         "(incremental by default, see --redraw)")
    ap.add_argument("--cmd-a", type=int, default=None,
                    help="command A for --cmd-a-fraction (default: the parent's CMD_A)")
    ap.add_argument("--cmd-b", type=int, default=None,
                    help="command B for --cmd-a-fraction (default: the parent's CMD_B)")
    ap.add_argument("--redraw", action="store_true",
                    help="with --cmd-a-fraction: draw the whole A set afresh instead of "
                         "keeping the robots already on A")
    ap.add_argument("--seed", type=int, default=None,
                    help="seed for the --cmd-a-fraction draw (default: random, recorded)")
    ap.add_argument("--fresh-state", action="store_true",
                    help="do not inherit the running strategy's hysteresis")
    ap.add_argument("--until", type=float, default=None,
                    help="end time, seconds on the recorded axis (default: the parent's end)")
    ap.add_argument("--name", default="", help="suffix for the branch folder name")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip checking the replayed frame against the parent's POS_ALL")
    args = ap.parse_args(argv)
    if (args.spec is None and args.cmd is None and args.controller is None
            and args.cmd_a_fraction is None):
        ap.error("nothing to change: give --spec, --cmd, --controller "
                 "and/or --cmd-a-fraction")
    if args.cmd_a_fraction is not None and not 0.0 <= args.cmd_a_fraction <= 1.0:
        ap.error("--cmd-a-fraction must be between 0 and 1")
    if args.cmd_a_fraction is None and (args.cmd_a is not None or args.cmd_b is not None
                                        or args.redraw or args.seed is not None):
        ap.error("--cmd-a / --cmd-b / --redraw / --seed only apply with --cmd-a-fraction")

    folder = os.path.abspath(args.folder)
    if not os.path.isdir(os.path.join(folder, "checkpoints")):
        sys.exit(f"{folder} has no checkpoints/ -- was it run with CHECKPOINT_ENABLED = True?")

    # ── Config of the run being branched ─────────────────────────────────────
    # Must be in place before config is imported; the step math needs fps,
    # which comes from that config, so the branch point is resolved after.
    snap = snapshot_path(folder)
    os.environ["SMARTICLE_CONFIG"] = snap
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    import numpy as np
    import config as cfg
    import simulation
    import checkpoint as ckpt
    import pos_all_io
    from strategy import build_strategy

    import hashlib
    _snap, _ = simulation._config_snapshot_dict()
    digest_now = hashlib.sha256(
        json.dumps(_snap, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    if not getattr(cfg, "CHECKPOINT_ENABLED", False):
        sys.exit(f"{snap} has CHECKPOINT_ENABLED off; nothing to branch from.")
    fps = float(cfg.RENDER_FPS_HEADLESS)        # one recorded frame per render frame
    s = int(args.at_step) if args.at_step is not None else int(round(args.at * fps))
    if s < 0:
        sys.exit("the branch point must be at or after the first recorded frame")

    # The moment may belong to an ancestor's segment: branch from the folder
    # that recorded it (its config may differ, so reload config if it moves).
    src = folder
    while s < first_step(src):
        src = parent_of(src)
    if src != folder:
        print(f"[branch] Step {s} precedes this branch's own segment; "
              f"branching from the ancestor {src}", flush=True)
        if snapshot_path(src) != snap:
            # Different config: rerun in a fresh interpreter with the ancestor.
            argv2 = [sys.executable, os.path.abspath(__file__), src] + \
                    [a for a in (argv or sys.argv[1:]) if os.path.abspath(a) != folder]
            sys.exit(subprocess.call(argv2))

    metas = ckpt.list_checkpoints(src)
    if not metas:
        sys.exit(f"no checkpoints in {src}")
    last_step = max(m["n_rec"] for m in metas) - 1
    if s > last_step:
        sys.exit(f"Step {s} is past the end of {src} (last recorded Step {last_step})")
    # Latest checkpoint whose state is at or before the end of frame s. No
    # checkpoint (= reload point) lies between it and s, so replay is exact.
    base = max((m for m in metas if m["n_rec"] <= s + 1),
               key=lambda m: (m["n_rec"], m["t"]))
    if base.get("config_digest") and base["config_digest"] != digest_now:
        print(f"[branch] [WARN] config digest {digest_now} differs from the "
              f"checkpoint's {base['config_digest']}: replay may not be exact. "
              f"Verification below will tell.")

    # ── Check the intervention before spending time on the replay ────────────
    # Nothing is written until all of this has passed, so a typo never leaves
    # a half-made branch folder behind.
    spec = load_spec(args.spec)
    if spec is not None:
        try:
            build_strategy(spec)
        except Exception as e:
            sys.exit(f"--spec is not a valid STRATEGY_SPEC: {type(e).__name__}: {e}")
    if args.cmd:
        from gait import parse_request
        try:
            parse_request(args.cmd, int(base["n"]))
        except ValueError as e:
            sys.exit(f"--cmd {args.cmd!r}: {e}")
    if args.cmd_a_fraction is not None:
        cmd_a, cmd_b = fraction_commands(cfg, snap, args.cmd_a, args.cmd_b)

    # The branch folder is created only once the replay and the intervention
    # have succeeded (see below); until then this is just its future path.
    # Every branch of a trial lives flat in <root trial>/branches/, however
    # deep the chain: lineage is in branch.json (and the "_from_bNNN" in the
    # name). Nesting each branch inside its parent made paths grow ~30 chars
    # per level and run into Windows' 260-character limit at the third level.
    bdir_root = os.path.join(root_of(src), "branches")
    bdir = os.path.join(bdir_root, "_pending")

    # Branches never record video (render the stitched trial instead) and
    # always write POS_ALL as npz.
    simulation.RECORD_VIDEO = False
    cfg.RECORD_VIDEO        = False
    cfg.SNAPSHOT_DIVISIONS  = 0
    cfg.POS_ALL_FORMAT      = "npz"

    # ── Replay to the branch point ───────────────────────────────────────────
    w0 = time.time()
    print(f"[branch] {os.path.basename(base['path'])}  (Step {base['n_rec'] - 1}, "
          f"t={base['t']:.3f} s) -> replay to Step {s}")
    runner = simulation.TrialRunner.from_checkpoint(base["path"], out_dir=bdir)
    replay_frames = s + 1 - runner.n_recorded()
    if replay_frames > 0:
        runner._run_frames(stop_rec=s + 1)
    if runner.n_recorded() != s + 1:
        sys.exit(f"replay ended at Step {runner.n_recorded() - 1}, expected {s}")
    t_star   = runner.t
    t_rec0   = runner.t_rec0 if runner.t_rec0 is not None else runner.time_hist[0]
    t_rel    = t_star - t_rec0
    print(f"[branch] at Step {s}: t = {t_star:.4f} s (recorded axis {t_rel:.4f} s), "
          f"{replay_frames} frame(s) replayed in {time.time() - w0:.1f} s")

    verified = None
    if not args.no_verify and runner.POS_ALL:
        verified = verify_frame(pos_all_io, src, s, runner.POS_ALL[-1])
        if verified is False:
            sys.exit("[branch] replayed state does not match the recording -- aborting")

    runner.truncate_records()
    first = runner.n_recorded()                          # == s + 1
    render_first = runner.render_offset

    # ── Intervention ─────────────────────────────────────────────────────────
    gc   = runner.gait_ctl
    ctx  = gc.ctx
    old  = ctx.get("strategy")
    controller = args.controller
    if spec is not None and controller is None:
        controller = "gait_control:strategy"
    report = {}

    if spec is not None:
        cfg.STRATEGY_SPEC = spec
        new = build_strategy(spec)
        if old is not None and not args.fresh_state:
            report = new.inherit_state(old)
        elif old is not None:
            # Hysteresis starts cold, but the decision log is a log: keep it.
            new.decisions     = list(old.decisions)
            new._last_decided = dict(old._last_decided)
            report = {"fresh_state": True}
        else:
            report = {"notes": ["parent ran no strategy; nothing to inherit"]}
        ctx["strategy"] = new

    if controller is not None:
        if controller.lower() == "none":
            cfg.ENABLE_RUNTIME_GAIT_CONTROL = False
        else:
            cfg.ENABLE_RUNTIME_GAIT_CONTROL = True
            cfg.RUNTIME_GAIT_CONTROLLER     = controller
        runner.runtime_gait_on, gc.callback = runner._resolve_gait_callback()
        if controller.lower() != "none" and not runner.runtime_gait_on:
            sys.exit(f"could not load controller {controller!r}")

    def queue(req):
        """
        Queue {robot_id: command} at t*. When a strategy keeps the decision
        log (the gait log is built from it), a user command is recorded there
        too, as a decision made at t*.
        """
        n_ok = sum(1 for rid, cmd in sorted(req.items())
                   if gc.request(rid, cmd, t_star))
        strat = ctx.get("strategy")
        if strat is not None and hasattr(strat, "decisions"):
            for rid, cmd in sorted(req.items()):
                strat.decisions.append((float(t_star), int(rid), int(cmd)))
                strat._last_decided[int(rid)] = int(cmd)
        return n_ok

    # Fraction first, then --cmd, so an explicit --cmd wins where they overlap
    # (a later request for a robot replaces its pending one).
    frac = None
    if args.cmd_a_fraction is not None:
        seed = args.seed if args.seed is not None else int.from_bytes(os.urandom(4), "little")
        req, frac = plan_fraction(gc.states, args.cmd_a_fraction, cmd_a, cmd_b,
                                  redraw=args.redraw, seed=seed)
        frac["queued"] = queue(req)
        cfg.CMD_A, cfg.CMD_B, cfg.CMD_A_FRACTION = cmd_a, cmd_b, args.cmd_a_fraction
        print(f"[branch] CMD_A_FRACTION -> {args.cmd_a_fraction} (A={cmd_a:+d}, "
              f"B={cmd_b:+d}, {frac['mode']}, seed {seed}): A {frac['a_before']} -> "
              f"{frac['a_after']} of {len(gc.states)}; "
              f"{len(frac['to_a'])} switched to A, {len(frac['to_b'])} to B")
        if frac["n_other"]:
            print(f"[branch] [WARN] {frac['n_other']} robot(s) run neither A nor B and "
                  f"were left as they are (use --redraw to set them to B)")

    queued = 0
    if args.cmd:
        from gait import parse_request
        queued = queue(parse_request(args.cmd, len(gc.states)))   # raises on a typo
        print(f"[branch] --cmd {args.cmd!r}: {queued} robot(s) queued")

    if (args.cmd or frac) and runner.runtime_gait_on:
        print(f"[branch] [WARN] runtime control is still on "
              f"({cfg.RUNTIME_GAIT_CONTROLLER}); it may override these commands at "
              f"its next tick. Add --controller none to hold them.")

    end_sim = (t_rec0 + args.until) if args.until is not None else float(cfg.MAX_RUNTIME)
    if end_sim <= t_star:
        sys.exit(f"--until must be after the branch point ({t_rel:.3f} s)")
    cfg.MAX_RUNTIME = simulation.MAX_RUNTIME = end_sim
    runner.running, runner.jam_time = True, None

    for line in describe(report):
        print("[branch] " + line)

    # ── Branch folder ────────────────────────────────────────────────────────
    os.makedirs(bdir_root, exist_ok=True)
    k = 1 + max([int(d[1:4]) for d in os.listdir(bdir_root)
                 if d.startswith("b") and d[1:4].isdigit()] or [0])
    lineage = (f"_from_{os.path.basename(src)[:4]}"
               if read_branch_json(src) is not None else "")
    bname = f"b{k:03d}{lineage}_t{t_rel:.2f}" + (f"_{args.name}" if args.name else "")
    bdir = os.path.join(bdir_root, bname)
    os.makedirs(bdir)
    runner.out_dir  = bdir
    runner.ckpt_dir = os.path.join(bdir, "checkpoints")
    runner.video_dir = bdir

    # ── Branch config, start checkpoint, run ─────────────────────────────────
    simulation.save_config_snapshot(bdir)
    bj = {
        "parent": os.path.relpath(src, bdir), "parent_abs": src,
        "trial_id": runner.trial_id,
        "branch_step": s, "first_step": first, "render_first": render_first,
        "t_sim": t_star, "t_rel": t_rel, "t_rec0": t_rec0, "fps": fps,
        "from_checkpoint": os.path.relpath(base["path"], bdir),
        "replayed_frames": replay_frames, "replay_verified": verified,
        "intervention": {"spec": spec, "cmd": args.cmd, "cmd_queued": queued,
                         "cmd_a_fraction": frac,
                         "controller": controller,
                         "inherit": spec is not None and not args.fresh_state,
                         "inherit_report": report},
        "end_sim": end_sim, "end_rel": end_sim - t_rec0,
        "created": time.strftime("%Y-%m-%d %H:%M:%S"), "git": git_commit(),
    }
    _write_json(os.path.join(bdir, "branch.json"), bj)

    # Checkpoint at the branch point itself, reloaded like every other, so a
    # branch of this branch can start anywhere in it.
    runner.save_checkpoint(reload=True, k=base["k"])
    w1 = time.time()
    runner.run()
    summary = runner.finalize()
    bj["summary"] = summary
    bj["wall_s"] = round(time.time() - w0, 1)
    _write_json(os.path.join(bdir, "branch.json"), bj)
    print(f"[branch] done in {time.time() - w1:.1f} s: {bdir}")
    print(f"[branch] stitch with:  python stitch.py \"{bdir}\"")
    return bdir


def fraction_commands(cfg, snap, cmd_a=None, cmd_b=None):
    """
    (CMD_A, CMD_B) for --cmd-a-fraction: given values, else the parent's.

    Snapshots written before config exposed CMD_A / CMD_B publicly do not
    record them (config would then just report its own defaults, which say
    nothing about the run), so the snapshot file itself is checked. Without a
    record, B falls back to the most common command in the parent's initial
    COMMAND_ARRAY, and A must be given.
    """
    from collections import Counter
    from gait import validate_command
    with open(snap, encoding="utf-8") as f:
        recorded = json.load(f)
    if cmd_a is None:
        if "CMD_A" not in recorded:
            sys.exit(f"{snap} does not record CMD_A; give --cmd-a")
        cmd_a = int(cfg.CMD_A)
    if cmd_b is None:
        if "CMD_B" in recorded:
            cmd_b = int(cfg.CMD_B)
        else:
            rest = Counter(c for c in cfg.COMMAND_ARRAY if c != cmd_a)
            if not rest:
                sys.exit(f"{snap} does not record CMD_B; give --cmd-b")
            cmd_b = int(rest.most_common(1)[0][0])
            print(f"[branch] CMD_B not recorded; using the parent's most common "
                  f"initial command {cmd_b:+d}")
    validate_command(cmd_a)
    validate_command(cmd_b)
    if cmd_a == cmd_b:
        sys.exit(f"CMD_A and CMD_B are both {cmd_a}")
    return int(cmd_a), int(cmd_b)


def plan_fraction(states, frac, cmd_a, cmd_b, redraw=False, seed=None):
    """
    Requests that bring round(N * frac) robots onto cmd_a -- the same count
    config.py's CMD_A_FRACTION gives at start-up. -> ({rid: cmd}, record)

    A robot's command here is the one it is heading to: a pending switch
    counts as already made.

    Incremental (default): robots already on A stay on A; if more are needed,
    they are drawn at random from the rest; if fewer, randomly drawn A robots
    go to B. So 0.12 -> 0.36 keeps the first 12% and adds 24%, which is what
    a sequence of dose steps needs. Robots on neither command are not touched.
    redraw: the A set is drawn afresh from all robots and everyone else is
    put on B.
    """
    import numpy as np
    n = len(states)
    target = int(round(n * frac))
    cur = [st.pending_cmd if st.pending_cmd is not None else st.command
           for st in states]
    on_a = [i for i in range(n) if cur[i] == cmd_a]
    rng = np.random.default_rng(seed)

    if redraw:
        a_set = {int(i) for i in rng.choice(n, size=target, replace=False)}
        want = {i: (cmd_a if i in a_set else cmd_b) for i in range(n)}
    else:
        a_set, want = set(on_a), {}
        if target > len(on_a):
            pool = [i for i in range(n) if i not in a_set]
            want = {int(i): cmd_a for i in
                    rng.choice(pool, size=target - len(on_a), replace=False)}
            a_set |= set(want)
        elif target < len(on_a):
            want = {int(i): cmd_b for i in
                    rng.choice(on_a, size=len(on_a) - target, replace=False)}
            a_set -= set(want)
    req = {i: c for i, c in want.items() if cur[i] != c}

    after = [req.get(i, cur[i]) for i in range(n)]
    record = {
        "fraction": frac, "target": target, "cmd_a": cmd_a, "cmd_b": cmd_b,
        "mode": "redraw" if redraw else "incremental", "seed": seed,
        "a_before": len(on_a), "a_after": sum(c == cmd_a for c in after),
        "to_a": sorted(i for i, c in req.items() if c == cmd_a),
        "to_b": sorted(i for i, c in req.items() if c == cmd_b),
        "n_other": sum(c not in (cmd_a, cmd_b) for c in after),
    }
    return req, record


def verify_frame(pos_all_io, folder, s, frame):
    """
    Compare a replayed frame with the parent's recorded Step s. True / False,
    or None when the parent's POS_ALL is missing. A CSV is scanned in chunks
    and the scan stops once past Step s (rows are in Step order).
    """
    import numpy as np
    path = pos_all_io.find_pos_all(folder)
    if path is None:
        print("[branch] (no parent POS_ALL to verify against)")
        return None
    if path.endswith(".npz"):
        pos, s0 = pos_all_io.load_array(path)
        rec = pos[s - s0].astype(np.float64)
    else:
        import pandas as pd
        rec = None
        for ch in pd.read_csv(path, usecols=pos_all_io.COLUMNS, chunksize=500_000):
            hit = ch[ch["Step"] == s]
            if len(hit):
                hit = hit.sort_values("Agent_ID")
                rec = hit[["X", "Y", "Theta"]].to_numpy(dtype=np.float64)
            if ch["Step"].iloc[-1] > s:
                break
        if rec is None:
            print(f"[branch] (Step {s} not found in {path})")
            return None
    got = np.round(np.asarray(frame, dtype=np.float64), 4)
    err = float(np.max(np.abs(got - np.round(rec, 4))))
    ok = err <= 2e-4                     # 4-decimal text vs float32 rounding
    print(f"[branch] replay check at Step {s}: max |diff| = {err:.2e} -> "
          f"{'OK' if ok else 'MISMATCH'}")
    return ok


def describe(report):
    if not report:
        return []
    if report.get("fresh_state"):
        return ["strategy replaced, hysteresis NOT inherited (--fresh-state)"]
    out = []
    if report.get("tick_clock"):
        out.append(f"inherited tick clock and {report.get('decisions', 0)} logged decisions")
    for new, old in report.get("group_rules", {}).items():
        out.append(f"group rule {new!r} <- streaks of {old!r}")
    for new, old in report.get("roles", {}).items():
        out.append(f"role {new!r} <- tracker state of {old!r}")
    out += [f"note: {n}" for n in report.get("notes", [])]
    return out


def _write_json(path, obj):
    from simulation import _jsonable
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()

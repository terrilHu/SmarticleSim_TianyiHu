"""
checkpoint.py  ─  Save / load a trial's world state so it can be resumed.

What a checkpoint is
====================
A pickle of everything simulation.TrialRunner carries from one frame to the
next: the pymunk space with every robot and the ring, the GaitController (its
per-robot GaitState, queued switches, and ctx -- which holds the live
LayeredStrategy with its hysteresis counters), the loop counters, the few
cached per-robot arrays, and both RNG states. Pygame objects, the video
recorder and the recorded data lists are NOT in it: those are rebuilt from
config, and the data before the checkpoint already lives in the parent run's
output files.

Why the run reloads its own checkpoints
=======================================
pymunk's pickle does not carry the solver's contact cache (warm-start
impulses). Restoring a space and stepping it is perfectly repeatable -- two
restores of the same checkpoint stay bit-identical forever -- but it is NOT the
same as never having stopped: the first step after a restore solves contacts
from a cold start. Measured (pymunk 7.1.0, 40 resting boxes, 600 steps after
the restore): 0.008 px apart, and the swarm is chaotic, so that grows.

So with CHECKPOINT_ENABLED on, TrialRunner writes each checkpoint and then
immediately continues from pickle.loads() of those same bytes. The recorded
trajectory therefore *is* the restored trajectory, and replaying from any
checkpoint reproduces the recording exactly -- which is what lets a branch be
started from any recorded frame (replay from the nearest earlier checkpoint
to that frame, then intervene). The price is that a run with checkpoints on
is a different realisation from the same run with them off; statistically
equivalent, but not comparable byte-for-byte.

File layout
===========
    8 bytes   magic  b"SMCKPT01"
    4 bytes   big-endian length of the JSON header
    header    JSON: everything needed to list / pick a checkpoint without
              decompressing it (trial id, time, frame counters, versions)
    rest      zlib-compressed pickle of the world bundle

The header can be read on its own with read_meta(), so finding "the latest
checkpoint at or before t" never has to unpickle anything.
"""

import glob
import json
import os
import pickle
import re
import struct
import sys
import time
import zlib

import pymunk

MAGIC   = b"SMCKPT01"
FORMAT  = 1
SUBDIR  = "checkpoints"

_NAME_RE = re.compile(r"ckpt_(\d+)_t([0-9.]+)\.ckpt$")


def checkpoint_dir(out_dir: str) -> str:
    return os.path.join(out_dir, SUBDIR)


def checkpoint_name(k: int, t: float) -> str:
    # Fixed-width time so a plain sort of the folder is chronological
    return f"ckpt_{k:04d}_t{t:09.3f}.ckpt"


# =============================================================================
# Serialisation
# =============================================================================

def dumps_world(bundle: dict) -> bytes:
    """
    Pickle a world bundle.

    The runtime gait callback is detached for the dump and put back after: it
    is resolved from config ("module:function") when the runner is built or
    resumed, and pickling it by reference would tie the checkpoint to whatever
    module happened to be __main__ when it was written.
    """
    gc = bundle.get("gait_ctl")
    cb = getattr(gc, "callback", None)
    if gc is not None:
        gc.callback = None
    try:
        return pickle.dumps(bundle, protocol=pickle.HIGHEST_PROTOCOL)
    finally:
        if gc is not None:
            gc.callback = cb


def loads_world(raw: bytes) -> dict:
    return pickle.loads(raw)


def write(path: str, raw: bytes, meta: dict, level: int = 6) -> int:
    """
    Write one checkpoint file atomically; returns its size in bytes.

    Written to a .tmp name and renamed, so an interrupted run never leaves a
    truncated file that looks like a valid checkpoint.
    """
    meta = dict(meta)
    meta.setdefault("format", FORMAT)
    meta.setdefault("pymunk", pymunk.version)
    meta.setdefault("python", sys.version.split()[0])
    meta.setdefault("created", time.strftime("%Y-%m-%d %H:%M:%S"))
    meta["raw_bytes"] = len(raw)
    head = json.dumps(meta, separators=(",", ":")).encode("utf-8")
    body = zlib.compress(raw, int(level))

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack(">I", len(head)))
        f.write(head)
        f.write(body)
    os.replace(tmp, path)
    return len(MAGIC) + 4 + len(head) + len(body)


def _read_header(f, path):
    if f.read(len(MAGIC)) != MAGIC:
        raise ValueError(f"{path} is not a smarticle checkpoint")
    (n,) = struct.unpack(">I", f.read(4))
    return json.loads(f.read(n).decode("utf-8"))


def read_meta(path: str) -> dict:
    with open(path, "rb") as f:
        meta = _read_header(f, path)
    meta["path"] = path
    return meta


def read(path: str, check_versions: bool = True):
    """
    Return (meta, raw pickle bytes) for one checkpoint file.

    A pymunk version mismatch is an error: the pickled space is pymunk's own
    internal format, and even where it loads, a different solver would not
    reproduce the recorded trajectory -- which is the whole point.
    """
    with open(path, "rb") as f:
        meta = _read_header(f, path)
        raw = zlib.decompress(f.read())
    meta["path"] = path
    if check_versions and meta.get("pymunk") != pymunk.version:
        raise RuntimeError(
            f"{path} was written with pymunk {meta.get('pymunk')}, this is "
            f"pymunk {pymunk.version}; replay would not be exact.")
    if meta.get("format") != FORMAT:
        raise RuntimeError(f"{path}: checkpoint format {meta.get('format')}, "
                           f"this code reads format {FORMAT}")
    return meta, raw


# =============================================================================
# Finding checkpoints
# =============================================================================

def list_checkpoints(out_dir_or_ckpt_dir: str) -> list:
    """
    All checkpoint headers under a trial folder (or its checkpoints/ folder),
    sorted by time.
    """
    d = out_dir_or_ckpt_dir
    if os.path.basename(os.path.normpath(d)) != SUBDIR:
        d = checkpoint_dir(d)
    out = [read_meta(p) for p in glob.glob(os.path.join(d, "ckpt_*.ckpt"))]
    return sorted(out, key=lambda m: (m["t"], m["k"]))


def latest_at_or_before(out_dir: str, t: float, tol: float = 1e-6) -> dict:
    """Header of the last checkpoint with time <= t (None if there is none)."""
    best = None
    for m in list_checkpoints(out_dir):
        if m["t"] <= t + tol:
            best = m
    return best

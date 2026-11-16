"""
pos_all_io.py  ─  One reader for POS_ALL in either on-disk format.

    trial_XXXX_POS_ALL.csv   Step,Agent_ID,X,Y,Theta  (4 decimals, the original)
    trial_XXXX_POS_ALL.npz   pos   float32 (frames, n, 3) -- x, y, theta
                             step0 int64  Step of the first frame

The npz form is what branches and stitched trials write (config.POS_ALL_FORMAT
= "npz"); at N=500 the CSV is ~1.1 MB per simulated second, the npz about a
fifth of that. read_pos_all() returns the same long-form DataFrame for both,
so a script that switches from pd.read_csv(path) to read_pos_all(path) needs
no other change. Values from an npz are rounded to the 4 decimals the CSV
would have carried, so the two formats feed analysis identical numbers.
"""

import glob
import os

import numpy as np

SUFFIXES = ("_POS_ALL.csv", "_POS_ALL.npz")
COLUMNS  = ["Step", "Agent_ID", "X", "Y", "Theta"]


def find_pos_all(folder):
    """The trial's POS_ALL file (CSV preferred if both exist), or None."""
    for suf in SUFFIXES:
        hits = sorted(glob.glob(os.path.join(folder, "*" + suf)))
        if hits:
            return hits[0]
    return None


def is_pos_all(path) -> bool:
    return str(path).endswith(SUFFIXES)


def sidecar(pos_all_path, suffix):
    """
    Path of a file named after the trial: sidecar(".../trial_0000_POS_ALL.npz",
    "_groups_d65.csv") -> ".../trial_0000_groups_d65.csv". Works for both
    formats, unlike the old path.replace("_POS_ALL.csv", suffix).
    """
    p = str(pos_all_path)
    for suf in SUFFIXES:
        if p.endswith(suf):
            return p[: -len(suf)] + suffix
    raise ValueError(f"not a POS_ALL path: {p}")


def write_npz(path, pos, step0=0):
    """pos: (frames, n, 3) array-like of x, y, theta."""
    pos = np.asarray(pos, dtype=np.float32)
    if pos.ndim != 3 or pos.shape[2] != 3:
        raise ValueError(f"pos must be (frames, n, 3), got {pos.shape}")
    np.savez_compressed(path, pos=pos, step0=np.int64(step0))


def load_array(path):
    """
    (pos (frames, n, 3) float32, step0) from either format. For a CSV, the
    Step column must be contiguous (it always is for files this code writes).
    """
    if str(path).endswith(".npz"):
        with np.load(path) as z:
            return z["pos"], int(z["step0"])
    import pandas as pd
    df = pd.read_csv(path, usecols=COLUMNS)
    steps = df["Step"].to_numpy()
    n = int(df["Agent_ID"].max())
    s0 = int(steps.min())
    frames = int(steps.max()) - s0 + 1
    if len(df) != frames * n:
        raise ValueError(f"{path}: {len(df)} rows is not {frames} frames x {n} robots")
    df = df.sort_values(["Step", "Agent_ID"], kind="stable")
    pos = df[["X", "Y", "Theta"]].to_numpy(dtype=np.float32).reshape(frames, n, 3)
    return pos, s0


def read_pos_all(path, usecols=None):
    """POS_ALL as the long-form DataFrame pd.read_csv gives, from either format."""
    import pandas as pd
    if not str(path).endswith(".npz"):
        return pd.read_csv(path, usecols=usecols)
    pos, s0 = load_array(path)
    frames, n, _ = pos.shape
    vals = np.round(pos.astype(np.float64), 4)
    df = pd.DataFrame({
        "Step":     np.repeat(np.arange(s0, s0 + frames, dtype=np.int64), n),
        "Agent_ID": np.tile(np.arange(1, n + 1, dtype=np.int64), frames),
        "X":        vals[:, :, 0].ravel(),
        "Y":        vals[:, :, 1].ravel(),
        "Theta":    vals[:, :, 2].ravel(),
    })
    if usecols is not None:
        df = df[[c for c in COLUMNS if c in set(usecols)]]
    return df


def n_robots(path):
    if str(path).endswith(".npz"):
        with np.load(path) as z:
            return int(z["pos"].shape[1])
    import pandas as pd
    return int(pd.read_csv(path, usecols=["Agent_ID"])["Agent_ID"].max())

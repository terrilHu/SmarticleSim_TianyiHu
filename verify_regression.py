"""
Deterministic regression + benchmark harness.

Usage:  python harness.py <workdir> <runtime_seconds> [tag]

Seeds python/numpy RNG before run_trial so that two different code versions
can be compared byte-for-byte on the produced artefacts.
"""
import hashlib
import json
import os
import random
import shutil
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[1])
os.environ["SDL_VIDEODRIVER"] = "dummy"

RUNTIME = float(sys.argv[2])
TAG = sys.argv[3] if len(sys.argv) > 3 else "run"

# Seed BEFORE importing config: config.py shuffles COMMAND_ARRAY / BODY_ASSIGNMENT
# at import time using the global RNG.
random.seed(20260728)
import numpy as np
np.random.seed(20260728)
import simulation
from analysis import actuationimpactCalculation
from spawn import load_initial_conditions

simulation.MAX_RUNTIME = RUNTIME

OUT = os.path.abspath(f"_hz_{TAG}")
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

import math
_om0 = 3.0 * 2 * math.pi
_am0 = 90.0
actuations = actuationimpactCalculation(_om0 / (2 * math.pi), _om0 / (2 * math.pi),
                                        math.radians(_am0), math.radians(_am0))

ALL_INIT = load_initial_conditions("init_conditions/init_conditions_200_H.json")

random.seed(4242)
np.random.seed(4242)

t0 = time.perf_counter()
res = simulation.run_trial(trial_id=0, seed=0, preview=False,
                           video_dir=OUT, out_dir=OUT,
                           actuations=actuations, ALL_INIT=ALL_INIT,
                           use_preset=True, init_idx=0)
elapsed = time.perf_counter() - t0

digests = {}
for fn in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, fn)
    if not os.path.isfile(p):
        continue
    if fn.endswith(".npz"):          # zip container stores mtimes -> hash contents
        z = np.load(p)
        h = hashlib.sha256()
        for k in sorted(z.files):
            h.update(k.encode())
            h.update(np.ascontiguousarray(z[k]).tobytes())
        digests[fn] = h.hexdigest()[:16]
    else:
        digests[fn] = hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]

report = {"tag": TAG, "elapsed_s": round(elapsed, 3),
          "result": {k: float(v) for k, v in res.items()},
          "files": digests}
print(json.dumps(report, indent=2))
with open(os.path.abspath(f"_hz_{TAG}.json"), "w") as f:
    json.dump(report, f, indent=2)

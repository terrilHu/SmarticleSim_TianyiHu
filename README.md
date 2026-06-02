# SmarticleSim

Physics simulation of Smarticle robots (3-link planar walkers) confined in a circular ring, built with [pymunk](http://www.pymunk.org/).

## Files

| File | Description |
|------|-------------|
| `config.py` | All experiment parameters — edit here to change robot count, motion frequency/amplitude, physics settings, etc. |
| `smarticle.py` | `Smarticle3Link` robot class and physics helpers (ring wall, box bodies) |
| `spawn.py` | Robot placement logic |
| `GetSpawnPositions.py` | Pre-generate and save initial conditions to JSON |
| `simulation.py` | Main simulation loop; runs trials and saves data |
| `naming.py` | Auto-generates experiment names from config parameters |
| `pos_all_alignment.py` | Computes per-frame alignment time series from `POS_ALL.csv` via Voronoi adjacency |
| `analysis.py` / `LoadingDataAnalysiswitntrending.py` | Data analysis utilities — work in progress |

## Typical workflow

```bash
# 1. Edit config.py as needed

# 2. Pre-generate initial conditions (optional)
python GetSpawnPositions.py

# 3. Run simulation
python simulation.py
```

## Output directories (not tracked by git)

| Directory | Contents |
|-----------|----------|
| `datafile/` | Per-trial CSVs and NPZ files |
| `videos/` | Recorded simulation videos |
| `init_conditions/` | Pre-generated initial condition JSONs |
| `spawn_images/` | Spawn layout preview images |

Each trial writes the following files under `datafile/<exp_name>/trial_XXXX/`:

| File | Contents |
|------|----------|
| `*_POS_ALL.csv` | Per-frame position and orientation of all robots. Columns: `Step, Agent_ID, X, Y, Theta` |
| `*_alignment.csv` | Global alignment time series computed from `POS_ALL`. Columns: `Step, mean_alignment` (0–1) |
| `*_GRs_ORDs_Mdist.csv` | Per-frame order parameters. Columns: `ORDs_abs, ORDs, ORD_diffs, ORD_diffs_abs, ROT_ORDS, Mdist` |
| `*_impacts.csv` | Per-frame interaction impact score for each robot. Columns: `time, s1, s2, ...` |
| `*_GRs.npz` | Radial distribution function g(r) across frames. Key: `gr`, shape `(frames, r_bins)` |
| `config_snapshot.json` | Copy of all `config.py` parameters at the time the experiment was run |

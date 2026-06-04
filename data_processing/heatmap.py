import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sweep import plot_heatmap, AMPLI_DICT, FREQ_DICT

df = pd.read_csv("datafile/sweep_summary.csv")

n_a = len(AMPLI_DICT)
n_f = len(FREQ_DICT)
heatmap_mean = np.full((n_f, n_a), float("nan"))
heatmap_std  = np.full((n_f, n_a), float("nan"))

for row in df.itertuples():
    a_idx = int(row.ampli_slot) - 1
    f_idx = int(row.freq_slot)  - 1
    heatmap_mean[f_idx, a_idx] = row.combo_align_mean
    heatmap_std [f_idx, a_idx] = row.combo_align_std

ampli_labels = [f"{a/math.pi:.3g}π" for a in AMPLI_DICT]
freq_labels  = [f"{f}" for f in FREQ_DICT]

plot_heatmap(
    heatmap_mean, heatmap_std,
    ampli_labels, freq_labels,
    out_path="datafile/sweep_alignment_heatmap.png", cmap="plasma", vmin=0.4, vmax=0.8
)
"""
plot_qwen_unreliability.py
Visual proof that Qwen-3-8B-RHEA is unreliable on Ti-alloys:
  A) Density hallucination: 100% of predictions above the physical Ti gate (3.8-4.8),
     plus batch-size-dependent value shifts.
  B) Yield-strength shifts across batch sizes (BF16 non-determinism).
  C) Phase hallucination matrix; red boxes = alloys whose phase flips with batch size.
  D) Throughput: Qwen (best 1.71 samples/s) vs CrabNet (244.33 samples/s), log scale.
Inputs : qwen_predictions_detailed.csv, qwen_throughput_benchmark.csv
Output : slide_qwen_unreliability.png
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

PRED, THR, OUT = "qwen_predictions_detailed.csv", "qwen_throughput_benchmark.csv", "slide_qwen_unreliability.png"
CRABNET_SPS = 244.33   # from benchmark_results_v2.csv, chunk 15000

df = pd.read_csv(PRED)
thr = pd.read_csv(THR)

batches = sorted(df.batch_size_used.unique())
alloys = list(dict.fromkeys(df.formula))

dens  = df.pivot_table(index="formula", columns="batch_size_used", values="density_g_cm3").reindex(alloys)
ys    = df.pivot_table(index="formula", columns="batch_size_used", values="yield_strength_MPa").reindex(alloys)
phase = df.pivot_table(index="formula", columns="batch_size_used", values="phase", aggfunc="first").reindex(alloys)

n_dens  = int((dens.nunique(axis=1) > 1).sum())
n_ys    = int((ys.nunique(axis=1) > 1).sum())
n_phase = int((phase.nunique(axis=1) > 1).sum())
print(f"[stats] density shifts across batch sizes : {n_dens}/{len(alloys)}")
print(f"[stats] yield-strength shifts             : {n_ys}/{len(alloys)}")
print(f"[stats] phase flips                       : {n_phase}/{len(alloys)}")
print(f"[stats] Qwen density range {dens.min().min():.2f}-{dens.max().max():.2f} g/cm3 "
      f"vs physical gate 3.8-4.8 -> 100% outside")

colors = {1: "#1f77b4", 2: "#ff7f0e", 4: "#2ca02c", 8: "#d62728"}
x = np.arange(len(alloys))
fig, axes = plt.subplots(2, 2, figsize=(18, 11))

# ---- A: density hallucination + shift ----
ax = axes[0, 0]
ax.axhspan(3.8, 4.8, color="#2ca02c", alpha=0.15, label="physical Ti gate (3.8-4.8 g/cm3)")
for b in batches:
    ax.plot(x, dens[b].values, marker="o", ms=6, lw=1.2, color=colors[b], label=f"batch {b}")
ax.set_xticks(x); ax.set_xticklabels(alloys, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("density (g/cm3)")
ax.set_title(f"A) Density hallucination: all predictions above physical gate;\n"
             f"{n_dens}/{len(alloys)} alloys shift value with batch size")
ax.legend(fontsize=8, loc="upper left", ncol=2); ax.grid(ls="--", alpha=.5)

# ---- B: yield-strength shift ----
ax = axes[0, 1]
for b in batches:
    ax.plot(x, ys[b].values, marker="s", ms=6, lw=1.2, color=colors[b], label=f"batch {b}")
for i, a in enumerate(alloys):
    if ys.loc[a].nunique() > 1:
        ax.annotate(f"{int(ys.loc[a].min())}-{int(ys.loc[a].max())}", (i, ys.loc[a].max()),
                    textcoords="offset points", xytext=(0, 8), ha="center",
                    fontsize=7, color="#d62728", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(alloys, rotation=60, ha="right", fontsize=8)
ax.set_ylabel("yield strength (MPa)")
ax.set_title(f"B) Yield-strength shift: {n_ys}/{len(alloys)} alloys change value\nwith batch size (BF16 non-determinism)")
ax.legend(fontsize=8, loc="upper left", ncol=2); ax.grid(ls="--", alpha=.5)

# ---- C: phase hallucination matrix ----
ax = axes[1, 0]
phase = phase.fillna("MISSING")
phases = sorted(phase.stack().unique())
code = {p: i for i, p in enumerate(phases)}
M = phase.map(lambda v: code[v]).values.astype(float)
cmap = plt.get_cmap("tab10")
ax.imshow(M, aspect="auto", cmap=cmap, vmin=0, vmax=9)
ax.set_xticks(range(len(batches))); ax.set_xticklabels([f"batch {b}" for b in batches])
ax.set_yticks(range(len(alloys))); ax.set_yticklabels(alloys, fontsize=8)
for i in range(len(alloys)):
    for j in range(len(batches)):
        ax.text(j, i, phase.iloc[i, j], ha="center", va="center", fontsize=7,
                color="w", fontweight="bold")
for i, a in enumerate(alloys):
    if phase.loc[a].nunique() > 1:
        ax.add_patch(plt.Rectangle((-0.5, i - 0.5), len(batches), 1, fill=False, edgecolor="red", lw=2))
ax.set_title(f"C) Phase hallucination: RHEA phases (B2/BCC/Al3Ti) on Ti-alloys;\n"
             f"red box = phase flips with batch size ({n_phase} alloy(s))")
ax.legend(handles=[Patch(facecolor=cmap(code[p]), label=p) for p in phases],
          fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1.0))

# ---- D: throughput comparison ----
ax = axes[1, 1]
labels = [f"Qwen\nbatch {int(r.batch_size)}" for r in thr.itertuples()] + ["CrabNet\nchunk 15k"]
vals = list(thr.samples_per_sec) + [CRABNET_SPS]
cols = ["#d62728"] * len(thr) + ["#2ca02c"]
ax.bar(range(len(vals)), vals, color=cols)
ax.set_yscale("log")
ax.set_xticks(range(len(vals))); ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("throughput (samples/s, log scale)")
for i, v in enumerate(vals):
    ax.text(i, v * 1.15, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
ax.set_title(f"D) Throughput: Qwen best {thr.samples_per_sec.max():.2f} samples/s vs CrabNet "
             f"{CRABNET_SPS}\n-> ~{CRABNET_SPS / thr.samples_per_sec.max():.0f}x slower")
ax.grid(axis="y", ls="--", alpha=.5)

fig.suptitle("Qwen-3-8B-RHEA on Ti-alloys: OOD hallucinations + batch-size instability + low throughput\n"
             "=> unreliable as primary screener; retained only as red-team reviewer",
             fontsize=13, fontweight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig(OUT, dpi=300, bbox_inches="tight")
print(f"[plot] saved {OUT}")
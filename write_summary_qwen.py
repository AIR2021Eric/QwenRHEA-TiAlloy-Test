import json, pandas as pd
thr  = pd.read_csv("qwen_throughput_benchmark.csv")
pred = pd.read_csv("qwen_predictions_detailed.csv")
d = pred.pivot_table(index="formula", columns="batch_size_used", values="density_g_cm3")
y = pred.pivot_table(index="formula", columns="batch_size_used", values="yield_strength_MPa")
p = pred.pivot_table(index="formula", columns="batch_size_used", values="phase", aggfunc="first")
summary = {
  "model": "qwen-3-8b-rhea-property-predictor",
  "role": "red-team reviewer (NOT primary screener)",
  "hardware": "RTX 2000 Ada 8 GB, 4-bit NF4 (bitsandbytes), torch cu121",
  "throughput_samples_per_s": {str(int(r.batch_size)): r.samples_per_sec for r in thr.itertuples()},
  "crabnet_reference_samples_per_s": 244.33,
  "reliability": {
    "deterministic_across_batch_sizes": False,
    "alloys_with_density_shift": int((d.nunique(axis=1) > 1).sum()),
    "alloys_with_ys_shift": int((y.nunique(axis=1) > 1).sum()),
    "alloys_with_phase_flip": int((p.nunique(axis=1) > 1).sum())
  },
  "ood_evidence": [
    "density 5.45-7.85 g/cm3 vs physical Ti gate 3.8-4.8",
    "RHEA phases (B2/BCC/Al3Ti) predicted on Ti alloys",
    "RHEA_EXTRAPOLATION flag on 100% of Ti-rich inputs"
  ],
  "verdict": "unreliable for Ti-alloy property values; OOD stress-test only"
}
json.dump(summary, open(r"..\comparison\model_results_qwenrhea.json", "w"), indent=2)
print("summary written to ..\\comparison\\model_results_qwenrhea.json")

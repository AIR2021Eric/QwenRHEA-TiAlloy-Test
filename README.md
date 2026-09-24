# Qwen-3-8B-RHEA Ti-Alloy OOD Stress-Test (Red-Team Project)
Role: red-team reviewer ONLY - NOT a primary screener.
Trained on refractory high-entropy alloys (RHEAs); severely OOD for Ti alloys.

## Setup
python -m venv venv ; activate ; pip install -r requirements_qwen.txt
(torch cu121 line may need: pip install torch --index-url https://download.pytorch.org/whl/cu121)

## Scripts
test_qwen_rhea.py            - single/batch inference + OOD guardrails
benchmark_qwen_throughput.py - batch sweep 1/2/4/8 + CSV exports
plot_qwen_unreliability.py   - 4-panel unreliability figure

## Key findings (2026-09-22)
Throughput 0.291 / 0.626 / 1.110 / 1.711 samples/s at batch 1/2/4/8 (no OOM at 8)
vs CrabNet 244.33 samples/s -> ~143x slower at best batch
Non-deterministic across batch sizes (BF16): density / yield / phase shifts
Density hallucinations 5.45-7.85 g/cm3 vs physical Ti gate 3.8-4.8
RHEA phases (B2 / BCC / Al3Ti) predicted on Ti alloys; guardrails flag 100% of Ti-rich inputs
Verdict: unreliable for Ti property values; retained as OOD stress-test only

## Outputs
qwen_throughput_benchmark.csv | qwen_predictions_detailed.csv |
slide_qwen_unreliability.png | results_archive_qwen/

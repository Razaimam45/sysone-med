# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| models/qwen_direct_lora | non-AR | pubmedqa | qwen_direct_lora | 500 | 0.6860 | 0.9453 | 0.2664 | 2.0991 | 0.5494 | 101.2102 | 142.5608 | 9.5845 | 2026-10-04T09:11:26.827601+00:00 |
| models/qwen_direct_lora | non-AR | med_qa | qwen_direct_lora | 1268 | 0.4543 | 0.7199 | 0.2692 | 1.7465 | 0.7934 | 117.9669 | 194.7093 | 7.7750 | 2026-10-04T09:11:26.827601+00:00 |
| models/qwen_direct_lora | non-AR | medmcqa | qwen_direct_lora | 2000 | 0.5640 | 0.7899 | 0.2259 | 1.4211 | 0.6439 | 41.6244 | 54.8950 | 22.6940 | 2026-10-04T09:11:26.827601+00:00 |
| models/qwen_direct_lora | non-AR | vqarad | qwen_direct_lora | 109 | 0.6606 | 0.9065 | 0.2460 | 1.1740 | 0.5381 | 488.6693 | 882.3743 | 1.8447 | 2026-10-04T09:11:26.827601+00:00 |
| models/qwen_direct_lora | non-AR | slake | qwen_direct_lora | 355 | 0.8507 | 0.9593 | 0.1129 | 0.4653 | 0.2357 | 195.5611 | 721.0925 | 3.1881 | 2026-10-04T09:11:26.827601+00:00 |
| models/qwen_direct_lora | non-AR | path_vqa | qwen_direct_lora | 2000 | 0.8630 | 0.9442 | 0.0820 | 0.4265 | 0.2187 | 292.9104 | 302.9375 | 3.6205 | 2026-10-04T09:11:26.827601+00:00 |
| models/qwen_direct_lora | non-AR | pmc_vqa | qwen_direct_lora | 1980 | 0.5732 | 0.8061 | 0.2329 | 1.3791 | 0.6449 | 116.6454 | 394.7029 | 6.3379 | 2026-10-04T09:11:26.827601+00:00 |

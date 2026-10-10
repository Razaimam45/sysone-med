# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | AR | pubmedqa | conventional_llm | 500 | 0.6800 | 0.7686 | 0.0935 | 0.8520 | 0.4600 | 61.7555 | 65.6096 | 17.0834 | 2026-10-04T07:44:58.676161+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | med_qa | conventional_llm | 1268 | 0.4306 | 0.6095 | 0.1796 | 1.5088 | 0.7436 | 52.2982 | 65.8616 | 18.8410 | 2026-10-04T07:44:58.676161+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | medmcqa | conventional_llm | 2000 | 0.5390 | 0.6220 | 0.0857 | 1.1355 | 0.6012 | 41.3716 | 42.8771 | 24.0155 | 2026-10-04T07:44:58.676161+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | vqarad | conventional_llm | 109 | 0.6514 | 0.6864 | 0.0814 | 0.5665 | 0.3881 | 273.3077 | 469.9739 | 3.3491 | 2026-10-04T07:44:58.676161+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | slake | conventional_llm | 355 | 0.7127 | 0.7928 | 0.0815 | 0.5161 | 0.3567 | 133.5128 | 409.9733 | 5.1120 | 2026-10-04T07:44:58.676161+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | path_vqa | conventional_llm | 2000 | 0.6190 | 0.7209 | 0.1019 | 0.6671 | 0.4660 | 173.5157 | 196.7946 | 5.9205 | 2026-10-04T07:44:58.676161+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | pmc_vqa | conventional_llm | 1980 | 0.4884 | 0.6255 | 0.1438 | 1.2264 | 0.6502 | 74.5616 | 132.4522 | 12.0522 | 2026-10-04T07:44:58.676161+00:00 |

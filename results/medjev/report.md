# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| junma/MedJev-Qwen3.5-0.8B | non-AR | pubmedqa | medjev | 500 | 0.6380 | 0.6296 | 0.0512 | 0.8766 | 0.5075 | 90.0417 | 124.9721 | 10.4338 | 2026-09-29T08:00:39.706421+00:00 |
| junma/MedJev-Qwen3.5-0.8B | non-AR | med_qa | medjev | 1268 | 0.2831 | 0.4939 | 0.2107 | 1.8293 | 0.8707 | 85.4842 | 106.6173 | 11.4268 | 2026-09-29T08:00:39.706421+00:00 |
| junma/MedJev-Qwen3.5-0.8B | non-AR | medmcqa | medjev | 2000 | 0.3590 | 0.4751 | 0.1176 | 1.3731 | 0.7416 | 74.9425 | 85.4061 | 13.0334 | 2026-09-29T08:00:39.706421+00:00 |

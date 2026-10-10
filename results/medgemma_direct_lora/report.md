# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| models/medgemma-direct | non-AR | pubmedqa | medgemma_direct_lora | 500 | 0.7260 | 0.9751 | 0.2502 | 3.9976 | 0.5177 | 209.9445 | 382.0523 | 4.4230 | 2026-09-29T08:58:17.450301+00:00 |
| models/medgemma-direct | non-AR | med_qa | medgemma_direct_lora | 1268 | 0.5331 | 0.8912 | 0.3582 | 2.6757 | 0.7978 | 238.8310 | 455.9898 | 3.8949 | 2026-09-29T08:58:17.450301+00:00 |
| models/medgemma-direct | non-AR | medmcqa | medgemma_direct_lora | 2000 | 0.6870 | 0.7948 | 0.1094 | 0.8749 | 0.4361 | 76.5872 | 124.6461 | 12.0080 | 2026-09-29T08:58:17.450301+00:00 |
| models/medgemma-direct | non-AR | vqarad | medgemma_direct_lora | 109 | 0.6972 | 0.9837 | 0.2911 | 2.9565 | 0.5608 | 376.2629 | 567.9898 | 2.4136 | 2026-09-29T08:58:17.450301+00:00 |
| models/medgemma-direct | non-AR | slake | medgemma_direct_lora | 355 | 0.9070 | 0.9873 | 0.0847 | 0.7283 | 0.1707 | 385.2262 | 565.4702 | 2.3856 | 2026-09-29T08:58:17.450301+00:00 |
| models/medgemma-direct | non-AR | path_vqa | medgemma_direct_lora | 2000 | 0.9005 | 0.9740 | 0.0774 | 0.4535 | 0.1737 | 381.2131 | 567.8637 | 2.4246 | 2026-09-29T08:58:17.450301+00:00 |
| models/medgemma-direct | non-AR | pmc_vqa | medgemma_direct_lora | 1980 | 0.5722 | 0.8517 | 0.2807 | 1.6961 | 0.6888 | 714.3352 | 1097.3382 | 1.2764 | 2026-09-29T08:58:17.450301+00:00 |

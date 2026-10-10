# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| models/medgemma-direct-headonly | non-AR | pubmedqa | medgemma_direct_headonly | 500 | 0.6560 | 0.6669 | 0.0456 | 0.8534 | 0.4879 | 198.3309 | 355.3878 | 4.7775 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | med_qa | medgemma_direct_headonly | 1268 | 0.5150 | 0.5389 | 0.0514 | 1.2393 | 0.6272 | 220.9585 | 415.9543 | 4.2606 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | medmcqa | medgemma_direct_headonly | 2000 | 0.6545 | 0.6596 | 0.0260 | 0.8595 | 0.4572 | 69.8819 | 112.0936 | 13.5725 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | vqarad | medgemma_direct_headonly | 109 | 0.7615 | 0.7604 | 0.1481 | 0.5647 | 0.3696 | 372.6245 | 551.6858 | 2.4827 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | slake | medgemma_direct_headonly | 355 | 0.8676 | 0.9085 | 0.0628 | 0.2953 | 0.1913 | 376.1003 | 549.4974 | 2.4598 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | path_vqa | medgemma_direct_headonly | 2000 | 0.8630 | 0.8856 | 0.0233 | 0.3054 | 0.1911 | 377.7203 | 551.9313 | 2.4495 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | pmc_vqa | medgemma_direct_headonly | 1980 | 0.5707 | 0.5845 | 0.0384 | 1.0334 | 0.5583 | 719.3076 | 1066.5057 | 1.2635 | 2026-09-29T08:06:05.167296+00:00 |
| models/medgemma-direct-headonly | non-AR | pubmedqa | medgemma_direct_headonly | 5 | 0.4000 | 0.7484 | 0.5549 | 1.5836 | 0.8324 | 142.0874 | 391.5600 | 4.9331 | 2026-10-03T18:58:07.288561+00:00 |

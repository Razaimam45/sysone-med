# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| convaiinnovations/laya | non-AR | pubmedqa | non_ar_decision | 500 | 0.5660 | 0.6802 | 0.1225 | 1.0658 | 0.6060 | 19.6436 | 47.6017 | 36.6836 | 2026-09-29T09:53:38.851313+00:00 |
| convaiinnovations/laya | non-AR | pubmedqa | non_ar_decision_b16 | 500 | 0.5660 | 0.6802 | 0.1225 | 1.0658 | 0.6060 | 18.7117 | 47.3692 | 39.6668 | 2026-09-29T09:53:38.851313+00:00 |
| convaiinnovations/laya | non-AR | med_qa | non_ar_decision | 1268 | 0.2114 | 0.3505 | 0.1407 | 1.6984 | 0.8391 | 18.6422 | 43.2124 | 39.4199 | 2026-09-29T09:53:38.851313+00:00 |
| convaiinnovations/laya | non-AR | med_qa | non_ar_decision_b16 | 1268 | 0.2114 | 0.3505 | 0.1407 | 1.6984 | 0.8391 | 18.4024 | 42.9913 | 41.9861 | 2026-09-29T09:53:38.851313+00:00 |
| convaiinnovations/laya | non-AR | medmcqa | non_ar_decision | 2000 | 0.2765 | 0.4026 | 0.1265 | 1.4700 | 0.7891 | 18.0977 | 36.9496 | 44.2575 | 2026-09-29T09:53:38.851313+00:00 |
| convaiinnovations/laya | non-AR | medmcqa | non_ar_decision_b16 | 2000 | 0.2765 | 0.4026 | 0.1265 | 1.4700 | 0.7891 | 18.1690 | 36.7071 | 43.2228 | 2026-09-29T09:53:38.851313+00:00 |

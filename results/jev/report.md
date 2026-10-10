# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| typesafe/jev-1.13 | non-AR | pubmedqa | jev_api | 500 | 0.7880 | 0.9059 | 0.1204 | 1.4255 | 0.3389 | 586.9450 | 673.9125 | 1.6662 | 2026-09-29T10:45:31.966981+00:00 |
| typesafe/jev-1.13 | non-AR | med_qa | jev_api | 1000 | 0.8610 | 0.8637 | 0.0283 | 0.3632 | 0.1944 | 583.9713 | 679.0164 | 1.6834 | 2026-09-29T10:45:31.966981+00:00 |
| typesafe/jev-1.13 | non-AR | medmcqa | jev_api | 1000 | 0.8500 | 0.8702 | 0.0267 | 0.4296 | 0.2060 | 586.5069 | 684.2442 | 1.6798 | 2026-09-29T10:45:31.966981+00:00 |

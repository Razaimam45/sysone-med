# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-med | non-AR | med_qa | laya_med_calibrated | 1268 | 0.3044 | 0.2266 | 0.0778 | 1.5741 | 0.7858 | 31.8186 | 51.1879 | 31.2425 | None |
| laya-med | non-AR | medmcqa | laya_med_calibrated | 2000 | 0.4080 | 0.2697 | 0.1383 | 1.3565 | 0.7349 | 6.1983 | 13.5096 | 144.2184 | None |
| laya-med | non-AR | path_vqa | laya_med_calibrated | 2000 | 0.8685 | 0.7094 | 0.1591 | 0.4187 | 0.2510 | 5.4508 | 9.1712 | 160.6617 | None |
| laya-med | non-AR | pmc_vqa | laya_med_calibrated | 1980 | 0.4535 | 0.3270 | 0.1266 | 1.2658 | 0.6879 | 9.4381 | 16.0636 | 98.0750 | None |
| laya-med | non-AR | pubmedqa | laya_med_calibrated | 500 | 0.6400 | 0.7648 | 0.1378 | 0.9234 | 0.5347 | 23.5305 | 41.5612 | 39.4624 | None |
| laya-med | non-AR | slake | laya_med_calibrated | 355 | 0.8000 | 0.7706 | 0.0403 | 0.4702 | 0.2988 | 8.8283 | 9.6159 | 129.2550 | None |
| laya-med | non-AR | vqarad | laya_med_calibrated | 109 | 0.5596 | 0.7779 | 0.2342 | 0.8296 | 0.5850 | 5.3807 | 10.0539 | 180.1902 | None |

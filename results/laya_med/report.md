# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| laya-med | non-AR | med_qa | laya_med_calibrated | 1268 | 0.3013 | 0.2151 | 0.0862 | 1.5866 | 0.7908 | 30.5590 | 51.5674 | 31.4670 | None |
| laya-med | non-AR | medmcqa | laya_med_calibrated | 2000 | 0.4120 | 0.2601 | 0.1519 | 1.3693 | 0.7414 | 5.8420 | 13.5110 | 148.4126 | None |
| laya-med | non-AR | path_vqa | laya_med_calibrated | 2000 | 0.8780 | 0.6680 | 0.2100 | 0.4518 | 0.2758 | 5.1049 | 8.2881 | 174.2833 | None |
| laya-med | non-AR | pmc_vqa | laya_med_calibrated | 1980 | 0.5278 | 0.3023 | 0.2254 | 1.2770 | 0.6928 | 8.4815 | 15.2362 | 107.0487 | None |
| laya-med | non-AR | pubmedqa | laya_med_calibrated | 500 | 0.6160 | 0.7386 | 0.1226 | 0.9292 | 0.5377 | 27.0384 | 41.5727 | 37.2291 | None |
| laya-med | non-AR | slake | laya_med_calibrated | 355 | 0.8789 | 0.7453 | 0.1351 | 0.3852 | 0.2253 | 8.2475 | 8.7428 | 135.7507 | None |
| laya-med | non-AR | vqarad | laya_med_calibrated | 109 | 0.5688 | 0.6997 | 0.1309 | 0.6870 | 0.4892 | 4.8868 | 11.1248 | 191.9787 | None |

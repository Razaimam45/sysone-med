# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| models/qwen_direct_headonly | non-AR | pubmedqa | qwen_direct_headonly | 500 | 0.6560 | 0.6419 | 0.0568 | 0.8800 | 0.4849 | 90.9952 | 128.1681 | 10.5360 | 2026-10-04T07:58:25.531792+00:00 |
| models/qwen_direct_headonly | non-AR | med_qa | qwen_direct_headonly | 1268 | 0.4487 | 0.4386 | 0.0384 | 1.3314 | 0.6760 | 104.4014 | 173.0713 | 8.7526 | 2026-10-04T07:58:25.531792+00:00 |
| models/qwen_direct_headonly | non-AR | medmcqa | qwen_direct_headonly | 2000 | 0.5585 | 0.5608 | 0.0192 | 1.0188 | 0.5490 | 35.5651 | 48.8758 | 26.2383 | 2026-10-04T07:58:25.531792+00:00 |
| models/qwen_direct_headonly | non-AR | vqarad | qwen_direct_headonly | 109 | 0.7064 | 0.7023 | 0.1339 | 0.5348 | 0.3692 | 469.0200 | 848.3067 | 1.9257 | 2026-10-04T07:58:25.531792+00:00 |
| models/qwen_direct_headonly | non-AR | slake | qwen_direct_headonly | 355 | 0.7831 | 0.8062 | 0.0462 | 0.3925 | 0.2670 | 186.8351 | 691.8177 | 3.3318 | 2026-10-04T07:58:25.531792+00:00 |
| models/qwen_direct_headonly | non-AR | path_vqa | qwen_direct_headonly | 2000 | 0.8480 | 0.8560 | 0.0157 | 0.3542 | 0.2216 | 282.1337 | 293.9488 | 3.7591 | 2026-10-04T07:58:25.531792+00:00 |
| models/qwen_direct_headonly | non-AR | pmc_vqa | qwen_direct_headonly | 1980 | 0.5667 | 0.5632 | 0.0223 | 1.0192 | 0.5502 | 110.1036 | 378.3428 | 6.6525 | 2026-10-04T07:58:25.531792+00:00 |

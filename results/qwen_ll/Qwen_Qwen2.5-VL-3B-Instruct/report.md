# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | AR | pubmedqa | native_generate | 500 | 0.6800 | 0.7686 | 0.0935 | 0.8520 | 0.4600 | 43.7172 | 47.2624 | 24.9040 | 2026-10-03T22:24:53.612585+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | med_qa | native_generate | 1268 | 0.4306 | 0.6095 | 0.1796 | 1.5088 | 0.7436 | 33.9830 | 46.6873 | 29.2944 | 2026-10-03T22:24:53.612585+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | medmcqa | native_generate | 2000 | 0.5390 | 0.6220 | 0.0857 | 1.1355 | 0.6012 | 22.6836 | 25.0990 | 43.8120 | 2026-10-03T22:24:53.612585+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | vqarad | native_generate | 109 | 0.6514 | 0.6864 | 0.0814 | 0.5665 | 0.3881 | 240.1690 | 437.0190 | 3.7220 | 2026-10-03T22:24:53.612585+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | slake | native_generate | 355 | 0.7127 | 0.7928 | 0.0815 | 0.5161 | 0.3567 | 110.9074 | 368.9545 | 5.9621 | 2026-10-03T22:24:53.612585+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | path_vqa | native_generate | 2000 | 0.6190 | 0.7209 | 0.1019 | 0.6671 | 0.4660 | 149.8511 | 160.1576 | 6.9307 | 2026-10-03T22:24:53.612585+00:00 |
| Qwen/Qwen2.5-VL-3B-Instruct | AR | pmc_vqa | native_generate | 1980 | 0.4884 | 0.6255 | 0.1438 | 1.2264 | 0.6502 | 52.7836 | 112.5479 | 16.2572 | 2026-10-03T22:24:53.612585+00:00 |

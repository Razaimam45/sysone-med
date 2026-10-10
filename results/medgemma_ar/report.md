# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| google/medgemma-4b-it | AR | pubmedqa | conventional_llm | 500 | 0.7140 | 0.9835 | 0.2735 | 2.5014 | 0.5436 | 1034.4244 | 1076.5517 | 0.9620 | 2026-09-28T20:41:50.741597+00:00 |
| google/medgemma-4b-it | AR | vqarad | conventional_llm | 109 | 0.7064 | 0.9887 | 0.2823 | 1.8157 | 0.5512 | 1168.1396 | 1213.4543 | 0.8509 | 2026-09-28T20:41:50.741597+00:00 |
| google/medgemma-4b-it | AR | med_qa | conventional_llm | 1268 | 0.4921 | 0.9475 | 0.4553 | 3.7843 | 0.9468 | 1064.8573 | 1105.1226 | 0.9352 | 2026-09-28T20:41:50.741597+00:00 |
| google/medgemma-4b-it | AR | medmcqa | conventional_llm | 2000 | 0.6015 | 0.9730 | 0.3715 | 3.1154 | 0.7606 | 1049.4959 | 1098.5047 | 0.9468 | 2026-09-28T20:41:50.741597+00:00 |
| google/medgemma-4b-it | AR | slake | conventional_llm | 355 | 0.7944 | 0.9834 | 0.1925 | 1.4620 | 0.3889 | 1159.0233 | 1207.1368 | 0.8584 | 2026-09-28T20:41:50.741597+00:00 |
| google/medgemma-4b-it | AR | path_vqa | conventional_llm | 2000 | 0.6225 | 0.9613 | 0.3388 | 2.2024 | 0.6963 | 1162.3445 | 1890.9618 | 0.8079 | 2026-09-28T20:41:50.741597+00:00 |
| google/medgemma-4b-it | AR | pmc_vqa | conventional_llm | 1980 | 0.4874 | 0.9709 | 0.4835 | 4.1273 | 0.9798 | 1160.2694 | 1186.9866 | 0.8597 | 2026-09-28T20:41:50.741597+00:00 |

# Final comparison

p50 is median latency; p95 is the 95th percentile. Lower latency, ECE, NLL and Brier are better.

Conventional LLM generation produces up to 32 greedy tokens and parses a label prefix; native generates one token. Generation probability metrics use first-token full-vocabulary mass, including an invalid outcome. Jevify conditions probabilities on allowed labels. These probability bases differ, so inspect label mass and invalid counts in results.json when comparing.

Raw means no post-hoc temperature scaling, even for the original Jevify checkpoint (which already contains its trained LoRA). AR rows use batch size 1; non-AR rows use the requested batch size. Architecture identifies whether a row came from autoregressive generation/scoring or a non-autoregressive decision head.

| Model | Architecture | Dataset | Mode | N | Accuracy | Confidence | ECE | NLL | Brier | p50 (ms) | p95 (ms) | Examples/s | Timestamp (UTC) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| google/medgemma-4b-it | AR | pubmedqa | native_generate | 500 | 0.7140 | 0.9835 | 0.2735 | 2.5014 | 0.5436 | 50.8692 | 70.9027 | 19.0180 | 2026-09-28T23:19:44.230618+00:00 |
| google/medgemma-4b-it | AR | vqarad | native_generate | 109 | 0.7064 | 0.9887 | 0.2823 | 1.8157 | 0.5512 | 148.1447 | 152.6453 | 6.7230 | 2026-09-28T23:19:44.230618+00:00 |
| google/medgemma-4b-it | AR | med_qa | native_generate | 1268 | 0.4921 | 0.9475 | 0.4553 | 3.7843 | 0.9468 | 51.1172 | 58.0844 | 21.0844 | 2026-09-28T23:19:44.230618+00:00 |
| google/medgemma-4b-it | AR | medmcqa | native_generate | 2000 | 0.6015 | 0.9730 | 0.3715 | 3.1154 | 0.7606 | 34.8097 | 37.8836 | 28.3200 | 2026-09-28T23:19:44.230618+00:00 |
| google/medgemma-4b-it | AR | slake | native_generate | 355 | 0.7944 | 0.9834 | 0.1925 | 1.4620 | 0.3889 | 151.8569 | 157.4965 | 6.5767 | 2026-09-28T23:19:44.230618+00:00 |
| google/medgemma-4b-it | AR | path_vqa | native_generate | 2000 | 0.6225 | 0.9613 | 0.3388 | 2.2024 | 0.6963 | 153.3581 | 161.5991 | 6.4982 | 2026-09-28T23:19:44.230618+00:00 |
| google/medgemma-4b-it | AR | pmc_vqa | native_generate | 1980 | 0.4874 | 0.9709 | 0.4835 | 4.1273 | 0.9798 | 147.4749 | 152.3022 | 6.7542 | 2026-09-28T23:19:44.230618+00:00 |

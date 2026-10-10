# Structured Laya-Med study

Use `latex/section.tex` as the replacement experimental section, with `latex/paper_style.tex` in the preamble. Compile from the project root. The standalone entry point is `paper_results.tex`; `latex/paper_results.pdf` is the compiled preview. `latex/abstract.tex` provides an optional evidence-aligned abstract.

## Baseline design

| Method | Backbone | Decision mechanism | Evaluation |
|---|---|---|---|
| Gemma (AR) | AR VLM | Autoregressive generation | Available |
| MedGemma (AR) | AR VLM | Autoregressive generation | Available |
| MedGemma (LL) | AR VLM | Candidate-label likelihood scoring | Available |
| MedGemma-Direct | AR VLM | Learned direct candidate scoring | Pending |
| Laya-Med | NAR multimodal | Direct decision modeling | Available |

AR/NAR describes backbone architecture. Whether inference generates output tokens is a separate property. The saved Gemma checkpoint is multimodal Gemma-3-4B-IT. The measured likelihood scorer operates on candidate labels, not full answer sequences. No measured values are assigned to the pending learned-head control.

The six tables cover cohorts; baseline taxonomy plus matched accuracy; modality-stratified probability quality; output validity; inference efficiency; and independent full-cohort evaluation. Table 2 has two panels under one numbered caption.

## Six PNG figures

`figures2/` contains exactly seven PNG assets. Figure 1 is the supplied architecture. Figure 2 is a 3.48-inch vertical a/b/c computational schematic. Figures 3–6 analyze paired errors, modality-stratified reliability, selective risk, and within-MedGemma latency, respectively. Figure 7 is the three-panel test-time adaptation figure. Generated PNGs use 450 dpi, tight cropping, and restrained Seaborn colors. PNG is static: the schematic depicts interaction through feedback and branching, not hover/click controls.

Source fingerprints, curve coordinates, and reliability-bin values are in `latex/artifacts/analysis_data.json`. Original machine-readable model identifiers remain in the provenance data; the paper uses architectural and inference-mechanism terminology.

## Build

```bash
python scripts/make_paper_assets.py
python -m pytest tests/test_paper_assets.py -q
tectonic --outdir latex paper_results.tex
```

The generator requires NumPy, pandas, Matplotlib and Seaborn. `scripts/make_latex_tables.py` delegates to the same generator. The separate `latex/tables/tables.tex` and `latex/figures.tex` aggregators are optional; do not include them alongside the complete section.

The compilation bundle includes all seven PNGs and necessary LaTeX files. References are in `latex/references.bib`; merge them into the paper's bibliography. The learned-head evaluation, architectural ablations and cross-model memory measurements remain unmeasured, rather than implied by the abstract or tables.

## Test-time adaptation assets

`analysis/make_tta_assets.py` adds the reliability subsection as one table and one
figure: `table6_tta_main.tex` (label `tab:tta_main`) and `figure7_tta_methods.png`
(label `fig:tta_methods`, mirrored into `figures2/`), with the machine-readable
cells in `latex/artifacts/tta_data.json`. They are generated from `tta_results/`
rather than from `results/`, are aggregated per dataset and then macro-averaged, use
a reference accuracy recomputed from the identical test rows, and recompute
conformal coverage from the stored prediction sets and the candidate options. Run
`python analysis/make_tta_assets.py` after `tta/run_tta.py`; see
`analysis/README.md` for the flags and the exclusion list.

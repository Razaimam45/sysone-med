# Analysis assets

## Test-time adaptation (`make_tta_assets.py`)

Reads `tta_results/<baseline>_tta/` and emits the TTA reliability assets:

```bash
python analysis/make_tta_assets.py
```

| Output | Content |
|---|---|
| `latex/tables/table6_tta_main.tex` | one table with four rows per pipeline (no TTA, routing, conformal, online recalibration) across the seven datasets plus a macro-average column; every cell is an accuracy percentage |
| `figures/figure7_tta_methods.png` | three panels, one per mechanism: retained accuracy versus coverage, realised versus target coverage, static versus online temperature (mirrored into `figures2/` for LaTeX) |
| `latex/artifacts/tta_data.json` | curve coordinates, per-alpha statistics, ranking summary (AURC and retained accuracy), operating points and exclusion notes |

Covered pipelines: MedJev, Direct-Head, Direct-LoRA, Laya and LayaMed. MedGemma (AR)
has no local candidate logits, Jev is a proprietary hosted endpoint, and plain
LayaMed is superseded by the vision-LoRA checkpoint that is reported as *LayaMed*.
MedGemma (LL) qualifies by construction but has no adaptation report, so it has no
row rather than a dashed one; it appears as soon as `tta_results/` contains it.

Three invariants matter for the paper. Metrics are aggregated **per dataset and then
macro-averaged**, matching `table2_accuracy.tex`. The no-TTA reference accuracy
is recomputed from the identical test rows, so every TTA number is a paired
comparison, and conformal coverage is recomputed from the stored prediction sets
plus the candidate options in `data/prepared/test.jsonl` (the report's own
`covered` flag compares an integer label against a boolean mask and is unreliable).
`validate()` re-derives every headline metric from the raw rows and exits non-zero
if a cell is out of range, a mechanism misses a dataset, a curve leaves the
attainable tie bracket, an argmax moves, or a conformal row counts a different
number of examples than routing. Tests live in `tests/test_tta_assets.py`.

Useful flags: `--describe` prints every reported quantity (per-dataset cells,
coverage, temperatures, calibration gaps) without writing files, `--no-figure`
skips the PNG, and `--skip-validation` writes the assets anyway.

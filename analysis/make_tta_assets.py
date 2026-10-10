#!/usr/bin/env python3
"""Test-time-adaptation table and figure for the Laya-Med study.

Reads the JSON reports written by ``tta/run_tta.py`` and emits three assets:

* ``latex/tables/table6_tta_main.tex``   one comprehensive adaptation table
* ``figures/figure7_tta_methods.png``    three panels, one per TTA mechanism
* ``latex/artifacts/tta_data.json``      machine-readable cells and coordinates

Run from the project root::

    python analysis/make_tta_assets.py

Design notes
------------
* **One table, four rows per pipeline.** Every pipeline contributes a block of
  four rows -- the unadapted decision, confidence routing, conformal prediction
  sets, and online recalibration -- across the seven datasets plus a
  macro-average column. Columns are datasets rather than metrics, so the table
  reads as one accuracy matrix instead of three separate metric tables.
* **One reported quantity.** Every cell is an accuracy percentage on the same
  50 examples per dataset, so rows are directly comparable. Each mechanism is
  read at one fixed operating point: routing and recalibration at 80% retention,
  conformal at ``alpha = 0.1`` (target 90% coverage) reporting the accuracy on
  singleton decisions.
* **Paired reference.** The no-TTA row is recomputed from the identical TTA test
  rows, and temperature scaling preserves the argmax, so recalibration can only
  change a decision through the confidence ordering it supplies. Ranking the
  recalibrated rows at the same retention as routing keeps those two rows
  comparable instead of duplicating the reference row.
* **Aggregation.** Per dataset, then macro-averaged; pooled micro-averages are
  never mixed with macro-averages. Text-only pipelines cover three datasets and
  average only those, matching ``table2_accuracy.tex``.
* **Roster.** MedGemma (AR) emits free-form text with no candidate logits and
  Jev is a hosted proprietary endpoint; plain LayaMed is superseded by the
  vision-LoRA checkpoint. MedGemma (LL) is this study's reference
  candidate-likelihood path and is not carried into the adaptation comparison,
  so it never appears in these assets.
* **Figure style.** ``figure7`` follows the Accuracy--Latency Pareto figure
  (``analysis/a.ipynb``): white panels, despined axes, one pale preferred-region
  shading per panel, one marker per pipeline, and a single shared marker legend.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def relative(path):
    """Path relative to the repository root when possible, otherwise absolute."""
    try:
        return str(Path(path).relative_to(ROOT))
    except ValueError:
        return str(path)

DS = ['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa']
DN = ['PubMedQA', 'MedQA', 'MedMCQA', 'VQA-RAD', 'SLAKE', 'PathVQA', 'PMC-VQA']
NAMES = dict(zip(DS, DN))
TEXT_DS = DS[:3]
IMAGE_DS = DS[3:]

RETENTION = 0.8           # fixed operating point for routing and recalibration
ALPHA = 0.1               # fixed operating point for split conformal
TARGET = 1.0 - ALPHA

BASELINES = [
    dict(key='medjev', display='MedJev', ours=False,
         dirs=['medjev_tta'], modality='text'),
    dict(key='direct_headonly', display='Direct-Head', ours=True,
         dirs=['medgemma_direct_headonly_tta'], modality='both'),
    dict(key='direct_lora', display='Direct-LoRA', ours=True,
         dirs=['medgemma_direct_lora_tta'], modality='both'),
    dict(key='laya', display='Laya', ours=False,
         dirs=['laya_tta', 'laya'], modality='text'),
    dict(key='laya_med', display='LayaMed', ours=True,
         dirs=['laya_med_vl_tta'], modality='both'),
]

# Glyphs and colours are inherited from analysis/a.ipynb so that a pipeline
# keeps the same marker in every figure of the paper. The Pareto notebook names
# the learned-head variants "MedGemma-Head" and "MedGemma-LoRA".
PAPER_GLYPH = {
    'MedJev': ('#D65F5F', 'P', 68),
    'Direct-Head': ('#8C564B', 'D', 70),
    'Direct-LoRA': ('#D979B5', '^', 70),
    'Laya': ('#8C78B8', 'X', 68),
    'LayaMed': ('#6B7280', 'v', 70),
}

# Paper palette (latex/paper_style.tex) reused for the figure chrome.
INK = '#223F52'
SPINE = '#6B7280'
TICK = '#374151'
REGION = '#EAF3F0'          # PaperAccent
GOLD = '#BC893D'            # Paper grid / ceiling accent
FONT_SIZE = 12


def colour(display):
    return PAPER_GLYPH[display][0]


def marker(display):
    return PAPER_GLYPH[display][1]


def marker_size(display):
    return PAPER_GLYPH[display][2]
# ---------------------------------------------------------------- loading
def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as exc:
        print(f'error: malformed JSON {path}: {exc}', file=sys.stderr)
        raise


def load_report(spec) -> dict:
    entry = spec if isinstance(spec, dict) else dict(key=spec, dirs=[spec])
    report = {'key': entry['key'], 'display': entry['display'],
              'ours': entry['ours'], 'modality': entry['modality']}
    for name in entry['dirs']:
        if (ROOT / 'tta_results' / name / 'routing.json').exists():
            report['dir'] = name
            break
    base = ROOT / 'tta_results' / report.get('dir', '')
    report['routing'] = _read_json(base / 'routing.json')
    report['conformal'] = _read_json(base / 'conformal.json')
    report['drift'] = _read_json(base / 'drift_cal.json')
    return report


def index_by_dataset(report):
    if not isinstance(report, dict):
        return {}
    if all(isinstance(v, list) for v in report.values()):
        return report
    out = {}
    for name, value in report.items():
        if isinstance(value, dict):
            rows = value.get('rows', value.get('items'))
            if isinstance(rows, list):
                out[name] = rows
    return out


# ------------------------------------------------- core metric functions
def risk_coverage(rows):
    """(coverage, risk) at every integer retention level 1..n.

    Rows are ranked by descending confidence. Within a block of tied
    confidences the retained subset is unknown, so each level inside a block
    reports the expected error rate over uniformly random tie-breaking. Risk is
    the error rate *among the retained rows*, so ``1 - risk`` is the accuracy
    a reader would quote at that retention.
    """
    n = len(rows)
    if not n:
        return np.array([]), np.array([])
    conf = np.asarray([r['confidence'] for r in rows], float)
    correct = np.asarray([r['correct'] for r in rows], float)
    order = np.argsort(-conf, kind='mergesort')
    conf, correct = conf[order], correct[order]
    values, starts = np.unique(conf, return_index=True)
    order_values = np.argsort(-values)
    starts, ends = starts[order_values], list(starts[order_values][1:]) + [n]
    risk = np.zeros(n)
    seen, errors = 0, 0
    for start, end in zip(starts, ends):
        size = end - start
        block_errors = float((1 - correct[start:end]).sum())
        for level in range(seen + 1, seen + size + 1):
            taken = level - seen
            risk[level - 1] = (errors + taken * block_errors / size) / level
        seen += size
        errors += block_errors
    return np.arange(1, n + 1) / n, risk


def accuracy_at(coverage, risk, target):
    """1 - risk at the achievable retention closest to `target`."""
    if len(coverage) == 0:
        return float('nan')
    index = int(np.argmin(np.abs(np.asarray(coverage, float) - target)))
    return 1.0 - float(risk[index])


def bracket_at(rows, target=RETENTION):
    """(low, high) accuracy attainable at exactly `target` retention, ties free."""
    n = len(rows)
    if not n:
        return float('nan'), float('nan')
    k = int(round(target * n))
    k = max(1, min(n, k))
    ordered = sorted(rows, key=lambda r: -r['confidence'])
    threshold = ordered[k - 1]['confidence']
    above = [r for r in ordered if r['confidence'] > threshold]
    tie = [r for r in ordered if r['confidence'] == threshold]
    need = k - len(above)
    earned = sum(int(r['correct']) for r in above)
    correctness = sorted(int(r['correct']) for r in tie)
    low = earned + sum(correctness[:need])
    high = earned + sum(correctness[len(correctness) - need:] if need else [])
    return low / k, high / k


def mean_accuracy(rows):
    return float(np.mean([r['correct'] for r in rows])) if rows else float('nan')
def ece(rows, bins=15):
    """Expected calibration error with equal-width confidence bins."""
    if not rows:
        return float('nan')
    conf = np.asarray([r['confidence'] for r in rows], float)
    correct = np.asarray([r['correct'] for r in rows], float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total, ece_value = len(rows), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi) if hi < 1.0 else (conf >= lo) & (conf <= hi)
        if not mask.any():
            continue
        ece_value += mask.sum() / total * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece_value)


def conformal_stats(rows, options=None):
    """Coverage, set size and singleton accuracy for one dataset/alpha group.

    ``conformal.json`` stores ``covered`` as ``label in set_mask`` where
    ``set_mask`` is a boolean array, so an integer label is compared against
    ``True``/``False`` and the stored flag is wrong for labels >= 2 (verified:
    90 of 350 rows disagree for MedJev at alpha=0.1). The stored prediction
    *set* is correct, so coverage is recomputed from the candidate options in
    ``data/prepared/test.jsonl`` whenever the row's options are known.
    """
    if not rows:
        return {}
    options = options or {}
    sets = [list(r.get('set') or r.get('pred_set') or []) for r in rows]
    covered, missing = [], 0
    for row, pred_set in zip(rows, sets):
        opts = options.get(row['id'])
        if opts and 0 <= int(row['label']) < len(opts):
            covered.append(float(opts[int(row['label'])] in pred_set))
        else:
            missing += 1
            covered.append(float(bool(row.get('covered', False))))
    sizes = np.asarray([len(s) for s in sets], float)
    singleton = sizes == 1
    decided = int(singleton.sum())
    decided_accuracy = (
        float(np.mean([int(r['prediction'] == r['label'])
                       for r, s in zip(rows, sets) if len(s) == 1]))
        if decided else float('nan')
    )
    return {
        'coverage': float(np.mean(covered)),
        'mean_set_size': float(sizes.mean()),
        'singleton_rate': float(singleton.mean()),
        'decided': decided,
        'decided_accuracy': decided_accuracy,
        'rows': len(rows),
        'missing_options': missing,
    }


def macro(values):
    """Macro-average over datasets, ignoring missing entries."""
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return float('nan')
    return float(np.mean(finite))


def group_spec(entry):
    if entry['modality'] == 'text':
        return [('Text Avg.', TEXT_DS), ('Overall Avg.', TEXT_DS)]
    return [('Text Avg.', TEXT_DS), ('Image Avg.', IMAGE_DS), ('Overall Avg.', DS)]


# ------------------------------------------------------- TTA aggregates
def load_options(path=None):
    """Map test row id -> candidate options, for recomputing conformal coverage."""
    path = Path(path) if path else ROOT / 'data' / 'prepared' / 'test.jsonl'
    options = {}
    if path.exists():
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                options[row['id']] = list(row['options'])
    return options


def _group(rows):
    out = {}
    for row in rows or []:
        out.setdefault(row['dataset'], []).append(row)
    return out


def collect(reports, options=None):
    """Collapse raw reports into per-dataset curves, sets and diagnostics.

    ``routing.json`` stores the score of every test row once per tau sweep, so
    the first block's rows are reused. ``conformal.json`` stores one row block
    per alpha, and ``drift_cal.json`` stores one flat row list plus the refit
    windows. Only these raw rows are consumed; pooled aggregates in the reports
    are recomputed per dataset before any macro-average.
    """
    options = options if options is not None else load_options()
    records = []
    for rep in reports:
        routing_raw = rep['routing'] or {}
        conformal_raw = rep['conformal'] or {}
        drift_raw = rep['drift'] or {}
        if not routing_raw and not conformal_raw and not drift_raw:
            continue
        blocks = routing_raw.get('results') or []
        routing = {}
        for d, rows in _group(blocks[0].get('rows') if blocks else []).items():
            coverage, risk = risk_coverage(rows)
            routing[d] = dict(rows=rows, base=mean_accuracy(rows),
                              coverage=coverage, risk=risk)
        conformal = {}
        for block in conformal_raw.get('results') or []:
            alpha = round(float(block['alpha']), 4)
            group = {d: conformal_stats(rows, options)
                     for d, rows in _group(block.get('rows')).items()}
            if group:
                conformal[alpha] = group
        drift_rows = drift_raw.get('rows') or []
        drift = {}
        for d, rows in _group(drift_rows).items():
            coverage, risk = risk_coverage(rows)
            drift[d] = dict(rows=rows, base=mean_accuracy(rows),
                            accuracy=mean_accuracy(rows),
                            coverage=coverage, risk=risk)
        datasets = [d for d in DS
                    if d in routing or d in conformal.get(ALPHA, {}) or d in drift]
        records.append(dict(
            key=rep['key'], display=rep['display'], ours=rep['ours'],
            modality=rep['modality'], datasets=datasets,
            routing=routing, conformal=conformal, drift=drift,
            static_temperature=float(routing_raw.get('temperature', float('nan'))),
            online_temperature=(float(np.mean([float(r['temperature']) for r in drift_rows]))
                                if drift_rows else float('nan')),
            windows=list(drift_raw.get('temperatures') or []),
            drift_alarms=list(drift_raw.get('drift_alarms') or []),
            test_n=int(routing_raw.get('test_n', 0)),
            calibration_n=int(conformal_raw.get('calibration_n', 0)),
            final_accuracy=float(drift_raw.get('final_accuracy', float('nan'))),
            tau=[float(b['tau']) for b in blocks],
            alpha=[round(float(b['alpha']), 4)
                   for b in (conformal_raw.get('results') or [])],
        ))
    return records
# --------------------------------------------- reported quantities
def base_cells(entry):
    """No-TTA accuracy per dataset, recomputed from the identical TTA rows."""
    return {d: g['base'] for d, g in entry['routing'].items()}


def routing_cells(entry, q=RETENTION):
    """Accuracy among the most confident fraction q (confidence routing)."""
    return {d: accuracy_at(g['coverage'], g['risk'], q)
            for d, g in entry['routing'].items()}


def conformal_cells(entry, alpha=ALPHA):
    """Accuracy on singleton decisions; datasets with no singleton are absent."""
    group = entry['conformal'].get(alpha) or {}
    out = {}
    for d, stats in group.items():
        if stats['decided'] > 0 and np.isfinite(stats['decided_accuracy']):
            out[d] = float(stats['decided_accuracy'])
    return out


def recalibration_cells(entry, q=RETENTION):
    """Accuracy among the most confident fraction q under the refit ordering."""
    return {d: accuracy_at(g['coverage'], g['risk'], q)
            for d, g in entry['drift'].items()}


def coverage_cells(entry, alpha=ALPHA):
    """Realised conformal coverage per dataset at one alpha."""
    return {d: float(s['coverage'])
            for d, s in (entry['conformal'].get(alpha) or {}).items()}


def macro_curve(entry):
    """Macro-averaged risk curve over every dataset the pipeline covers."""
    datasets = [d for d in DS if d in entry['routing']]
    if not datasets:
        return np.array([]), np.array([])
    length = min(len(entry['routing'][d]['risk']) for d in datasets)
    risk = np.mean([entry['routing'][d]['risk'][:length] for d in datasets], axis=0)
    return np.arange(1, length + 1) / length, risk


def calibration_gap(rows):
    """Mean confidence minus accuracy; positive means overconfidence."""
    if not rows:
        return float('nan')
    return float(np.mean([r['confidence'] for r in rows])
                 - np.mean([r['correct'] for r in rows]))


def _pooled(entry, which, key):
    rows = [r for g in entry[which].values() for r in g['rows']]
    if not rows:
        return float('nan')
    if key == 'gap':
        return calibration_gap(rows)
    if key == 'ece':
        return ece(rows)
    return float(np.mean([r['correct'] for r in rows]))


def describe(records):
    """Print every reported quantity so the paper text can quote real values."""
    print('per-dataset rows: base | routing@80 | conformal@0.1 | recal@80 (per cent)')
    for rec in records:
        print(f"\n== {rec['display']}  ours={rec['ours']}  modality={rec['modality']}  "
              f"datasets={len(rec['datasets'])}")
        for d in rec['datasets']:
            parts = []
            for _, fn in REPORTED_ROWS:
                cells = fn(rec)
                parts.append('   --' if d not in cells else f'{100 * cells[d]:5.1f}')
            print(f'   {NAMES[d]:<10}' + ' | '.join(parts))
        averages = [macro([fn(rec).get(d) for d in rec['datasets']]) for _, fn in REPORTED_ROWS]
        print(f'   {"Average":<10}' + ' | '.join(
            '   --' if not np.isfinite(v) else f'{100 * v:5.1f}' for v in averages))
        alpha_group = rec['conformal'].get(ALPHA) or {}
        decided = sum(s['decided'] for s in alpha_group.values())
        missing = sum(s['missing_options'] for s in alpha_group.values())
        cov = macro(list(coverage_cells(rec).values()))
        cov_pooled = (float(sum(s['coverage'] * s['rows'] for s in alpha_group.values())
                            / max(1, sum(s['rows'] for s in alpha_group.values())))
                      if alpha_group else float('nan'))
        size = macro([s['mean_set_size'] for s in alpha_group.values()])
        print(f'   conformal alpha={ALPHA}: realised coverage {100 * cov:.1f}% macro / '
              f'{100 * cov_pooled:.1f}% pooled vs target {100 * TARGET:.0f}%, '
              f'singletons {decided}/{rec["test_n"]}, mean set size {size:.2f}, '
              f'rows without options {missing}')
        window_t = [w['temperature'] for w in rec['windows']]
        print(f'   temperatures: static {rec["static_temperature"]:.2f} -> online '
              f'{rec["online_temperature"]:.2f} (windows '
              f'{", ".join(f"{t:.2f}" for t in window_t)}), '
              f'grid boundary {any(w["at_grid_boundary"] for w in rec["windows"])}, '
              f'drift alarms {len(rec["drift_alarms"])}')
        print(f'   calibration gap pp before {100 * _pooled(rec, "routing", "gap"):+.1f} -> '
              f'after {100 * _pooled(rec, "drift", "gap"):+.1f}; '
              f'ECE before {100 * _pooled(rec, "routing", "ece"):.1f} -> '
              f'after {100 * _pooled(rec, "drift", "ece"):.1f}; '
              f'micro accuracy {_pooled(rec, "drift", "acc") * 100:.1f} / '
              f'final {rec["final_accuracy"] * 100:.1f}')
# ---------------------------------------------------------- LaTeX table
REPORTED_ROWS = [
    ('no TTA', base_cells),
    ('+~Routing', routing_cells),
    ('+~Conformal', conformal_cells),
    ('+~Online recal.', recalibration_cells),
]

TABLE6_CAPTION = (
    'Test-time adaptation on the matched pilot (50 test examples per dataset). '
    'Each pipeline contributes a block of four rows, and every cell is an accuracy '
    'percentage: the unadapted decision with all examples retained; confidence routing '
    'with the 80\\% most confident decisions retained; split conformal prediction at '
    '$\\alpha=0.1$ evaluated on singleton decisions only; and online temperature '
    'recalibration ranked by the sliding-window refit confidence at the same 80\\% '
    'retention. A dash in the conformal row means no example in that dataset received '
    'a singleton set, so the pipeline deferred every decision there. Shading is scaled '
    'within each column and the best entry in a column is bold. The unadapted row is '
    'recomputed from the identical TTA test rows, and temperature scaling preserves the '
    'argmax, so the recalibration row can change decisions only through the confidence '
    'ordering it supplies. The Average column macro-averages the datasets a pipeline '
    'covers, matching Table~\\ref{tab:accuracy}; dashes are never averaged in, so the '
    'two text-only pipelines average their three text tasks. One example moves a cell '
    'by 2.0 points.'
)


def heat(value, low, high, ramp=((252, 233, 220), (176, 216, 177))):
    """Pale-warm to green background for one cell (paper accent palette)."""
    if value is None or not np.isfinite(value):
        return ''
    span = high - low
    t = 0.0 if span <= 1e-12 else float((value - low) / span)
    floor, ceiling = ramp
    channels = tuple(int(round(lo + t * (hi - lo))) for lo, hi in zip(floor, ceiling))
    return '\\cellcolor[RGB]{%d,%d,%d} ' % channels


def num(value, decimals=1):
    if value is None or not np.isfinite(value):
        return '---'
    return f'{value:.{decimals}f}'


def _table6_cells(records):
    """Rows per pipeline, each with a macro-average column appended."""
    blocks = []
    for rec in records:
        rows = []
        for _, fn in REPORTED_ROWS:
            cells = dict(fn(rec))
            cells['avg'] = macro(list(cells.values()))
            rows.append(cells)
        blocks.append((rec, rows))
    return blocks


def render_table6(records):
    """Return the LaTeX source of the comprehensive adaptation table."""
    blocks = _table6_cells(records)
    columns = DS + ['avg']
    present = {c: [cells[c] for _, rows in blocks for cells in rows
                   if c in cells and np.isfinite(cells[c])] for c in columns}
    low = {c: (min(vals) if vals else 0.0) for c, vals in present.items()}
    high = {c: (max(vals) if vals else 1.0) for c, vals in present.items()}
    best = {c: (max(vals) if vals else None) for c, vals in present.items()}

    def cell(value, column):
        if value is None or not np.isfinite(value):
            return '---'
        text = num(100 * value, 1)
        if best[column] is not None and abs(value - best[column]) < 1e-12:
            text = f'\\textbf{{{text}}}'
        return heat(value, low[column], high[column]) + text

    lines = [
        '% Generated by analysis/make_tta_assets.py. Do not edit by hand.',
        '\\begin{table*}[t]',
        '\\centering',
        '\\small',
        f'\\caption{{{TABLE6_CAPTION}}}',
        '\\label{tab:tta_main}',
        '\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l' + 'r' * 8 + '@{}}',
        '\\toprule',
        '\\textbf{Pipeline} & '
        + ' & '.join(f'\\textbf{{{NAMES[d]}}}' for d in DS)
        + ' & \\textbf{Average} \\\\',
        '\\midrule',
    ]
    for block_index, (rec, rows) in enumerate(blocks):
        for row_index, (label, fn) in enumerate(REPORTED_ROWS):
            if row_index == 0:
                name = rec['display']
                lead = (f'\\textcolor{{PaperPurple}}{{\\textbf{{{name}}}}}' if rec['ours']
                        else f'\\textbf{{{name}}}')
            else:
                lead = f'\\hspace{{1.1em}}{label}'
            cells = rows[row_index]
            lines.append(' & '.join([lead] + [cell(cells.get(c), c) for c in columns]) + ' \\\\')
        if block_index < len(blocks) - 1:
            lines.append('\\midrule')
    lines += ['\\bottomrule', '\\end{tabular*}', '\\end{table*}', '']
    return '\n'.join(lines)


def write_table6(records, tables_dir=None):
    tables_dir = Path(tables_dir) if tables_dir else ROOT / 'latex' / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    path = tables_dir / 'table6_tta_main.tex'
    path.write_text(render_table6(records))
    return path
# ---------------------------------------------------------------- figure
def _figure_setup():
    """Matplotlib/Seaborn settings copied from analysis/a.ipynb's Pareto figure."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    import seaborn as sns
    rcParams['pdf.fonttype'] = 42
    rcParams['axes.unicode_minus'] = False
    sns.set_theme(context='paper', style='white', font_scale=1.0)
    return plt, sns


def style_axes(ax, sns):
    sns.despine(ax=ax, top=True, right=True, left=False, bottom=False,
                offset=2, trim=False)
    ax.spines['left'].set_color(SPINE)
    ax.spines['bottom'].set_color(SPINE)
    ax.spines['left'].set_linewidth(0.75)
    ax.spines['bottom'].set_linewidth(0.75)
    ax.tick_params(axis='both', which='major', colors=TICK, labelsize=FONT_SIZE,
                   length=3, width=0.75, pad=3)
    ax.grid(False)
    ax.set_facecolor('#FFFFFF')


def _panel_title(ax, text):
    ax.set_title(text, loc='left', pad=8, fontsize=FONT_SIZE, color=INK,
                 fontweight='bold')


def _style(rec):
    col = colour(rec['display'])
    return dict(color=col, marker=marker(rec['display']), size=marker_size(rec['display']),
                lw=2.0 if rec['ours'] else 1.4,
                ls='-' if rec['ours'] else (0, (3, 2)))


def _point(ax, rec, x, y, scale=1.0, highlight=False, zorder=5):
    st = _style(rec)
    ax.scatter([x], [y], marker=st['marker'], s=st['size'] * scale,
               facecolor=st['color'], edgecolor=INK if highlight else 'white',
               linewidth=1.2, zorder=zorder)


def panel_selective(ax, sns, records):
    """Accuracy among retained decisions versus retained coverage."""
    style_axes(ax, sns)
    ax.axvspan(RETENTION * 100, 103, facecolor=REGION, alpha=.7, lw=0, zorder=-6)
    ax.axvline(RETENTION * 100, color=SPINE, lw=.8, ls=(0, (2, 2)), zorder=-4)
    export, y_min, y_max = [], 100.0, 0.0
    for rec in records:
        cov, risk = macro_curve(rec)
        if not len(cov):
            continue
        st = _style(rec)
        x, y = cov * 100, (1 - risk) * 100
        ax.plot(x, y, color=st['color'], lw=st['lw'], ls=st['ls'], alpha=.95, zorder=3)
        point = accuracy_at(cov, risk, RETENTION) * 100
        _point(ax, rec, RETENTION * 100, point, highlight=True)
        y_min, y_max = min(y_min, y.min()), max(y_max, y.max())
        export.append(dict(pipeline=rec['display'], ours=rec['ours'],
                           datasets=[d for d in DS if d in rec['routing']],
                           coverage=[float(v) for v in cov],
                           retained_accuracy=[float(v) for v in (1 - risk)],
                           operating_point=dict(retention=RETENTION, accuracy=float(point / 100))))
    pad = max(3.0, 0.08 * (y_max - y_min))
    ax.set_xlim(5, 103)
    ax.set_ylim(max(0, y_min - pad), min(100, y_max + pad))
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_xlabel('Retained coverage (%)', fontsize=FONT_SIZE, color=INK)
    ax.set_ylabel('Retained accuracy (%)', fontsize=FONT_SIZE, color=INK)
    ax.text(0.97, 0.96, '80% retention', transform=ax.transAxes, ha='right',
            va='top', fontsize=9, color=SPINE)
    _panel_title(ax, '(a)  Selective routing')
    return export


def panel_conformal(ax, sns, records):
    """Realised versus target coverage, with exact calibration on the diagonal."""
    style_axes(ax, sns)
    lo, hi = 62, 97
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], facecolor=REGION, alpha=.7,
                    lw=0, zorder=-6)
    ax.plot([lo, hi], [lo, hi], color=SPINE, lw=1.0, ls=(0, (3, 2)), zorder=3)
    export = []
    for rec in records:
        st = _style(rec)
        alphas = sorted(rec['conformal'], reverse=True)
        points = []
        for alpha in alphas:
            group = rec['conformal'][alpha]
            realised = macro([s['coverage'] for s in group.values()])
            if not np.isfinite(realised):
                continue
            points.append(dict(alpha=alpha, target=1 - alpha, realised=float(realised),
                               mean_set_size=float(macro([s['mean_set_size']
                                                          for s in group.values()]))))
        if not points:
            continue
        xs = [p['target'] * 100 for p in points]
        ys = [p['realised'] * 100 for p in points]
        ax.plot(xs, ys, color=st['color'], lw=st['lw'], ls=st['ls'], alpha=.95, zorder=3)
        for x, y, p in zip(xs, ys, points):
            highlight = abs(p['target'] - TARGET) < 1e-9
            _point(ax, rec, x, y, scale=1.0 if highlight else 0.55, highlight=highlight)
        export.append(dict(pipeline=rec['display'], ours=rec['ours'], points=points))
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xticks([70, 80, 90])
    ax.set_yticks([70, 80, 90])
    ax.set_xlabel('Target coverage (%)', fontsize=FONT_SIZE, color=INK)
    ax.set_ylabel('Realised coverage (%)', fontsize=FONT_SIZE, color=INK)
    ax.text(0.07, 0.93, 'meets target', transform=ax.transAxes, ha='left',
            va='top', fontsize=9, color=SPINE)
    ax.text(0.93, 0.10, 'under-covered', transform=ax.transAxes, ha='right',
            va='bottom', fontsize=9, color=SPINE)
    _panel_title(ax, '(b)  Conformal sets')
    return export
def panel_recalibration(ax, sns, records):
    """Static fitted temperature versus the online refit temperature."""
    style_axes(ax, sns)
    lo, hi = 0.7, 30
    ax.fill_between([lo, hi], [lo, hi], [hi, hi], facecolor=REGION, alpha=.7,
                    lw=0, zorder=-6)
    ax.plot([lo, hi], [lo, hi], color=SPINE, lw=1.0, ls=(0, (3, 2)), zorder=3)
    ax.axhline(20, color=GOLD, lw=.9, ls=(0, (4, 3)), zorder=3)
    export = []
    for rec in records:
        static, online = rec['static_temperature'], rec['online_temperature']
        if not (np.isfinite(static) and np.isfinite(online)):
            continue
        st = _style(rec)
        window_t = [w['temperature'] for w in rec['windows']]
        if len(window_t) > 1:
            ax.vlines(static, min(window_t), max(window_t), color=st['color'],
                      lw=5, alpha=.28, zorder=4, capstyle='round')
        ax.annotate('', xy=(static, online), xytext=(static, static),
                    arrowprops=dict(arrowstyle='->', color=st['color'],
                                    lw=max(1.2, st['lw']), alpha=.75,
                                    shrinkA=2, shrinkB=2), zorder=4)
        _point(ax, rec, static, online, highlight=True)
        export.append(dict(
            pipeline=rec['display'], ours=rec['ours'],
            static_temperature=float(static), online_temperature=float(online),
            window_temperatures=[float(t) for t in window_t],
            drift_alarms=len(rec['drift_alarms']),
            at_grid_boundary=any(w.get('at_grid_boundary') for w in rec['windows']),
            calibration_gap_after=float(_pooled(rec, 'drift', 'gap')),
        ))
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ticks = [1, 2, 5, 10, 20]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.set_yticklabels([str(t) for t in ticks])
    ax.set_xlabel('Static fitted $T$', fontsize=FONT_SIZE, color=INK)
    ax.set_ylabel('Online window $T$', fontsize=FONT_SIZE, color=INK)
    ax.text(0.06, 0.86, 'online $T$ above static', transform=ax.transAxes, ha='left',
            va='top', fontsize=9, color=SPINE)
    ax.text(0.94, 0.07, 'grid ceiling = 20; no drift alarms', transform=ax.transAxes,
            ha='right', va='bottom', fontsize=9, color=SPINE)
    _panel_title(ax, '(c)  Online recalibration')
    return export


def figure7(records, figures_dir=None):
    """Render the three-panel TTA figure; returns (path, panel export)."""
    from matplotlib.lines import Line2D
    figures_dir = Path(figures_dir) if figures_dir else ROOT / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt, sns = _figure_setup()
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.5))
    panels = {
        'selective': panel_selective(axes[0], sns, records),
        'conformal': panel_conformal(axes[1], sns, records),
        'recalibration': panel_recalibration(axes[2], sns, records),
    }
    handles, labels = [], []
    for rec in records:
        handles.append(Line2D([], [], marker=marker(rec['display']),
                              color='none', markersize=9,
                              markerfacecolor=colour(rec['display']),
                              markeredgecolor='white', markeredgewidth=1.2))
        labels.append(rec['display'])
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.005),
               ncol=len(records), frameon=False, fontsize=FONT_SIZE,
               handletextpad=0.3, columnspacing=1.8)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.205,
                        wspace=0.30)
    path = figures_dir / 'figure7_tta_methods.png'
    fig.savefig(path, dpi=450, bbox_inches='tight', pad_inches=0.05,
                facecolor='white')
    plt.close(fig)
    # The paper includes PNGs from figures2/, which mirrors figures/; keep the
    # mirror in step when the canonical directory is the one being written.
    mirror = ROOT / 'figures2'
    if figures_dir == ROOT / 'figures' and mirror.is_dir():
        shutil.copy2(path, mirror / path.name)
    return path, panels
# ---------------------------------------------------------- validation
def reference_at(rows, target=RETENTION):
    """Accuracy of the top-k most confident rows, computed without the curve."""
    order = sorted(range(len(rows)), key=lambda i: -rows[i]['confidence'])
    k = max(1, int(round(target * len(order))))
    return float(np.mean([rows[i]['correct'] for i in order[:k]]))


def _pairs(rows):
    return {r['id']: (int(r['label']), int(r['prediction'])) for r in rows}


def validate(records):
    """Return a list of invariant violations; empty means the assets are safe."""
    problems = []
    if len(records) != len(BASELINES):
        problems.append(f'expected {len(BASELINES)} pipelines, found {len(records)}')
    for rec in records:
        name = rec['display']
        if not rec['datasets']:
            problems.append(f'{name}: no datasets reported')
        for d in rec['datasets']:
            if d not in DS:
                problems.append(f'{name}: unexpected dataset {d}')
        for label, cells in (('no TTA', base_cells(rec)), ('routing', routing_cells(rec)),
                             ('recalibration', recalibration_cells(rec))):
            for d, value in cells.items():
                if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                    problems.append(f'{name}/{label}/{d}: out of range: {value}')
            if len(cells) != len(rec['datasets']):
                problems.append(f'{name}/{label}: covers {len(cells)} of '
                                f'{len(rec["datasets"])} datasets')
        # The conformal row may legitimately dash a dataset whose prediction sets
        # are never singletons; it must still be non-empty and in range.
        conformal_cells_ = conformal_cells(rec)
        for d, value in conformal_cells_.items():
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                problems.append(f'{name}/conformal/{d}: out of range: {value}')
        if not conformal_cells_:
            problems.append(f'{name}/conformal: no singleton decisions anywhere')
        if ALPHA not in rec['conformal']:
            problems.append(f'{name}: no conformal results at alpha={ALPHA}')
        for alpha, group in rec['conformal'].items():
            for d, stats in group.items():
                if stats['missing_options']:
                    problems.append(f'{name}/alpha={alpha}/{d}: {stats["missing_options"]} '
                                    'rows had no candidate options for coverage')
                if not 0.0 <= stats['coverage'] <= 1.0:
                    problems.append(f'{name}/alpha={alpha}/{d}: coverage out of range')
        for label, groups in (('routing', rec['routing']), ('recalibration', rec['drift'])):
            for d, group in groups.items():
                if abs(RETENTION * len(group['rows']) - round(RETENTION * len(group['rows']))) > 1e-9:
                    continue  # bracket_at reports integer retention only
                got = accuracy_at(group['coverage'], group['risk'], RETENTION)
                low, high = bracket_at(group['rows'])
                if not low - 1e-9 <= got <= high + 1e-9:
                    problems.append(f'{name}/{label}/{d}: {got:.4f} outside the '
                                    f'attainable bracket [{low:.4f}, {high:.4f}]')
        # Temperature scaling may only reorder confidence, never move the argmax,
        # and every method is scored on the same ids per dataset.
        route_by_ds = _group([r for g in rec['routing'].values() for r in g['rows']])
        drift_by_ds = _group([r for g in rec['drift'].values() for r in g['rows']])
        if set(route_by_ds) != set(drift_by_ds):
            problems.append(f'{name}: routing and recalibration disagree on datasets')
        for d, rows in drift_by_ds.items():
            other = route_by_ds.get(d)
            if other is None:
                continue
            if _pairs(rows) != _pairs(other):
                problems.append(f'{name}/{d}: recalibration changed the argmax')
        alpha_group = rec['conformal'].get(ALPHA, {})
        for d, stats in alpha_group.items():
            scored = len(route_by_ds.get(d, []))
            if stats['rows'] != scored:
                problems.append(f'{name}/{d}: conformal scored {stats["rows"]} rows, '
                                f'routing scored {scored}')
    return problems
# ------------------------------------------------------------- artifact
def build_artifact(records, panels, table_path, figure_path):
    blocks = _table6_cells(records)
    pipelines = []
    for rec, rows in blocks:
        entries = {label: cells for (label, _), cells in zip(REPORTED_ROWS, rows)}
        alpha_group = rec['conformal'].get(ALPHA) or {}
        pipelines.append(dict(
            pipeline=rec['display'], ours=rec['ours'], modality=rec['modality'],
            datasets=rec['datasets'],
            macro_accuracy={label: macro(list(cells.values())) for label, cells in entries.items()},
            ranking=dict(
                aurc=(float(macro_curve(rec)[1].mean()) if rec['routing'] else float('nan')),
                retained_accuracy={f'{q:g}': macro([routing_cells(rec, q).get(d) for d in DS])
                                   for q in (0.5, 0.6, 0.8, 0.9)}),
            conformal=dict(
                alpha=ALPHA, target_coverage=TARGET,
                realised_coverage=macro([s['coverage'] for s in alpha_group.values()]),
                mean_set_size=macro([s['mean_set_size'] for s in alpha_group.values()]),
                singletons=sum(s['decided'] for s in alpha_group.values()),
                test_n=rec['test_n']),
            calibration=dict(
                static_temperature=rec['static_temperature'],
                online_temperature=rec['online_temperature'],
                window_temperatures=[w['temperature'] for w in rec['windows']],
                at_grid_boundary=any(w.get('at_grid_boundary') for w in rec['windows']),
                drift_alarms=len(rec['drift_alarms']),
                gap_before=_pooled(rec, 'routing', 'gap'),
                gap_after=_pooled(rec, 'drift', 'gap'),
                ece_before=_pooled(rec, 'routing', 'ece'),
                ece_after=_pooled(rec, 'drift', 'ece')),
        ))
    table_rows = [
        dict(pipeline=rec['display'], ours=rec['ours'],
             rows=[dict(label=label, cells=dict(cells))
                   for (label, _), cells in zip(REPORTED_ROWS, rows)])
        for rec, rows in blocks
    ]
    return {
        'generated_by': 'analysis/make_tta_assets.py',
        'sources': ['tta_results/<pipeline>_tta/routing.json',
                    'tta_results/<pipeline>_tta/conformal.json',
                    'tta_results/<pipeline>_tta/drift_cal.json',
                    'data/prepared/test.jsonl'],
        'roster': dict(included=[r['display'] for r in records],
                       excluded=['MedGemma (AR)', 'Jev', 'MedGemma (LL)',
                                 'LayaMed (dense checkpoint)']),
        'operating_points': dict(retention=RETENTION, alpha=ALPHA,
                                 target_coverage=TARGET, examples_per_dataset=50),
        'table6': dict(label='tab:tta_main', file=relative(table_path),
                       columns=[NAMES[d] for d in DS] + ['Average'], rows=table_rows),
        'figure7': dict(file=relative(figure_path), panels=panels),
        'pipelines': pipelines,
    }
# ------------------------------------------------------------------ main
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description='Build the TTA table, figure and JSON artifact.')
    parser.add_argument('--describe', action='store_true',
                        help='print every reported quantity and stop')
    parser.add_argument('--skip-validation', action='store_true',
                        help='write the assets even if an invariant fails')
    parser.add_argument('--no-figure', action='store_true',
                        help='skip the PNG (useful when matplotlib is missing)')
    parser.add_argument('--tables-dir', type=Path, default=None)
    parser.add_argument('--figures-dir', type=Path, default=None)
    parser.add_argument('--artifact', type=Path, default=None)
    args = parser.parse_args(argv)

    reports = [load_report(spec) for spec in BASELINES]
    records = collect(reports, load_options())
    if not records:
        print('error: no TTA reports under tta_results/', file=sys.stderr)
        return 1
    if args.describe:
        describe(records)
        return 0
    if not args.skip_validation:
        problems = validate(records)
        if problems:
            print('validation failed:', file=sys.stderr)
            for problem in problems:
                print(f'  - {problem}', file=sys.stderr)
            return 2
        print(f'validated {len(records)} pipelines '
              '(cells in [0,1], paired rows, argmax preserved)')

    table_path = write_table6(records, args.tables_dir)
    print(f'wrote {relative(table_path)}')

    panels = {}
    figure_path = ROOT / 'figures' / 'figure7_tta_methods.png'
    if not args.no_figure:
        figure_path, panels = figure7(records, args.figures_dir)
        print(f'wrote {relative(figure_path)}')

    artifact_path = args.artifact or ROOT / 'latex' / 'artifacts' / 'tta_data.json'
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = build_artifact(records, panels, table_path, figure_path)
    artifact_path.write_text(json.dumps(artifact, indent=2, default=float) + '\n')
    print(f'wrote {relative(artifact_path)}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Aggregate TTA method outputs into a single comparison table.

Reads the JSON reports written by each method script and prints a compact
comparison: baseline accuracy, adapted accuracy, and the method-specific
headline metric (coverage frontier / set sizes / drift alarms).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load(path):
    with Path(path).open() as handle:
        return json.load(handle)


def baseline_acc(report, temperature=None):
    rows = report.get('baseline') or report.get('rows') or []
    if not rows:
        return None
    temp = temperature if temperature is not None else report.get('temperature', 1.0)
    return float(np.mean([int((np.asarray(r['logits'], float) / temp).argmax() == r['label']) for r in rows]))


def summarize(report):
    method = report.get('method')
    if method == 'test_time_routing':
        best = max(report['results'], key=lambda b: (b['coverage'], b['accepted_accuracy'] or 0))
        return {'headline': f"best accepted_acc={best['accepted_accuracy']} @cov={best['coverage']:.2f}"}
    if method == 'conformal_prediction':
        b = report['results'][0]
        return {'headline': f"alpha={b['alpha']} cov={b['empirical_coverage']:.2f} "
                           f"set={b['mean_set_size']:.2f} decided_acc={b['decided_accuracy']}"}
    if method == 'domain_shifted_calibration':
        return {'headline': f"final_acc={report['final_accuracy']:.3f} "
                           f"mean_T={report['mean_temperature']:.2f} alarms={len(report['drift_alarms'])}"}
    if method == 'rlcd_test_time_finetune':
        return {'headline': f"baseline={report.get('baseline_accuracy')} "
                           f"adapted={report.get('adapted_accuracy')}"}
    return {'headline': str(method)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports', nargs='+', help='JSON report paths from the TTA methods')
    args = p.parse_args()
    print(f"{'method':32s} {'test_n':>6s}  headline")
    print('-' * 80)
    for path in args.reports:
        report = load(path)
        s = summarize(report)
        print(f"{report.get('method', Path(path).stem):32s} {report.get('test_n', 0):6d}  {s['headline']}")


if __name__ == '__main__':
    main()
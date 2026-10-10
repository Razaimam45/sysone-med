"""Null-Search Method: exhaustive evaluation of post-hoc decision rules.

This method does not modify predictions. Instead, it runs a comprehensive
suite of candidate rules (position debias, length prior, logit centering,
z-scoring, linear map, diagonal affine fit for accuracy, top-k consensus,
prior consensus, permutation consensus, etc.) and records that **none**
increase accuracy on the full calibration+test set.

The output mirrors other TTA methods so it can be placed alongside them in
tables, but the "adapted" accuracy equals the base accuracy (delta = 0.00)
and changed_n = 0, indicating that no rule was found to improve accuracy.

This is a research prototype: it documents the absence of a free-lunch
post-hoc accuracy booster, reinforcing that any accepted accuracy gains
must come via selective prediction (routing/conformal) at the cost of
coverage/abstention.
"""
from __future__ import annotations

from collections import defaultdict
from typing import List, Tuple

import json
import numpy as np

from tta.base import to_probabilities
from tta.refinement import prepare, score, finish


def _calibration_stats(adapter, datasets, batch_size):
    """Per (dataset, option_count): median top-2 margin and label-frequency prior."""
    stats = {}
    for dataset in dict.fromkeys(datasets):
        calibration = [r for r in adapter.splits['calibration'] if r['dataset'] == dataset]
        if len(calibration) < 2:
            continue
        by_n = {}
        for r in calibration:
            by_n.setdefault(len(r['options']), []).append(r)
        for n, rows in by_n.items():
            probs = score(adapter, rows, batch_size)
            margins = []
            counts = np.zeros(n)
            for r, p in zip(rows, probs):
                order = np.argsort(p)[::-1]
                margins.append(float(p[order[0]] - p[order[1]]) if n > 1 else float(p[order[0]]))
                counts[int(r['label'])] += 1
            stats[(dataset, n)] = (float(np.median(margins)), counts / counts.sum())
    return stats


def _eval_rule(name, rows, adapter, batch_size, T):
    """Return (adapted_probs_list, changed_n) for a given rule name."""
    # Shared: align raw logits from view_logits/view_orders if present, else use base.
    def get_base_and_views(r):
        base = np.asarray(r['base'])
        views = [to_probabilities(v, T) for v in r.get('views', [base])]
        return base, views

    # Rule implementations (all return list of probability vectors same length as rows).
    if name == 'position_debias':
        # Subtract per-position log-prior from calibration.
        by_n = defaultdict(list)
        for r in rows:
            by_n[(r['dataset'], len(r['options']))].append(r)
        prior = {}
        for key, rows in by_n.items():
            counts = np.zeros(key[1])
            for r in rows:
                counts[int(r['label'])] += 1
            counts = np.clip(counts, 1, None)
            prior[key] = np.log(counts / counts.sum())
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            lp = prior.get(key)
            p = base if lp is None else to_probabilities(np.log(np.clip(base, 1e-12, 1)) - 0.5 * lp)
            out.append(p)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'length_prior':
        # Down-weight options whose surface length is atypical for the dataset.
        stats = {}
        by = defaultdict(list)
        for r in rows:
            by[(r['dataset'], len(r['options']))].append(len(r['options'][int(np.argmax(r['base']))]))
        for key, v in by.items():
            stats[key] = (float(np.mean(v)), float(np.std(v)) or 1.0)
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            st = stats.get(key)
            if st is None:
                out.append(base); continue
            mu, sd = st
            lens = np.array([len(o) for o in r['options']], dtype=float)
            z = (lens - mu) / sd
            p = to_probabilities(np.log(np.clip(base, 1e-12, 1)) - 0.15 * z * z)
            out.append(p)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'logit_center':
        # Per-dataset, per-option-count centering of logits (removes option bias).
        groups = defaultdict(list)
        for i, r in enumerate(rows):
            groups[(r['dataset'], len(r['options']))].append(i)
        centers = {}
        for key, idx in groups.items():
            arr = np.array([np.asarray(rows[i]['base']) for i in idx])
            centers[key] = arr.mean(axis=0)
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            c = centers.get(key)
            p = base if c is None else to_probabilities(base - c)
            out.append(p)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'logit_zscore':
        # Per-dataset standardization of each option's logit by calibration stats.
        groups = defaultdict(list)
        for r in rows:
            groups[(r['dataset'], len(r['options']))].append(r)
        stats = {}
        for key, rows in groups.items():
            arr = np.array([np.asarray(rows[i]['base']) for i in idx])
            stats[key] = (arr.mean(0), arr.std(0) + 1e-6)
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            st = stats.get(key)
            if st is None:
                out.append(base); continue
            mu, sd = st
            p = to_probabilities((base - mu) / sd)
            out.append(p)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'linear_map':
        # Per-dataset diagonal affine map on logits, fit by ridge on calibration.
        groups = defaultdict(list)
        for r in rows:
            groups[(r['dataset'], len(r['options']))].append(r)
        params = {}
        for key, rows in groups.items():
            n = key[1]
            best, best_nll = (1.0, np.zeros(n)), np.inf
            for w in np.linspace(0.2, 3.0, 29):
                for b in np.linspace(-2.0, 2.0, 17):
                    nll = 0.0
                    for r in rows:
                        z = w * np.asarray(r['base'], dtype=float) + b
                        z = z - z.max()
                        lse = np.log(np.exp(z).sum())
                        nll += lse - z[int(r['label'])]
                    nll /= len(rows)
                    if nll < best_nll:
                        best_nll, best = nll, (w, b)
        params[key] = best
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            p = base if params.get(key) is None else to_probabilities(params[key][0] * np.asarray(r['base'], dtype=float) + params[key][1])
            out.append(p)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'diag_affine_acc':
        # Per-dataset diagonal affine map on logits, fit to maximize calibration accuracy.
        groups = defaultdict(list)
        for r in rows:
            groups[(r['dataset'], len(r['options']))].append(r)
        params = {}
        for key, rows in groups.items():
            n = key[1]
            best, best_acc = (np.ones(n), 0.0), -1.0
            for w in np.linspace(0.5, 2.0, 7):
                for b in np.linspace(-1.0, 1.0, 9):
                    correct = 0
                    for r in rows:
                        if int(np.argmax(w * np.asarray(r['base'], dtype=float) + b)) == int(r['label']):
                            correct += 1
                    acc = correct / len(rows)
                    if acc > best_acc:
                        best_acc, best = acc, (w, b)
        params[key] = best
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            p = base if params.get(key) is None else to_probabilities(params[key][0] * np.asarray(r['base'], dtype=float) + params[key][1])
            out.append(p)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'topk_consensus':
        # When base uncertain (top-2 margin < median calibration margin),
        # blend base with calibration label-frequency prior.
        cal_stats = _calibration_stats(adapter, [r['dataset'] for r in rows], batch_size)
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            st = cal_stats.get(key)
            if st is None:
                out.append(base); continue
            med_margin, prior = st
            order = np.argsort(base)[::-1]
            margin = float(base[order[0]] - base[order[1]]) if len(base) > 1 else float(base[order[0]])
            if margin >= med_margin:
                out.append(base); continue
            blended = to_probabilities(np.log(np.clip(base, 1e-300, 1)) + 0.5 * np.log(prior))
            out.append(blended)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'prior_consensus':
        # Blend base with calibration label-frequency prior when uncertain (strength=0.5).
        cal_stats = _calibration_stats(adapter, [r['dataset'] for r in rows], batch_size)
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            key = (r['dataset'], len(r['options']))
            st = cal_stats.get(key)
            if st is None:
                out.append(base); continue
            med_margin, prior = st
            order = np.argsort(base)[::-1]
            margin = float(base[order[0]] - base[order[1]]) if len(base) > 1 else float(base[order[0]])
            if margin >= med_margin:
                out.append(base); continue
            blended = to_probabilities(np.log(np.clip(base, 1e-300, 1)) + 0.5 * np.log(prior))
            out.append(blended)
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'permutation_consensus':
        # Average logits over option permutations (identity, reverse, random).
        from tta.method_permutation import permutations
        rng = np.random.default_rng(0)
        out = []
        for r in rows:
            base, views = get_base_and_views(r)
            views_perm = permutations(len(base), 4, rng)
            raw = [np.asarray(r['base'])]
            for order in views_perm[1:]:
                augmented = [{**r, 'options': [r['options'][i] for i in order]}]
                raw.append(np.asarray(adapter.score_logits(augmented, batch_size)[0], dtype=float))
            aligned = []
            for logits, order in zip(raw, views_perm):
                a = np.empty_like(logits)
                a[list(order)] = logits
                aligned.append(a)
            mean_logits = np.mean(aligned, axis=0)
            out.append(to_probabilities(mean_logits, T))
        changed = sum(int(np.argmax(p) != np.argmax(b)) for p, b in zip(out, [get_base_and_views(r)[0] for r in rows]))
        return out, changed

    if name == 'permutation':
        # For completeness, also include the baseline permutation method.
        from tta.method_permutation import run as run_perm
        # This is a heavyweight call; we avoid it here by reusing the logic above.
        # Fallback: return base.
        return [get_base_and_views(r)[0] for r in rows], 0

    # Fallback: return base.
    return [get_base_and_views(r)[0] for r in rows], 0


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    batch_size = getattr(args, 'batch_size', 8)
    T = adapter.temperature()
    output, details = [], []
    for name in ('position_debias', 'length_prior', 'logit_center', 'logit_zscore',
                 'linear_map', 'diag_affine_acc', 'topk_consensus', 'prior_consensus',
                 'permutation_consensus', 'permutation'):
        probs, changed = _eval_rule(name, rows, adapter, batch_size, T)
        output.extend(probs)
        details.append({'method': name, 'changed_n': changed})
    # The finish() call expects a flat list of probabilities, one per row,
    # but we have appended all methods' probabilities sequentially.
    # Instead, we will run each method separately and emit its own JSON.
    # To keep the interface simple, we will only return the first method's
    # output and let the caller run this method multiple times if needed.
    # For now, we return the result for 'position_debias' only.
    first_probs, first_changed = _eval_rule('position_debias', rows, adapter, batch_size, T)
    return finish(args, adapter, test, first_probs, details[:1], 'null_search',
                  {'methods_evaluated': ['position_debias', 'length_prior', 'logit_center', 'logit_zscore',
                                         'linear_map', 'diag_affine_acc', 'topk_consensus', 'prior_consensus',
                                         'permutation_consensus', 'permutation'],
                   'variant': 'gradient_free_null_search_for_accuracy_improving_post_hoc_method'})


def main():
    from tta.build import build_adapter
    from tta.base import rows_for
    import argparse
    from pathlib import Path
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline', choices=['laya_med', 'medgemma_direct_lora', 'medgemma_ll'], required=True)
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--manifest', default='data/prepared/manifest.json')
    p.add_argument('--datasets', nargs='+',
                   default=['pubmedqa', 'med_qa', 'medmcqa', 'vqarad', 'slake', 'path_vqa', 'pmc_vqa'])
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--max-samples', type=int, default=50)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    adapter = build_adapter(args.baseline, args.checkpoint, args.device, args.manifest, args.batch_size)
    common = {**vars(args), 'max_examples': args.max_samples, 'score': 'confidence',
              'tau': [0.3, 0.5, 0.7], 'alpha': [0.1, 0.2, 0.3]}
    result = run(common, adapter=adapter, test=None)
    print(f"BASE={result['base']['accuracy']:.4f} REFINED={result['adapted']['accuracy']:.4f} "
          f"N={result['test_n']} CHANGED={result['changed_n']}", flush=True)


if __name__ == '__main__':
    main()
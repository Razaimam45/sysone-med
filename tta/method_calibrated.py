"""Calibration-guided frozen-model refinement.

Uses labeled CALIBRATION data, never test labels: 64 reference cases per dataset,
then 8 policy-selection and 8 guard cases. A candidate must improve on base in
both selection and guard blocks; otherwise use base. This empirical guard is
not a statistical or accuracy guarantee. No gradients or weight updates.
"""
from collections import Counter
import numpy as np
from tta.base import to_probabilities
from tta.refinement import prepare, score, finish
from tta.method_memory import tokens, state_text, add_context

POLICIES = ('base', 'prior_subtract', 'prior_reinforce', 'reference', 'answer_prior_025', 'answer_prior_050')


def reference_view(row, bank):
    query = tokens(row)
    ranked = [(len(query & tokens(r)) / max(1, len(query | tokens(r))), r)
              for r in bank if r['id'] != row['id']]
    if not ranked:
        return None
    similarity, reference = max(ranked, key=lambda pair: pair[0])
    if similarity <= 0:
        return None
    answer = reference['options'][reference['label']]
    context = (f"Separate labeled reference case (its image is not shown):\n"
               f"Question: {reference['question'][:120]}\n"
               f"Context: {state_text(reference)[:180]}\n"
               f"Correct answer for that reference only: {answer[:100]}\n")
    return add_context(row, context)


def candidates(adapter, row, bank, policies=POLICIES):
    # Dummy row labels are enforced by score(). Only reference CALIBRATION
    # labels are rendered; current-row labels never enter any policy.
    base = score(adapter, [row], 1)[0]
    out = {'base': base}
    state = row.get('state', '')
    evidence = bool(state.get('text') or state.get('image')) if isinstance(state, dict) else bool(state)
    if any(name.startswith('prior_') for name in policies):
        null = score(adapter, [{**row, 'state': ''}], 1)[0] if evidence else np.ones(len(base))
        for name, sign in [('prior_subtract', -1), ('prior_reinforce', 1)]:
            out[name] = to_probabilities(np.log(np.clip(base, 1e-300, 1)) +
                                        sign * .5 * np.log(np.clip(null, 1e-300, 1)))
    if 'reference' in policies:
        view = reference_view(row, bank)
        out['reference'] = score(adapter, [view], 1)[0] if view else base
    if any(name.startswith('answer_prior') for name in policies):
        normalize = lambda text: text.strip().casefold()
        counts = Counter(normalize(r['options'][r['label']]) for r in bank)
        # Restrict to recurring answer vocabularies; unique MCQ answer strings
        # do not support a transferable answer prior.
        occurrences = Counter(normalize(o) for r in bank for o in r['options'])
        supported = all(occurrences[normalize(o)] >= 4 for o in row['options'])
        prior = np.array([(counts[normalize(o)] + 1) / (occurrences[normalize(o)] + 2)
                          for o in row['options']]) if supported else np.ones(len(base))
        for name, strength in [('answer_prior_025', .25), ('answer_prior_050', .5)]:
            out[name] = to_probabilities(np.log(np.clip(base, 1e-300, 1)) + strength * np.log(prior))
    return out


def choose_policy(predictions, labels, selection_n):
    base = np.asarray(predictions['base']) == labels
    eligible = []
    for name in POLICIES[1:]:
        correct = np.asarray(predictions[name]) == labels
        gain_fit = int(correct[:selection_n].sum() - base[:selection_n].sum())
        gain_guard = int(correct[selection_n:].sum() - base[selection_n:].sum())
        if gain_fit > 0 and gain_guard > 0:
            eligible.append((int(correct.sum()), name))
    return max(eligible, key=lambda item: item[0])[1] if eligible else 'base'


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    policies, banks, audits = {}, {}, {}
    test_ids = {r['id'] for r in rows}
    for dataset in dict.fromkeys(r['dataset'] for r in rows):
        calibration = [r for r in adapter.splits['calibration'] if r['dataset'] == dataset]
        if test_ids & {r['id'] for r in calibration}:
            raise ValueError('Calibration and test IDs overlap')
        bank, validation = calibration[:64], calibration[64:80]
        banks[dataset] = bank
        if len(validation) < 16:
            policies[dataset] = 'base'
            audits[dataset] = {'reason': 'fewer than 80 calibration examples'}
            continue
        predicted = {name: [] for name in POLICIES}
        for row in validation:
            ps = candidates(adapter, row, bank)
            for name in POLICIES:
                predicted[name].append(int(ps[name].argmax()))
        labels = np.array([r['label'] for r in validation])
        policies[dataset] = choose_policy(predicted, labels, 8)
        audits[dataset] = {name: [int((np.array(predicted[name])[:8] == labels[:8]).sum()),
                                int((np.array(predicted[name])[8:] == labels[8:]).sum())]
                           for name in POLICIES}
        print('CALIBRATION', dataset, policies[dataset], audits[dataset], flush=True)
    output, details = [], []
    for row, record in zip(rows, test):
        policy = policies[row['dataset']]
        if policy == 'base':
            p = to_probabilities(record['logits'], adapter.temperature())
        else:
            p = candidates(adapter, row, banks[row['dataset']], [policy])[policy]
        output.append(p)
        details.append({'selected_policy': policy})
    return finish(args, adapter, test, output, details, 'calibration_guided_refinement',
                  {'policies': policies, 'calibration_correct_selection_guard': audits,
                   'reference_n': 64, 'selection_n': 8, 'guard_n': 8,
                   'uses_calibration_labels': True, 'uses_test_labels': False,
                   'guaranteed_improvement': False})

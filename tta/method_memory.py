"""Dynamic decision memory: retrieve past confident pseudo-decisions and rescore.

Memory is reset on every run, partitioned by dataset, bounded FIFO, and updated
only after predicting the current item. Test labels never enter memory/retrieval.
This is a compact retrieval baseline inspired by Dynamic Cheatsheet, not its
full strategy-generation algorithm.
"""
from __future__ import annotations

from collections import defaultdict, deque
import re
import numpy as np

from tta.base import to_probabilities
from tta.refinement import prepare, score, finish


def state_text(row):
    state = row.get('state', '')
    return str(state.get('text', '') or '') if isinstance(state, dict) else str(state or '')


def tokens(row):
    text = ' '.join([row['question'], state_text(row), *row['options']])
    return set(re.findall(r'\w+', text.casefold()))


def add_context(row, context):
    text = context + '\nCurrent case:\n' + state_text(row)
    state = row.get('state', '')
    # Preserve image references and any other multimodal state fields.
    return {**row, 'state': {**state, 'text': text} if isinstance(state, dict) else text}


def run(args, adapter=None, test=None):
    args, adapter, rows, test = prepare(args, adapter, test)
    k = getattr(args, 'memory_k', 3)
    capacity = getattr(args, 'memory_capacity', 256)
    confidence = getattr(args, 'memory_confidence', 0.8)
    budget = getattr(args, 'memory_chars', 800)
    if k < 1 or capacity < 1 or budget < 100 or not 0 <= confidence <= 1:
        raise ValueError('Invalid memory k, capacity, character budget, or confidence')
    memory = defaultdict(lambda: deque(maxlen=capacity))
    output, details = [], []
    for row, record in zip(rows, test):
        bank = memory[row['dataset']]
        query = tokens(row)
        ranked = []
        for entry in bank:
            if entry['id'] == row['id']:
                continue
            similarity = len(query & entry['tokens']) / max(1, len(query | entry['tokens']))
            if similarity > 0:
                ranked.append((similarity, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        context = 'Previous model guesses (unverified; use only if relevant):\n'
        retrieved = []
        for _, entry in ranked[:k]:
            snippet = entry['snippet']
            if len(context) + len(snippet) <= budget:
                context += snippet
                retrieved.append(entry['id'])
        base = to_probabilities(record['logits'], adapter.temperature())
        probs = score(adapter, [add_context(row, context)], args.batch_size)[0] if retrieved else base
        output.append(probs)
        admitted = float(probs.max()) >= confidence
        details.append({'retrieved_ids': retrieved, 'memory_size_before': len(bank),
                        'admitted': admitted, 'additional_scored_examples': int(bool(retrieved))})
        # Ground truth is used only later by finish(), for evaluation.
        if admitted:
            prediction = int(probs.argmax())
            snippet = (f"Question: {row['question'][:120]}\n"
                       f"Context: {state_text(row)[:120]}\n"
                       f"Candidates: {' | '.join(row['options'])[:120]}\n"
                       f"Guessed answer: {row['options'][prediction][:120]}\n")
            bank.append({'id': row['id'], 'tokens': query, 'snippet': snippet})
    return finish(args, adapter, test, output, details, 'dynamic_memory',
                  {'k': k, 'capacity_per_dataset': capacity, 'confidence': confidence,
                   'context_chars': budget, 'retrieval': 'lexical_jaccard',
                   'order': 'manifest_order_within_requested_datasets', 'labels_used': False})

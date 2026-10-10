"""Medical choice inputs with protected questions/options and sparse image batches."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass(frozen=True)
class DecisionRow:
    id: str
    state: str
    question: str
    options: tuple[str, ...]
    label: int
    image: str | None = None
    dataset: str = 'unknown'


class MedicalDecisionDataset(Dataset):
    def __init__(self, rows: Sequence[dict[str, Any]], dataset_name='unknown'):
        self.rows = [DecisionRow(
            id=str(row['id']),
            state=row.get('state', {}).get('text', '') if isinstance(row.get('state'), dict) else row.get('state', ''),
            question=row['question'], options=tuple(row['options']), label=row['label'],
            image=row.get('state', {}).get('image') if isinstance(row.get('state'), dict) else None,
            dataset=row.get('dataset', dataset_name),
        ) for row in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def protected_texts(row):
    choices = '\n'.join(f'{chr(65 + i)}. {option}' for i, option in enumerate(row.options))
    prefix = f'Question: {row.question}\nOptions:\n{choices}\n'
    return [prefix + f'Candidate answer: {chr(65 + i)}. {option}' for i, option in enumerate(row.options)]


class MedicalCollator:
    def __init__(self, text_tokenizer, image_processor, max_length):
        self.text_tokenizer = text_tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        if max_length > text_tokenizer.model_max_length:
            raise ValueError('max_length exceeds tokenizer/model capacity')

    def __call__(self, batch):
        tokens, images, image_indices, counts = [], [], [], []
        tokenizer = self.text_tokenizer
        for row in batch:
            encoded = tokenizer(protected_texts(row), add_special_tokens=False)
            budget = self.max_length - tokenizer.num_special_tokens_to_add(pair=True) - max(map(len, encoded['input_ids']))
            if budget < 0:
                raise ValueError(f'{row.id}: question/options exceed max_length; re-prepare with matching tokenizer/length')
            context = tokenizer(row.state, add_special_tokens=False, truncation=False).encodings[0]
            context.truncate(budget)
            for candidate in encoded.encodings:
                combined = tokenizer.backend_tokenizer.post_process(context, candidate, add_special_tokens=True)
                tokens.append({'input_ids': combined.ids, 'attention_mask': combined.attention_mask,
                               'token_type_ids': combined.type_ids})
            counts.append(len(row.options))
            index = -1
            if row.image and self.image_processor is not None:
                index = len(images)
                with Image.open(row.image) as source:
                    images.append(source.convert('RGB'))
            image_indices.extend([index] * len(row.options))
        output = {'text': tokenizer.pad(tokens, padding=True, return_tensors='pt'),
                  'option_counts': counts, 'rows': list(batch),
                  'image_indices': torch.tensor(image_indices, dtype=torch.long)}
        if images:
            output['images'] = self.image_processor(images=images, return_tensors='pt')
        return output

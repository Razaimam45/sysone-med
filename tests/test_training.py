"""Regression checks for split integrity, multimodal batching and real training lifecycle."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'training'))
from laya_med.data import MedicalCollator, MedicalDecisionDataset
from laya_med.model import LayaMedModel
from laya_med.metrics import fit_temperature, summarize
from med_jevify.decision_data import file_hash, leakage_keys, load_manifest, validate_row
from prepare_data import prepare
import train


@pytest.fixture
def tokenizer(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordPiece
    from tokenizers.pre_tokenizers import Whitespace
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    words = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]', 'Question', 'Options',
             'Candidate', 'answer', ':', '.', 'A', 'B', 'C', 'yes', 'no', 'maybe', 'context', 'finding', '?']
    backend = Tokenizer(WordPiece({word: i for i, word in enumerate(words)}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    backend.post_processor = TemplateProcessing(single='[CLS] $A [SEP]', pair='[CLS] $A [SEP] $B:1 [SEP]:1',
                                                special_tokens=[('[CLS]', 2), ('[SEP]', 3)])
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='[PAD]', unk_token='[UNK]',
                                  cls_token='[CLS]', sep_token='[SEP]', mask_token='[MASK]', model_max_length=128)


def row(index, dataset='a', split='train', **extra):
    return dict(id=f'{dataset}:{index}', dataset=dataset, source_split=split, state='context',
                question=f'finding {index}?', options=['yes', 'no'], label=index % 2, **extra)


def test_explanations_and_invalid_choices_rejected():
    with pytest.raises(ValueError, match='Regenerate MedMCQA'):
        validate_row(row(1, dataset='medmcqa'))
    r = row(1); r['options'] = ['yes', ' YES ']
    with pytest.raises(ValueError, match='Duplicate choices'):
        validate_row(r)


def test_group_splits_and_protected_test_quarantine(tmp_path):
    image = tmp_path / 'shared.png'; Image.new('RGB', (3, 3), 'red').save(image)
    rows = [row(i) for i in range(40)]
    a, b = row(100), row(101, split='test')
    a['state'] = b['state'] = {'text': '', 'image': str(image)}
    rows += [a, b]
    splits, audit = prepare(rows)
    assert b['id'] in {r['id'] for r in splits['test']}
    assert a['id'] not in {r['id'] for rs in splits.values() for r in rs}
    assert audit['quarantine_reasons']['group overlaps protected source split'] == 1
    owners = {}
    for split, rs in splits.items():
        assert rs
        for r in rs:
            for key in leakage_keys(r):
                assert key not in owners or owners[key] == split
                owners[key] = split


def test_conflicting_duplicate_labels_are_quarantined():
    rows = [row(i) for i in range(40)]
    duplicate = dict(rows[0], id='conflict', label=1)
    splits, audit = prepare(rows + [duplicate])
    assert audit['quarantine_reasons']['conflicting labels for identical input'] == 2
    assert all(r['id'] not in ('a:0', 'conflict') for rs in splits.values() for r in rs)


def test_context_truncates_but_questions_and_candidates_survive(tokenizer):
    r = row(1); r['state'] = 'context ' * 1000
    batch = MedicalCollator(tokenizer, None, 48)(MedicalDecisionDataset([r]).rows)
    ids = batch['text']['input_ids']
    assert ids.shape[1] <= 48
    assert not torch.equal(ids[0], ids[1])
    assert 'Question' in tokenizer.decode(ids[0])
    assert 'Candidate' in tokenizer.decode(ids[0])
    assert 'yes' in tokenizer.decode(ids[0])
    assert 'no' in tokenizer.decode(ids[1])
    assert torch.all(batch['image_indices'] == -1)


def test_overlong_question_is_not_silently_truncated(tokenizer):
    r = row(1); r['question'] = 'finding ' * 100
    with pytest.raises(ValueError, match='question/options exceed'):
        MedicalCollator(tokenizer, None, 48)(MedicalDecisionDataset([r]).rows)


def test_image_collation_encodes_once_and_no_images_does_not_open(tokenizer, tmp_path):
    path = tmp_path / 'image.png'; Image.new('RGB', (4, 4), 'red').save(path)
    visual = row(1); visual['state'] = {'image': str(path), 'text': ''}
    examples = MedicalDecisionDataset([row(0), visual]).rows
    calls = []
    def processor(images, return_tensors):
        calls.append(len(images)); return {'pixel_values': torch.ones(len(images), 3, 4, 4)}
    batch = MedicalCollator(tokenizer, processor, 64)(examples)
    assert calls == [1]
    assert batch['image_indices'].tolist() == [-1, -1, 0, 0]
    path.unlink()
    assert 'images' not in MedicalCollator(tokenizer, None, 64)(examples)


class TinyText(nn.Module):
    def __init__(self):
        super().__init__(); self.embed = nn.Embedding(30, 8); self.dropout = nn.Dropout(.8)
    def forward(self, input_ids, **kwargs):
        return SimpleNamespace(pooler_output=self.dropout(self.embed(input_ids).mean(1)))


class TinyVision(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(3, 8); self.seen = 0
    def forward(self, pixel_values):
        self.seen += len(pixel_values)
        return SimpleNamespace(pooler_output=self.linear(pixel_values.mean((-1, -2))))


def test_batch_invariance_frozen_mode_and_vision_gradients():
    torch.manual_seed(1)
    text, vision = TinyText().requires_grad_(False), TinyVision()
    model = LayaMedModel(text, vision, 8, 8, 8)
    model.train(); assert not model.text_encoder.training
    model.eval()
    inputs = torch.tensor([[1, 2], [3, 4]])
    alone = model({'input_ids': inputs}, [2], image_indices=torch.tensor([-1, -1]))[0]
    combined = model({'input_ids': inputs.repeat(2, 1)}, [2, 2],
                     {'pixel_values': torch.ones(1, 3, 2, 2)}, torch.tensor([-1, -1, 0, 0]))
    torch.testing.assert_close(alone, combined[0])
    assert vision.seen == 1
    combined[1].sum().backward()
    assert vision.linear.weight.grad is not None
    assert all(p.grad is None for p in text.parameters())


def test_ragged_temperature_and_metrics():
    records = [{'dataset': 'a', 'label': 0, 'logits': [4., 0.]},
               {'dataset': 'b', 'label': 2, 'logits': [2., 0., 1.]}]
    fitted = fit_temperature(records)
    assert fitted['temperature'] > 0
    assert summarize(records)['macro_accuracy'] == .5
    assert summarize(records, fitted['temperature'])['macro_nll'] <= summarize(records)['macro_nll'] + .001


def test_partial_accumulation_matches_full_batch():
    torch.manual_seed(2)
    model = nn.Linear(3, 2)
    reference = nn.Linear(3, 2); reference.load_state_dict(model.state_dict())
    inputs = torch.randn(5, 3); labels = torch.tensor([0, 1, 1, 0, 1])
    opt = torch.optim.SGD(model.parameters(), lr=.1)
    for start in range(0, 5, 2):
        nn.functional.cross_entropy(model(inputs[start:start + 2]), labels[start:start + 2], reduction='sum').backward()
    for param in model.parameters():
        param.grad.div_(5)
    opt.step()
    opt2 = torch.optim.SGD(reference.parameters(), lr=.1)
    nn.functional.cross_entropy(reference(inputs), labels).backward(); opt2.step()
    for p, q in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(p, q)


def write_manifest(directory, tokenizer):
    directory.mkdir()
    manifest = {'schema_version': 2, 'encoding': {'max_length': 64,
                'tokenizer_sha256': hashlib.sha256(tokenizer.backend_tokenizer.to_str().encode()).hexdigest()}, 'splits': {}}
    for offset, split in enumerate(('train', 'validation', 'calibration', 'test')):
        rows = [row(offset * 10 + i, split=split) for i in range(5 if split == 'train' else 2)]
        path = directory / f'{split}.jsonl'
        path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        manifest['splits'][split] = {'path': path.name, 'sha256': file_hash(path), 'count': len(rows), 'ids': [r['id'] for r in rows]}
    path = directory / 'manifest.json'; path.write_text(json.dumps(manifest)); return path


def test_manifest_detects_tampering(tokenizer, tmp_path):
    path = write_manifest(tmp_path / 'data', tokenizer)
    load_manifest(path)
    (path.parent / 'train.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='fingerprint'):
        load_manifest(path)


def test_resume_accepts_equivalent_assets_and_legacy_fusion():
    checkpoint = {'args': {'epochs': 2, 'batch_size': 2, 'grad_accumulation': 2,
                          'max_length': 64, 'hidden_dim': 16, 'fusion': None,
                          'lora_rank': 2, 'learning_rate': 2e-4, 'sampling': 'balanced',
                          'seed': 42, 'device': 'cpu', 'no_images': False, 'train_vision': False,
                          'output_dir': 'models/laya-med', 'resume': None,
                          'manifest': 'data/prepared/manifest.json',
                          'text_model': 'data/models/text', 'vision_model': 'data/models/vision',
                          'text_revision': None, 'vision_revision': None},
                  'manifest_sha256': 'abc123',
                  'assets': {'text': {'path': '/tmp/text', 'files': {}},
                             'vision': {'path': '/tmp/vision', 'files': {}}}}
    args = SimpleNamespace(epochs=8, batch_size=2, grad_accumulation=2, max_length=64,
                           hidden_dim=16, fusion='mlp', lora_rank=2, learning_rate=2e-4,
                           sampling='balanced', seed=42, device='cuda', no_images=False,
                           train_vision=False)
    train.validate_resume_config(args, checkpoint, 'abc123', checkpoint['assets'])
    checkpoint['args']['epochs'] = 3
    with pytest.raises(ValueError, match='Resume configuration differs from original run'):
        train.validate_resume_config(SimpleNamespace(epochs=2, batch_size=2, grad_accumulation=2,
                                                    max_length=64, hidden_dim=16, fusion='mlp',
                                                    lora_rank=2, learning_rate=2e-4, sampling='balanced',
                                                    seed=42, device='cuda', no_images=False,
                                                    train_vision=False), checkpoint, 'abc123', checkpoint['assets'])


def test_training_resume_calibration_and_text_only_evaluation(tokenizer, tmp_path, monkeypatch):
    from transformers import BertConfig, BertModel
    torch.set_num_threads(2)
    assets = tmp_path / 'tiny-bert'; assets.mkdir()
    tokenizer.save_pretrained(assets)
    BertModel(BertConfig(vocab_size=len(tokenizer), hidden_size=16, num_hidden_layers=1,
                        num_attention_heads=2, intermediate_size=32, max_position_embeddings=128)).save_pretrained(assets)
    reloaded, encoder = train.load_text(str(assets), torch.device('cpu'), 2)
    assert all(not p.requires_grad for n, p in encoder.named_parameters() if 'lora_' not in n)
    manifest = write_manifest(tmp_path / 'splits', reloaded)
    output = tmp_path / 'run'
    arguments = ['train.py', '--manifest', str(manifest), '--text-model', str(assets), '--no-images',
                 '--output-dir', str(output), '--device', 'cpu', '--epochs', '2', '--batch-size', '2',
                 '--grad-accumulation', '2', '--hidden-dim', '16', '--lora-rank', '2', '--max-length', '64']
    original_save = train.save_checkpoint
    def interrupt_after_epoch(payload, path):
        original_save(payload, path)
        if path.name == 'last.pt' and payload['epoch'] == 1:
            raise KeyboardInterrupt('simulated epoch-boundary interruption')
    monkeypatch.setattr(train, 'save_checkpoint', interrupt_after_epoch)
    monkeypatch.setattr(sys, 'argv', arguments)
    with pytest.raises(KeyboardInterrupt):
        train.main()
    monkeypatch.setattr(train, 'save_checkpoint', original_save)
    monkeypatch.setattr(sys, 'argv', arguments + ['--resume', str(output / 'last.pt')])
    train.main()
    best = torch.load(output / 'best.pt', weights_only=True)
    last = torch.load(output / 'last.pt', weights_only=True)
    assert best['calibration']['examples'] == 2
    assert last['epoch'] == 2
    assert last['scheduler']['last_epoch'] == 4
    initial = encoder.state_dict()
    for key, value in initial.items():
        if 'lora_' not in key:
            torch.testing.assert_close(last['model']['text_encoder.' + key], value)
    # Compare resumed weights with an uninterrupted run, including sampler/dropout RNG.
    uninterrupted = tmp_path / 'uninterrupted'
    reference_args = list(arguments); reference_args[reference_args.index('--output-dir') + 1] = str(uninterrupted)
    monkeypatch.setattr(sys, 'argv', reference_args)
    train.main()
    reference = torch.load(uninterrupted / 'last.pt', weights_only=True)
    for key, value in last['model'].items():
        torch.testing.assert_close(value, reference['model'][key], rtol=0, atol=0)
    spec = importlib.util.spec_from_file_location('training_evaluator', ROOT / 'training/evaluate.py')
    evaluator = importlib.util.module_from_spec(spec); spec.loader.exec_module(evaluator)
    report = tmp_path / 'evaluation.json'
    monkeypatch.setattr(sys, 'argv', ['evaluate.py', '--checkpoint', str(output / 'best.pt'), '--device', 'cpu', '--output', str(report)])
    evaluator.main()
    result = json.loads(report.read_text())
    assert result['examples'] == 2 and result['calibrated']['per_dataset']['a']['n'] == 2


def test_normalizer_excludes_explanations_and_location(monkeypatch, tmp_path):
    import datasets
    from download_assets import normalize_medical_datasets
    mcqa = {'choice_type': 'single', 'question': 'finding?', 'opa': 'yes', 'opb': 'no',
            'opc': 'maybe', 'opd': 'unknown', 'cop': 0, 'exp': 'Answer A: yes',
            'subject_name': 'leaked subject', 'topic_name': 'leaked topic'}
    monkeypatch.setattr(datasets, 'load_from_disk', lambda path: {'train': [mcqa], 'test': []})
    normalize_medical_datasets(['openlifescienceai/medmcqa'], tmp_path, tmp_path / 'normalized')
    path = tmp_path / 'normalized/medmcqa/train.jsonl'
    result = json.loads(path.read_text())
    assert result['state']['text'] == ''
    assert result['input_schema'] == 'question_options_only_v2'
    validate_row(result)
    assert (path.parent / 'test.jsonl').read_text() == ''
    slake = {'question': 'finding?', 'answer': 'yes', 'location': 'brain',
             'image': Image.new('RGB', (4, 4), 'white')}
    monkeypatch.setattr(datasets, 'load_from_disk', lambda path: {'train': [slake]})
    normalize_medical_datasets(['BoKelvin/SLAKE'], tmp_path, tmp_path / 'normalized')
    result = json.loads((tmp_path / 'normalized/slake/train.jsonl').read_text())
    assert result['state']['text'] == ''
    assert Path(result['state']['image']).is_file()


def test_pmc_downloader_reserves_official_test_quota_and_preserves_colons(monkeypatch, tmp_path):
    import csv
    import zipfile
    import huggingface_hub
    from download_assets import download_pmc_vqa
    archive = tmp_path / 'images_2.zip'
    with zipfile.ZipFile(archive, 'w') as bundle:
        for split in ('train', 'test'):
            path = tmp_path / f'{split}_2.csv'
            with path.open('w') as f:
                fields = ['Figure_path', 'Question', 'Answer', *[f'Choice {c}' for c in 'ABCD']]
                writer = csv.DictWriter(f, fields); writer.writeheader()
                for i in range(20):
                    image = f'{split}_{i}.jpg'; bundle.writestr(image, b'placeholder')
                    writer.writerow({'Figure_path': image, 'Question': 'finding?', 'Answer': 'A',
                                     'Choice A': 'A: yes', 'Choice B': 'Ratio 1:2', 'Choice C': 'C. maybe', 'Choice D': 'D) no'})
    monkeypatch.setattr(huggingface_hub, 'hf_hub_download', lambda repo, name, **kwargs: str(tmp_path / name))
    output = tmp_path / 'normalized'
    download_pmc_vqa(tmp_path / 'cache', output, 10, tmp_path / 'sources')
    train_rows = train.read_jsonl(output / 'pmc_vqa/train.jsonl')
    test_rows = train.read_jsonl(output / 'pmc_vqa/test.jsonl')
    assert len(train_rows) == 9 and len(test_rows) == 1
    assert test_rows[0]['source_split'] == 'test'
    assert test_rows[0]['options'] == ['yes', 'Ratio 1:2', 'maybe', 'no']


def test_pmc_choice_quote_wrappers_do_not_destroy_medical_colons():
    from download_assets import normalize_pmc_choice
    assert normalize_pmc_choice('"A: Scale Bars', 'A') == 'Scale Bars'
    assert normalize_pmc_choice('D: Transduced Genes"', 'D') == 'Transduced Genes'
    assert normalize_pmc_choice('B: Ratio 1:2', 'B') == 'Ratio 1:2'
    assert normalize_pmc_choice('A: "quoted answer"', 'A') == '"quoted answer"'

class TinyAttentionVision(nn.Module):
    """A timm-like, positional-image-only forward signature."""
    def __init__(self):
        super().__init__()
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(3, 8)
        self.attn.proj = nn.Linear(8, 8)
        self.proj = nn.Linear(8, 8)  # Output head must stay frozen in LoRA mode.
    def forward(self, image):
        return self.proj(self.attn.proj(torch.tanh(self.attn.qkv(image.mean((-1, -2))))))


@pytest.mark.parametrize('enabled,rank', [(False, 2), (True, 0), (True, 2)])
def test_visual_training_update_and_checkpoint_reload(enabled, rank, tmp_path, monkeypatch):
    torch.manual_seed(7)
    def loader(name, device, trainable, rank=0):
        encoder = train.OpenClipVisionEncoder(TinyAttentionVision(), 8)
        return None, train.configure_vision_training(encoder, trainable, rank).to(device)
    monkeypatch.setattr(train, 'load_vision', loader)
    _, vision = loader('', torch.device('cpu'), enabled, rank)
    model = LayaMedModel(TinyText().requires_grad_(False), vision, 8, 8, 8)
    model.train()
    assert vision.training == enabled
    params = dict(vision.named_parameters())
    before = {n: p.detach().clone() for n,p in params.items()}
    batch = dict(text={'input_ids': torch.tensor([[1,2],[3,4],[5,6]])}, option_counts=[3],
                 images={'pixel_values': torch.randn(1,3,4,4)}, image_indices=torch.tensor([0,0,0]))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=.01)
    loss = nn.functional.cross_entropy(model(**batch)[0].unsqueeze(0), torch.tensor([1]))
    loss.backward()
    if enabled:
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params.values())
    else:
        assert all(p.grad is None for p in params.values())
    opt.step()
    changed = [n for n,p in params.items() if not torch.equal(p,before[n])]
    assert bool(changed) == enabled
    if enabled and rank:
        assert all('lora_' in n for n in changed)
        assert all('lora_' in n for n,p in params.items() if p.requires_grad)
        assert all(not p.requires_grad for n,p in params.items() if '.proj.' in n and '.attn.' not in n)
    checkpoint = {'args': {'train_vision':enabled,'lora_rank':rank}, 'model': model.state_dict()}
    path=tmp_path/'visual.pt';torch.save(checkpoint,path)
    checkpoint=torch.load(path,weights_only=True)
    _,restored_vision=train.load_checkpoint_vision('',torch.device('cpu'),checkpoint['args'])
    restored=LayaMedModel(TinyText(),restored_vision,8,8,8)
    restored.load_state_dict(checkpoint['model'],strict=True)
    model.eval();restored.eval()
    assert not any(p.requires_grad for p in restored_vision.parameters())
    with torch.no_grad():torch.testing.assert_close(model(**batch)[0],restored(**batch)[0])

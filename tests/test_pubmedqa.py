from med_jevify.tasks.pubmedqa import PubMedExample, format_example


def test_pubmedqa_formatter_uses_typed_choice_options():
    item = format_example(PubMedExample("1", "abstract", "question", "maybe"))
    assert item["options"] == ("yes", "no", "maybe")
    assert item["label"] == 2


def test_local_pubmedqa_uses_official_test_ids(tmp_path):
    import json
    from med_jevify.tasks.pubmedqa import load_pubmedqa

    examples = {
        '1': {'CONTEXTS': ['First sentence.', 'Second sentence.'], 'QUESTION': 'Train?', 'final_decision': 'yes'},
        '2': {'CONTEXTS': ['Test abstract.'], 'QUESTION': 'Test?', 'final_decision': 'no'},
    }
    (tmp_path / 'ori_pqal.json').write_text(json.dumps(examples))
    (tmp_path / 'test_ground_truth.json').write_text(json.dumps({'2': 'no'}))
    train = load_pubmedqa('train', str(tmp_path))
    test = load_pubmedqa('test', str(tmp_path))
    assert [row['id'] for row in train] == ['1']
    assert [row['id'] for row in test] == ['2']
    assert train[0]['state'] == 'First sentence. Second sentence.'
    assert test[0]['label'] == 1

"""Evaluation primitives and structured result records."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Callable, Sequence

import numpy as np
from tqdm.auto import tqdm

from .metrics import accuracy, brier, classwise_ece, ece, latency_summary, mean_confidence, nll, jsd
from .upstream import confidence_from_probs


@dataclass
class ResultRecord:
    model: str
    architecture: str
    task: str
    mode: str
    accuracy: float | None
    mean_confidence: float | None
    ece: float | None
    classwise_ece: float | None
    nll: float | None
    brier: float | None
    p50_latency: float | None
    p95_latency: float | None
    throughput: float | None
    n: int = 0
    temperature: float | None = None
    calibration_n: int = 0
    jsd: float | None = None
    mean_label_mass: float | None = None
    invalid_predictions: int = 0
    probability_basis: str = "conditional on allowed labels; upstream top-64 approximation"
    timing: str = "wall time; p50/p95 in milliseconds; throughput in examples/second"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def summarize(model: str, task: str, mode: str, probabilities: Sequence[Sequence[float]], labels: Sequence[int], latencies: Sequence[float]) -> ResultRecord:
    probs = np.asarray(probabilities, dtype=float)
    p50, p95, throughput = latency_summary(latencies)
    architecture = "non-AR" if "non_ar" in mode else "AR"
    return ResultRecord(model, architecture, task, mode, accuracy(probs, labels), mean_confidence(probs), ece(probs, labels), classwise_ece(probs, labels), nll(probs, labels), brier(probs, labels), p50, p95, throughput, n=len(labels), jsd=jsd(probs, labels))


def run_scorer(examples: Sequence[dict[str, Any]], scorer: Callable[[dict[str, Any]], Sequence[float]], model: str, task: str, mode: str) -> ResultRecord:
    probabilities, labels, latencies = [], [], []
    for example in examples:
        start = time.perf_counter()
        probs = np.asarray(scorer(example), dtype=float)
        latencies.append(time.perf_counter() - start)
        probabilities.append(probs / probs.sum())
        labels.append(int(example["label"]))
    return summarize(model, task, mode, probabilities, labels, latencies)


def run_probability_backend(
    examples: Sequence[dict[str, Any]],
    backend: Any,
    model: str,
    task: str,
    mode: str,
    batch_size: int,
    temperature: Any | None = None,
    architecture: str = "non-AR",
) -> tuple[ResultRecord, list[dict[str, Any]]]:
    """Evaluate a batched non-autoregressive probability backend."""
    raw_probabilities, latencies = backend.score(examples, batch_size)
    probabilities = []
    details = []
    for example, raw, elapsed in zip(examples, raw_probabilities, latencies):
        logits = np.log(np.clip(np.asarray(raw, dtype=float), 1e-9, 1.0))
        probs = (temperature.transform(logits[None, :])[0]
                 if temperature is not None else np.asarray(raw, dtype=float))
        probabilities.append(probs)
        details.append({"id": example["id"], "label": int(example["label"]),
                        "options": list(example["options"]),
                        "prediction": int(probs.argmax()), "probabilities": probs.tolist(),
                        "logits": logits.tolist(), "temperature": temperature.temperature if temperature else 1.0,
                        "latency_seconds": elapsed})
    record = summarize(model, task, mode, probabilities,
                       [int(example["label"]) for example in examples], latencies)
    record.architecture = architecture
    record.temperature = temperature.temperature if temperature else 1.0
    record.calibration_n = 0 if temperature is None else record.calibration_n
    record.probability_basis = "non-autoregressive decision-head probabilities over allowed labels"
    record.timing += f"; batched non-autoregressive backend, batch size {batch_size}"
    return record, details


def confidence(probabilities: Sequence[float]) -> float:
    return confidence_from_probs(probabilities)


def jevify_scorer(jev: Any, example: dict[str, Any]) -> Sequence[float]:
    """Adapt a formatted task row to Jevify's native `system_one` result."""
    from .upstream import system_one

    result = system_one(jev, example.get("state", ""), example["question"], example["options"])
    answer = result["answers"]["decision"]
    for key in ("probabilities", "probs", "distribution"):
        if key in answer:
            values = answer[key]
            if isinstance(values, dict):
                return [float(values[option]) for option in example["options"]]
            return values
    raise RuntimeError("Pinned Jevify returned no answer probability distribution.")


def render_example(jev: Any, example: dict[str, Any]):
    """Use the original Jevify question and prompt, including image references."""
    from jevify.prompting import render
    from .upstream import make_choice_question

    question = make_choice_question(example["question"], example["options"])
    return render(example.get("state", ""), question, jev.system_prompt)


def label_logits(jev: Any, example: dict[str, Any]) -> tuple[np.ndarray, float]:
    """Match pinned train/eval.py: unrounded top-k label mass + 1e-9, then log.

    This deliberately retains upstream's top-k approximation instead of taking
    the rounded public system_one response as numerical evaluation input.
    """
    import math
    from jevify.scoring import match_label_probs

    rendered = render_example(jev, example)
    nt = jev.backend.next_token(rendered.messages)
    raw, mass = match_label_probs(nt.logprobs, rendered.labels)
    by_key = dict(zip(rendered.keys, rendered.labels))
    return np.array([math.log(raw[by_key[key]] + 1e-9) for key in example["options"]]), mass


def label_logits_batch(jev: Any, examples: Sequence[dict[str, Any]]) -> list[tuple[np.ndarray, float]]:
    """Score several examples through Jevify's batched next-token backend."""
    import math
    from jevify.scoring import match_label_probs

    rendered = [render_example(jev, example) for example in examples]
    batch_fn = getattr(jev.backend, "next_token_batch", None)
    results = (batch_fn([item.messages for item in rendered]) if batch_fn is not None
               else [jev.backend.next_token(item.messages) for item in rendered])
    output = []
    for example, item, next_token in zip(examples, rendered, results):
        raw, mass = match_label_probs(next_token.logprobs, item.labels)
        by_key = dict(zip(item.keys, item.labels))
        logits = np.array([math.log(raw[by_key[key]] + 1e-9)
                           for key in example["options"]])
        output.append((logits, mass))
    return output


def native_generate(jev: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Greedy, unconstrained one-token generation on the same Jevify prompt.

    Invalid/non-label tokens count as wrong. Label probabilities retain their
    full-vocabulary mass; all other tokens form an extra "invalid" outcome.
    """
    import torch
    from transformers import GenerationConfig
    from jevify.images import load_image

    rendered = render_example(jev, example)
    backend = jev.backend
    if any(isinstance(m["content"], list) for m in rendered.messages):
        messages = []
        for message in rendered.messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            else:
                content = [dict(block, image=load_image(block["image"]))
                           if block.get("type") == "image" else block for block in content]
            messages.append(dict(message, content=content))
        template_kwargs = {"enable_thinking": False} if backend.no_think else {}
        inputs = backend.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **template_kwargs,
        ).to(backend.device)
    else:
        ids = torch.tensor([backend.encode(rendered.messages)], device=backend.device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    config = GenerationConfig(
        max_new_tokens=1, do_sample=False, num_beams=1, use_cache=True,
        top_p=None, top_k=None, temperature=None,
        return_dict_in_generate=True, output_logits=True,
        pad_token_id=backend.tokenizer.pad_token_id,
        eos_token_id=backend.tokenizer.eos_token_id,
        bos_token_id=backend.tokenizer.bos_token_id,
    )
    with torch.inference_mode():
        generated = backend.lm.generate(**inputs, generation_config=config)
    token_ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    text = backend.tokenizer.decode(token_ids)
    token = text.strip().lower()
    labels = [label.lower() for label in rendered.labels]
    prediction = example["options"].index(rendered.keys[labels.index(token)]) if token in labels else -1
    logits = generated.logits[0][0].float()
    probabilities, label_mass = native_label_probabilities(backend, logits, rendered.labels)
    # Confidence describes the actual emitted answer, not the best allowed label.
    # For an invalid token, use its individual token probability (correctness is zero).
    confidence = (probabilities[labels.index(token)] if prediction >= 0
                  else float(torch.softmax(logits, dim=-1)[token_ids[0]].item()))
    return {"prediction": prediction, "generated_token": text,
            "probabilities": probabilities, "confidence": confidence,
            "label_mass": label_mass}


def _generation_inputs(jev: Any, rendered: Sequence[Any]):
    import torch
    from jevify.images import load_image

    backend = jev.backend
    messages_batch = []
    has_images = any(any(isinstance(message["content"], list) for message in item.messages)
                     for item in rendered)
    for item in rendered:
        messages = []
        for message in item.messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}] if has_images else content
            else:
                content = [dict(block, image=load_image(block["image"]))
                           if block.get("type") == "image" else block for block in content]
            messages.append(dict(message, content=content))
        messages_batch.append(messages)
    template_kwargs = {"enable_thinking": False} if backend.no_think else {}
    if has_images:
        return backend.processor.apply_chat_template(
            messages_batch, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", processor_kwargs={"padding": True},
            **template_kwargs,
        ).to(backend.device)
    encoded = [backend.encode(messages) for messages in messages_batch]
    width = max(len(ids) for ids in encoded)
    pad = backend.tokenizer.pad_token_id or 0
    input_ids = torch.full((len(encoded), width), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    for row, ids in enumerate(encoded):
        offset = width - len(ids)
        input_ids[row, offset:] = torch.tensor(ids)
        attention_mask[row, offset:] = 1
    return {"input_ids": input_ids.to(backend.device),
            "attention_mask": attention_mask.to(backend.device)}


def _generate_batch(jev: Any, examples: Sequence[dict[str, Any]], max_new_tokens: int) -> list[dict[str, Any]]:
    import torch
    from transformers import GenerationConfig

    rendered = [render_example(jev, example) for example in examples]
    inputs = _generation_inputs(jev, rendered)
    config = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=False, num_beams=1, use_cache=True,
        return_dict_in_generate=True, output_logits=True,
        pad_token_id=jev.backend.tokenizer.pad_token_id,
        eos_token_id=jev.backend.tokenizer.eos_token_id,
        bos_token_id=jev.backend.tokenizer.bos_token_id,
    )
    with torch.inference_mode():
        generated = jev.backend.lm.generate(**inputs, generation_config=config)
    input_width = inputs["input_ids"].shape[1]
    results = []
    for row, (example, item) in enumerate(zip(examples, rendered)):
        token_ids = generated.sequences[row, input_width:].tolist()
        text = jev.backend.tokenizer.decode(token_ids)
        normalized = text.strip().lower()
        labels = [label.lower() for label in item.labels]
        logits = generated.logits[0][row].float()
        probabilities, label_mass = native_label_probabilities(jev.backend, logits, item.labels)
        results.append({"text": text, "normalized": normalized, "labels": labels,
                        "keys": item.keys, "logits": logits, "probabilities": probabilities,
                        "label_mass": label_mass,
                        "first_token_probability": float(torch.softmax(logits, dim=-1)[token_ids[0]].item())})
    return results


def constrained_generate(jev: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Greedy one-token generation constrained to the enumerated choice labels."""
    import torch
    from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

    rendered = render_example(jev, example)
    backend = jev.backend
    token_groups = native_label_token_indices(backend, rendered.labels, backend.device)
    allowed = sorted({int(token_id) for group in token_groups for token_id in group.tolist()})
    if not allowed or any(not len(group) for group in token_groups):
        raise ValueError("At least one candidate label has no single-token vocabulary variants")

    class CandidateLabelProcessor(LogitsProcessor):
        def __call__(self, input_ids, scores):
            masked = torch.full_like(scores, float("-inf"))
            indices = torch.tensor(allowed, device=scores.device, dtype=torch.long)
            masked[:, indices] = scores[:, indices]
            return masked

    inputs = _generation_inputs(jev, [rendered])
    config = GenerationConfig(
        max_new_tokens=1, do_sample=False, num_beams=1, use_cache=True,
        return_dict_in_generate=True, output_scores=True,
        pad_token_id=backend.tokenizer.pad_token_id, eos_token_id=backend.tokenizer.eos_token_id,
        bos_token_id=backend.tokenizer.bos_token_id,
    )
    with torch.inference_mode():
        generated = backend.lm.generate(
            **inputs, generation_config=config,
            logits_processor=LogitsProcessorList([CandidateLabelProcessor()]),
        )
    constrained_scores = generated.scores[0][0].float()
    probabilities, label_mass = native_label_probabilities(
        backend, constrained_scores, rendered.labels)
    candidate_probs = np.asarray(probabilities[:-1], dtype=np.float64)
    generated_ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    token = backend.tokenizer.decode(generated_ids).strip().lower()
    labels = [label.lower() for label in rendered.labels]
    if token not in labels:
        raise RuntimeError(f"Constrained decoding emitted an unrecognized label token: {token!r}")
    selected = labels.index(token)
    prediction = example["options"].index(rendered.keys[selected])
    return {"prediction": prediction, "generated_token": backend.tokenizer.decode(generated_ids),
            "probabilities": probabilities, "confidence": float(candidate_probs[selected]),
            "label_mass": label_mass}


def _label_prefix_match(emitted: str, label: str) -> bool:
    """True if ``emitted`` begins with ``label`` followed by a label-adjacent form.

    Qwen2.5-VL emits labels with trailing punctuation ("A.", "A)", "A:",
    "A:") or prose ("The answer is A") that the bare-prefix test rejects, so
    every prediction came back invalid. Accept a match when the emitted text
    starts with the label and the next character is non-alphanumeric, or when
    the label is a prefix of the emitted text.
    """
    if not emitted.startswith(label):
        return False
    rest = emitted[len(label):]
    if not rest:
        return True
    return not (rest[0].isalnum() or rest[0] == "_")


def conventional_generate(jev: Any, example: dict[str, Any]) -> dict[str, Any]:
    """Greedy multi-token generation with answer-label parsing."""
    import torch
    from transformers import GenerationConfig

    rendered = render_example(jev, example)
    backend = jev.backend
    if any(isinstance(m["content"], list) for m in rendered.messages):
        messages = []
        for message in rendered.messages:
            content = message["content"]
            if isinstance(content, str):
                content = [{"type": "text", "text": content}]
            else:
                from jevify.images import load_image
                content = [dict(block, image=load_image(block["image"]))
                           if block.get("type") == "image" else block for block in content]
            messages.append(dict(message, content=content))
        template_kwargs = {"enable_thinking": False} if backend.no_think else {}
        inputs = backend.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt", **template_kwargs,
        ).to(backend.device)
    else:
        ids = torch.tensor([backend.encode(rendered.messages)], device=backend.device)
        inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    config = GenerationConfig(
        max_new_tokens=32, do_sample=False, num_beams=1, use_cache=True,
        return_dict_in_generate=True, output_logits=True,
        pad_token_id=backend.tokenizer.pad_token_id,
        eos_token_id=backend.tokenizer.eos_token_id,
        bos_token_id=backend.tokenizer.bos_token_id,
    )
    with torch.inference_mode():
        generated = backend.lm.generate(**inputs, generation_config=config)
    token_ids = generated.sequences[0, inputs["input_ids"].shape[1]:].tolist()
    text = backend.tokenizer.decode(token_ids)
    normalized = text.strip().lower()
    labels = [label.lower() for label in rendered.labels]
    prediction = -1
    matched_index = -1
    # Match the generated text against each answer label. Models emit many
    # label-adjacent forms ("A.", "A)", "A:", "The answer is A", "A\n") that are
    # not exact prefixes of the bare label, so accept a label when the emitted
    # text starts with the label followed by a non-alphanumeric delimiter or
    # when the label itself is a prefix of the emitted text.
    for index, label in enumerate(labels):
        if normalized == label or normalized.startswith(label + " ") or normalized.startswith(label + "\n"):
            prediction = example["options"].index(rendered.keys[index])
            matched_index = index
            break
        if _label_prefix_match(normalized, label):
            prediction = example["options"].index(rendered.keys[index])
            matched_index = index
            break
    logits = generated.logits[0][0].float()
    probabilities, label_mass = native_label_probabilities(backend, logits, rendered.labels)
    confidence = (probabilities[matched_index] if matched_index >= 0
                  else float(torch.softmax(logits, dim=-1)[token_ids[0]].item()))
    return {"prediction": prediction, "generated_text": text,
            "probabilities": probabilities, "confidence": confidence,
            "label_mass": label_mass}



def _parsed_generation_result(example: dict[str, Any], result: dict[str, Any], mode: str) -> dict[str, Any]:
    prediction = -1
    matched_index = -1
    for index, label in enumerate(result["labels"]):
        if (result["normalized"] == label or result["normalized"].startswith(label + " ")
                or result["normalized"].startswith(label + "\n")):
            prediction = example["options"].index(result["keys"][index])
            matched_index = index
            break
    confidence = (result["probabilities"][matched_index] if matched_index >= 0
                  else result["first_token_probability"])
    return {"prediction": prediction,
            "generated_token" if mode == "native_generate" else "generated_text": result["text"],
            "probabilities": result["probabilities"], "confidence": confidence,
            "label_mass": result["label_mass"]}


def native_label_probabilities(backend, logits, labels):
    """Aggregate full-vocabulary token variants; keep non-label mass explicitly.

    Vocabulary decoding is cached during warm-up, outside measured examples.
    No extra model forward pass is performed for native probability metrics.
    """
    import torch

    indices = native_label_token_indices(backend, labels, logits.device,
                                         vocabulary_size=int(logits.shape[-1]))
    token_probs = torch.softmax(logits, dim=-1)
    masses = [float(token_probs[index].sum().item()) for index in indices]
    total = sum(masses)
    # Guard float32 roundoff without discarding any non-label outcome.
    if total > 1.0:
        masses = [mass / total for mass in masses]
        total = 1.0
    return masses + [max(0.0, 1.0 - total)], total


def native_label_token_indices(backend, labels, device, vocabulary_size=None):
    """Return vocabulary IDs whose decoded token matches each answer label."""
    import torch

    if not hasattr(backend, "_native_decoded_tokens"):
        if vocabulary_size is None:
            output_embeddings = getattr(backend.lm, "get_output_embeddings", None)
            if output_embeddings is None or output_embeddings() is None:
                raise ValueError("Vocabulary size is required to decode candidate-label tokens")
            vocabulary_size = int(output_embeddings().weight.shape[0])
        backend._native_decoded_tokens = [
            text.strip().lower() for text in backend.tokenizer.batch_decode(
                [[i] for i in range(vocabulary_size)], skip_special_tokens=False)
        ]
        backend._native_label_indices = {}
    key = tuple(labels)
    if key not in backend._native_label_indices:
        backend._native_label_indices[key] = [
            torch.tensor([i for i, token in enumerate(backend._native_decoded_tokens)
                          if token == label.lower()], device=device, dtype=torch.long)
            for label in labels
        ]
    return backend._native_label_indices[key]


def reset_cache(jev: Any) -> None:
    """Give both modes a cold prompt cache for each timed example."""
    jev.backend._cache = None
    jev.backend._prefix_ids = []


def synchronize(jev: Any) -> None:
    """Include asynchronous accelerator work in wall-clock latency."""
    import torch
    device = str(jev.backend.device)
    if device.startswith("cuda"):
        torch.cuda.synchronize(jev.backend.device)
    elif device.startswith("mps"):
        torch.mps.synchronize()


def timed_call(jev: Any, function, example):
    reset_cache(jev)
    synchronize(jev)
    start = time.perf_counter()
    value = function(jev, example)
    synchronize(jev)
    return value, time.perf_counter() - start


def timed_batch_call(jev: Any, function, examples: Sequence[dict[str, Any]]):
    reset_cache(jev)
    synchronize(jev)
    start = time.perf_counter()
    value = function(jev, examples)
    synchronize(jev)
    return value, time.perf_counter() - start


def run_native(examples, jev, model, task, batch_size=1):
    details, latencies = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        if len(batch) == 1:
            # Keep the single-example path identical to the historical benchmark.
            results, elapsed = timed_call(jev, native_generate, batch[0])
            results, batch_elapsed = [results], elapsed
        else:
            results, batch_elapsed = timed_batch_call(
                jev, lambda model, rows: _generate_batch(model, rows, 1), batch)
            results = [_parsed_generation_result(row, result, "native_generate")
                       for row, result in zip(batch, results)]
        per_example = batch_elapsed / len(batch)
        for example, result in zip(batch, results):
            latencies.append(per_example)
            details.append(dict(result, id=example["id"], label=int(example["label"]),
                                options=list(example["options"]) + ["__invalid__"],
                                latency_seconds=per_example))
    record = summarize(model, task, "native_generate", [r["probabilities"] for r in details],
                       [r["label"] for r in details], latencies)
    correct = np.array([r["prediction"] == r["label"] for r in details])
    confidences = np.array([r["confidence"] for r in details])
    record.accuracy = float(correct.mean())
    record.mean_confidence = float(confidences.mean())
    record.ece = prediction_ece(confidences, correct)
    record.mean_label_mass = float(np.mean([r["label_mass"] for r in details]))
    record.invalid_predictions = sum(r["prediction"] < 0 for r in details)
    record.probability_basis = "full vocabulary: allowed label masses + invalid outcome; ECE uses emitted-answer confidence"
    return record, details


def run_constrained(examples, jev, model, task):
    details, latencies = [], []
    for example in tqdm(examples, desc=f"{task} constrained generation", unit="example"):
        result, elapsed = timed_call(jev, constrained_generate, example)
        latencies.append(elapsed)
        details.append(dict(result, id=example["id"], label=int(example["label"]),
                            options=list(example["options"]) + ["__invalid__"],
                            latency_seconds=elapsed))
    record = summarize(model, task, "constrained_generate",
                       [row["probabilities"] for row in details],
                       [row["label"] for row in details], latencies)
    record.mean_label_mass = 1.0
    record.probability_basis = "softmax over candidate-label token logits after hard vocabulary masking"
    record.timing += "; greedy max_new_tokens=1; invalid outputs are impossible"
    return record, details


def run_conventional(examples, jev, model, task, batch_size=1):
    details, latencies = [], []
    for start in range(0, len(examples), batch_size):
        batch = examples[start:start + batch_size]
        if len(batch) == 1:
            result, batch_elapsed = timed_call(jev, conventional_generate, batch[0])
            results = [result]
        else:
            results, batch_elapsed = timed_batch_call(
                jev, lambda model, rows: _generate_batch(model, rows, 32), batch)
            results = [_parsed_generation_result(row, result, "conventional_llm")
                       for row, result in zip(batch, results)]
        per_example = batch_elapsed / len(batch)
        for example, result in zip(batch, results):
            latencies.append(per_example)
            details.append(dict(result, id=example["id"], label=int(example["label"]),
                                options=list(example["options"]) + ["__invalid__"],
                                latency_seconds=per_example))
    record = summarize(model, task, "conventional_llm",
                       [r["probabilities"] for r in details],
                       [r["label"] for r in details], latencies)
    correct = np.array([r["prediction"] == r["label"] for r in details])
    confidences = np.array([r["confidence"] for r in details])
    record.accuracy = float(correct.mean())
    record.mean_confidence = float(confidences.mean())
    record.ece = prediction_ece(confidences, correct)
    record.mean_label_mass = float(np.mean([r["label_mass"] for r in details]))
    record.invalid_predictions = sum(r["prediction"] < 0 for r in details)
    record.probability_basis = "first-token full vocabulary: allowed label masses + invalid outcome; parsed text determines prediction"
    record.timing += "; greedy max_new_tokens=32"
    return record, details


def prediction_ece(confidences, correct, bins=15):
    """ECE of actual generated answers, including invalid outputs as incorrect."""
    confidences, correct = np.asarray(confidences), np.asarray(correct)
    edges = np.linspace(0, 1, bins + 1)
    value = 0.0
    for i in range(bins):
        mask = (confidences >= edges[i] if i == 0 else confidences > edges[i]) & (confidences <= edges[i+1])
        if mask.any():
            value += mask.mean() * abs(confidences[mask].mean() - correct[mask].mean())
    return float(value)

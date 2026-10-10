"""MedGemma-Direct adapter: a trained non-autoregressive decision baseline.

MedGemma-Direct is a fine-tuned MedGemma-4B with a pointer head that scores each
option against a ``<decide>`` position. It supports both text and multimodal
(text+image) decision rows, so it can be evaluated on every dataset in the
manifest.

The adapter implements the same ``score(examples, batch_size) -> (probabilities,
latencies)`` interface as :class:`med_jevify.non_ar.LayaAdapter`, so it can be
plugged into :func:`med_jevify.evaluate.run_probability_backend` unchanged.

Evaluation modes
----------------
``lora=False`` (head-only): the checkpoint stores only the trained
decision-head weights; the MedGemma backbone is fully frozen. At scoring time
the adapter loads the backbone once, runs the full forward pass (backbone +
decision head) per example, and converts the resulting candidate scores to
probabilities with softmax.

``lora=True`` (LoRA): the checkpoint additionally stores rank-`rank` LoRA
adapters on the q/k/v/o projections of the language model. The adapter wraps the
backbone with the same ``LoraConfig`` used at training time before loading the
saved LoRA weights, then runs the full forward pass per example.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

MEDGEMMA_DIRECT_RUN = "models/medgemma-direct-headonly"
MEDGEMMA_DIRECT_DTYPE = "bf16"


def _resolve_dtype(dtype: str) -> str:
    """Resolve dtype string to torch dtype attribute name."""
    if dtype == "auto":
        return "bfloat16"
    if dtype == "bf16":
        return "bfloat16"
    if dtype == "fp16":
        return "float16"
    if dtype == "fp32":
        return "float32"
    return dtype


def state_text(example: dict[str, Any]) -> str:
    """Extract the text context from a prepared decision row."""
    state = example.get("state", "")
    if isinstance(state, dict):
        return str(state.get("text", "") or "")
    return str(state or "")


def is_multimodal(example: dict[str, Any]) -> bool:
    state = example.get("state")
    return isinstance(state, dict) and bool(state.get("image"))


class MedGemmaDirectAdapter:
    """Score text and multimodal decision rows through a local MedGemma-Direct
    head-only checkpoint.

    The frozen MedGemma backbone is loaded once and the trained decision head
    is applied per example. Both text-only and image-conditioned rows are
    supported; the collator groups them so image rows keep their image attached.
    """

    def __init__(self, run: str = MEDGEMMA_DIRECT_RUN, device: str = "cuda:0",
                 dtype: str = MEDGEMMA_DIRECT_DTYPE, max_length: int = 512,
                 lora: bool | None = None, backbone: str | None = None):
        from transformers import AutoModelForCausalLM, AutoProcessor

        # Make the training modules importable from the source checkout.
        training_dir = str(Path(__file__).resolve().parents[2] / "training")
        if training_dir not in sys.path:
            sys.path.insert(0, training_dir)
        from laya_med.medgemma_direct import MedGemmaDirectCollator, MedGemmaDirectModel

        self.run = run
        self.device = torch.device(device)
        self.max_length = max_length

        # Load checkpoint metadata for the backbone path, head size and LoRA rank.
        checkpoint = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=True)
        saved_args = checkpoint["args"]
        self.head_size = int(saved_args.get("head_size", 512))
        self.lora_rank = int(saved_args.get("lora_rank", 0))
        medgemma_path = saved_args.get(
            "medgemma_model",
            "/home/razaimam/.cache/huggingface/hub/models--google--medgemma-4b-it/"
            "snapshots/290cda5eeccbee130f987c4ad74a59ae6f196408",
        )
        # The backbone used at training time is recorded in the checkpoint args;
        # fall back to MedGemma for checkpoints produced before this field existed.
        self.backbone_name = backbone or saved_args.get("backbone", "medgemma")

        # Auto-detect LoRA from the checkpoint if not explicitly requested.
        trainable = checkpoint.get("trainable_model", {})
        has_lora = any("lora_" in key for key in trainable)
        if lora is None:
            lora = has_lora
        self.lora = lora
        if lora and self.lora_rank <= 0:
            raise ValueError(f"LoRA checkpoint {run} has no positive --lora-rank in args")

        # Load the backbone and processor through the registry so Qwen2.5-VL and
        # future backbones load with the correct model/processor classes.
        from med_jevify.backbones import load_backbone, resolve_backbone
        spec = resolve_backbone(self.backbone_name)
        backbone, self.processor = load_backbone(spec, getattr(torch, _resolve_dtype(dtype)),
                                                  str(self.device))
        backbone.requires_grad_(False)

        # Wrap the language model with the same LoRA config used at training time.
        if lora:
            from peft import LoraConfig, TaskType, get_peft_model
            language_model = get_peft_model(
                backbone.model.language_model,
                LoraConfig(r=self.lora_rank, lora_alpha=2 * self.lora_rank, lora_dropout=0.05,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                           task_type=TaskType.FEATURE_EXTRACTION),
            )
            backbone.model.language_model = language_model

        config = getattr(backbone.config, "text_config", backbone.config)
        hidden_size = int(config.hidden_size)

        # Build the model: backbone (optionally LoRA-wrapped) + decision head.
        self.model = MedGemmaDirectModel(backbone, hidden_size, self.head_size).to(self.device)

        # Load the trainable weights (decision head + LoRA adapters) from the checkpoint.
        state = checkpoint["trainable_model"]
        head_state = {
            key.replace("decision_head.", "", 1): value
            for key, value in state.items()
            if key.startswith("decision_head.")
        }
        self.model.decision_head.load_state_dict(head_state)
        if lora:
            # Keys are prefixed with "backbone." relative to the MedGemmaDirectModel
            # instance, so load onto the whole model (strict=False to skip the
            # decision_head keys, which were already loaded above).
            self.model.load_state_dict(state, strict=False)
        self.model.eval()

        self.collator = MedGemmaDirectCollator(self.processor, max_length)
        if self.backbone_name != "medgemma":
            # Qwen2.5-VL's processor rejects truncation on image groups, so it
            # needs a specialized collator; other backbones use the default.
            from med_jevify.backbones import collator_for, resolve_backbone
            spec = resolve_backbone(self.backbone_name)
            self.collator = collator_for(spec, self.processor, max_length)

    @staticmethod
    def _wrap_row(example: dict[str, Any]) -> Any:
        """Convert a manifest dict into the attribute object the collator expects."""
        state = example.get("state", "")
        image = None
        if isinstance(state, dict):
            image = state.get("image")
            state = state.get("text", "")
        return type("Row", (), {
            "id": example["id"],
            "dataset": example["dataset"],
            "question": example["question"],
            "options": example["options"],
            "label": example["label"],
            "state": state,
            "image": image,
        })()

    def _move_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        for group in batch["groups"]:
            group["inputs"] = {
                key: value.to(self.device) if torch.is_tensor(value) else value
                for key, value in group["inputs"].items()
            }
        return batch

    def _forward(self, example: dict[str, Any]) -> np.ndarray:
        """Run the backbone + decision head and return raw per-option scores (logits)."""
        row = self._wrap_row(example)
        batch = self.collator([row])
        batch = self._move_batch(batch)
        # The decision head is float32 while the frozen MedGemma backbone is
        # bfloat16, so run the forward under autocast (matching training).
        amp = self.device.type == "cuda"
        with torch.inference_mode():
            if amp:
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    scores = self.model(batch["groups"], batch["option_counts"])
            else:
                scores = self.model(batch["groups"], batch["option_counts"])
        return scores[0].float().cpu().numpy()

    def _score_one(self, example: dict[str, Any]) -> np.ndarray:
        begin = time.perf_counter()
        logits = self._forward(example)
        probs = np.exp(logits - np.max(logits))
        probs /= probs.sum()
        self._last_latency = time.perf_counter() - begin
        return probs

    def score_logits(self, examples: Sequence[dict[str, Any]], batch_size: int
                     ) -> list[np.ndarray]:
        """Return raw per-option scores (logits) for each example, one at a time.

        Unlike :meth:`score`, this skips the softmax so the values can be fed
        directly to temperature scaling / test-time adaptation.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        return [self._forward(example) for example in examples]

    def score(self, examples: Sequence[dict[str, Any]], batch_size: int
              ) -> tuple[list[np.ndarray], list[float]]:
        """Score text and multimodal rows one at a time.

        The pointer head is per-record, so batching is not used here; the
        ``batch_size`` argument is accepted for interface compatibility.
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        probabilities, latencies = [], []
        for example in examples:
            probabilities.append(self._score_one(example))
            latencies.append(self._last_latency)
        return probabilities, latencies
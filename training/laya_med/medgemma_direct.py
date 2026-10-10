"""Direct candidate scoring on a causal multimodal MedGemma backbone."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import torch
from PIL import Image
from torch import nn


class MedGemmaDirectCollator:
    """Expand each decision into candidate-conditioned MedGemma prompts."""

    def __init__(self, processor: Any, max_length: int):
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        self.processor = processor
        self.max_length = max_length

    @staticmethod
    def _prompt(row: Any, candidate_index: int) -> str:
        choices = "\n".join(f"{chr(65 + i)}. {option}" for i, option in enumerate(row.options))
        candidate = row.options[candidate_index]
        return (
            f"Question: {row.question}\nOptions:\n{choices}\n"
            f"Candidate answer to assess: {chr(65 + candidate_index)}. {candidate}\n"
            f"Clinical context: {row.state}"
        )

    def __call__(self, rows: Sequence[Any]) -> dict[str, Any]:
        grouped: dict[bool, dict[str, list[Any]]] = {
            False: {"messages": [], "indices": []},
            True: {"messages": [], "indices": []},
        }
        option_counts = []
        candidate_index = 0
        for row in rows:
            if len(row.options) < 2:
                raise ValueError(f"{row.id}: at least two answer choices are required")
            option_counts.append(len(row.options))
            image = None
            if row.image:
                with Image.open(Path(row.image)) as source:
                    image = source.convert("RGB")
            for index in range(len(row.options)):
                text = self._prompt(row, index)
                if image is None:
                    content: Any = text
                else:
                    content = [{"type": "image", "url": image}, {"type": "text", "text": text}]
                group = grouped[image is not None]
                group["messages"].append([{"role": "user", "content": content}])
                group["indices"].append(candidate_index)
                candidate_index += 1

        encoded_groups = []
        for has_image in (False, True):
            group = grouped[has_image]
            if not group["messages"]:
                continue
            model_inputs = self.processor.apply_chat_template(
                group["messages"], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
                processor_kwargs={"padding": True, "truncation": True,
                                  "max_length": self.max_length},
            )
            encoded_groups.append({
                "inputs": dict(model_inputs),
                "candidate_indices": torch.tensor(group["indices"], dtype=torch.long),
            })
        return {"groups": encoded_groups, "option_counts": option_counts, "rows": list(rows)}


class MedGemmaDirectModel(nn.Module):
    """Score candidate-conditioned final hidden states without token generation."""

    def __init__(self, backbone: nn.Module, hidden_size: int, head_size: int = 512):
        super().__init__()
        if hidden_size <= 0 or head_size < 2:
            raise ValueError("hidden_size and head_size must be positive")
        if not hasattr(backbone, "model"):
            raise ValueError("Expected a causal multimodal model exposing its base model as .model")
        self.backbone = backbone
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size, head_size), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(head_size, head_size // 2), nn.GELU(), nn.Linear(head_size // 2, 1),
        )

    def train(self, mode: bool = True):
        super().train(mode)
        vision_tower = getattr(getattr(self.backbone, "model", None), "vision_tower", None)
        if vision_tower is not None:
            vision_tower.eval()
        return self

    def extract_features(self, groups: Sequence[dict[str, Any]],
                         option_counts: Sequence[int]) -> list[torch.Tensor]:
        """Per-candidate last-token hidden states BEFORE the decision head.

        Used by the head-only training mode to cache representations once and
        train only the decision head afterwards. Returns one tensor per
        decision, each with its own option count, matching forward()'s layout.
        """
        if any(count < 2 for count in option_counts):
            raise ValueError("Every decision must contain at least two candidates")
        candidate_count = sum(option_counts)
        features: list[torch.Tensor | None] = [None] * candidate_count
        for group in groups:
            inputs = group["inputs"]
            outputs = self.backbone.model(**inputs, use_cache=False, return_dict=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("MedGemma base model did not return last_hidden_state")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                last_positions = torch.full((hidden.shape[0],), hidden.shape[1] - 1,
                                            device=hidden.device, dtype=torch.long)
            else:
                positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
                last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
                if torch.any(last_positions < 0):
                    raise ValueError("Every candidate prompt must contain at least one attended token")
            candidate_features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
            indices = group["candidate_indices"].tolist()
            if len(indices) != len(candidate_features):
                raise ValueError("Candidate indices do not match the encoded prompt batch")
            for index, feature in zip(indices, candidate_features):
                if index < 0 or index >= candidate_count or features[index] is not None:
                    raise ValueError("Candidate index is invalid or duplicated")
                # Cast to float32 before caching: the decision head is float32, so
                # cached features must match it regardless of the backbone's autocast
                # regime (bf16/float16) at cache time.
                features[index] = feature.detach().float().cpu()
        if any(feature is None for feature in features):
            raise ValueError("Candidate prompts do not cover every choice")
        flat_features = torch.stack([f for f in features if f is not None])
        result = []
        offset = 0
        for count in option_counts:
            result.append(flat_features[offset:offset + count])
            offset += count
        return result

    def forward(self, groups: Sequence[dict[str, Any]], option_counts: Sequence[int]) -> list[torch.Tensor]:
        if any(count < 2 for count in option_counts):
            raise ValueError("Every decision must contain at least two candidates")
        candidate_count = sum(option_counts)
        scores: list[torch.Tensor | None] = [None] * candidate_count
        for group in groups:
            inputs = group["inputs"]
            outputs = self.backbone.model(**inputs, use_cache=False, return_dict=True)
            hidden = getattr(outputs, "last_hidden_state", None)
            if hidden is None:
                raise RuntimeError("MedGemma base model did not return last_hidden_state")
            attention_mask = inputs.get("attention_mask")
            if attention_mask is None:
                last_positions = torch.full((hidden.shape[0],), hidden.shape[1] - 1,
                                            device=hidden.device, dtype=torch.long)
            else:
                positions = torch.arange(hidden.shape[1], device=hidden.device).expand_as(attention_mask)
                last_positions = positions.masked_fill(attention_mask == 0, -1).max(dim=1).values
                if torch.any(last_positions < 0):
                    raise ValueError("Every candidate prompt must contain at least one attended token")
            candidate_features = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_positions]
            group_scores = self.decision_head(candidate_features).squeeze(-1)
            indices = group["candidate_indices"].tolist()
            if len(indices) != len(group_scores):
                raise ValueError("Candidate indices do not match the encoded prompt batch")
            for index, score in zip(indices, group_scores):
                if index < 0 or index >= candidate_count or scores[index] is not None:
                    raise ValueError("Candidate index is invalid or duplicated")
                scores[index] = score
        if any(score is None for score in scores):
            raise ValueError("Candidate prompts do not cover every choice")
        flat_scores = torch.stack(scores)
        result = []
        offset = 0
        for count in option_counts:
            result.append(flat_scores[offset:offset + count])
            offset += count
        return result

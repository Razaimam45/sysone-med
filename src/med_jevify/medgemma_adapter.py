"""Official Hugging Face processor adapter for Gemma/MedGemma multimodal inputs."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def load_model(model_id: str, device: str = "auto", dtype: str = "auto") -> tuple[Any, Any, str]:
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install model extras with `pip install -e '.[models]'`.") from exc
    resolved_device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    resolved_dtype = torch.bfloat16 if dtype == "auto" and (resolved_device in {"cuda", "mps"}) else torch.float32
    try:
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=resolved_dtype)
    except OSError as exc:
        raise RuntimeError(
            f"Unable to load {model_id}. Accept the model license on Hugging Face and run "
            "`huggingface-cli login` with a token that can access the gated repository."
        ) from exc
    return model.to(resolved_device).eval(), processor, resolved_device


def build_messages(example: dict[str, Any], image: str | Path | None = None) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    if image is not None:
        content.append({"type": "image", "image": str(image)})
    content.append({"type": "text", "text": f"State:\n{example.get('state', '')}\nQuestion: {example['question']}"})
    return [{"role": "user", "content": content}]


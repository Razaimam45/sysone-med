"""Backbone registry for MedGemma-Direct-style decision baselines.

Each backbone is a (model_class, processor_class, structure) triple where
``structure`` describes how the multimodal causal LM exposes its language
model and hidden size. The training and evaluation code paths are identical
across backbones once this registry resolves the concrete classes.

Supported structures
--------------------
``medgemma`` (default)
    ``AutoModelForCausalLM`` / ``AutoProcessor``. The model exposes
    ``.model.language_model`` and ``.model.vision_tower``. The processor
    implements ``apply_chat_template`` with image content blocks.

``qwen2_5_vl``
    ``Qwen2_5_VLForConditionalGeneration`` / ``Qwen2_5_VLProcessor``. Same
    interface shape as MedGemma: ``.model.language_model`` and
    ``.model.visual``. The existing ``MedGemmaDirectCollator`` and
    ``MedGemmaDirectModel`` work unchanged.

``internvl3``
    Reserved. ``AutoModelForCausalLM`` / ``AutoProcessor`` with
    ``trust_remote_code=True``. The model exposes ``.language_model`` and
    ``.vision_model`` directly (no ``.model`` wrapper), and its processor is a
    bare Qwen2 tokenizer that strips image blocks, so multimodal evaluation is
    not supported under the current transformers version. Text-only evaluation
    is supported via the ``internvl3_text`` structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image


@dataclass(frozen=True)
class BackboneSpec:
    """Resolved backbone: concrete model/processor classes plus structure tag.

    ``repo`` is the Hugging Face repo id used to load the model and processor;
    ``name`` is the internal structure tag used to pick the loading path.
    """

    repo: str
    name: str
    model_cls: Any
    processor_cls: Any
    structure: str
    trust_remote_code: bool = False


def _qwen_classes():
    from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor
    return Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLProcessor


def _internvl_classes():
    from transformers import AutoModelForCausalLM, AutoProcessor
    return AutoModelForCausalLM, AutoProcessor


# Registry of HF model ids -> resolved backbone spec.
BACKBONES: dict[str, BackboneSpec] = {
    "google/medgemma-4b-it": BackboneSpec(
        repo="google/medgemma-4b-it",
        name="medgemma",
        model_cls=None,  # resolved lazily via AutoModelForCausalLM
        processor_cls=None,
        structure="medgemma",
    ),
    "Qwen/Qwen2.5-VL-3B-Instruct": BackboneSpec(
        repo="Qwen/Qwen2.5-VL-3B-Instruct",
        name="qwen2_5_vl",
        model_cls=_qwen_classes()[0],
        processor_cls=_qwen_classes()[1],
        structure="qwen2_5_vl",
    ),
    "OpenGVLab/InternVL3-2B": BackboneSpec(
        repo="OpenGVLab/InternVL3-2B",
        name="internvl3",
        model_cls=_internvl_classes()[0],
        processor_cls=_internvl_classes()[1],
        structure="internvl3_text",
        trust_remote_code=True,
    ),
}

# Friendly aliases used by run scripts and the paper.
ALIASES: dict[str, str] = {
    "medgemma": "google/medgemma-4b-it",
    "qwen": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen2_5_vl": "Qwen/Qwen2.5-VL-3B-Instruct",
    "internvl3": "OpenGVLab/InternVL3-2B",
    "internvl": "OpenGVLab/InternVL3-2B",
}


def resolve_backbone(name: str) -> BackboneSpec:
    """Resolve a backbone name (alias or HF id) to a :class:`BackboneSpec`."""
    key = ALIASES.get(name, name)
    if key not in BACKBONES:
        raise ValueError(
            f"Unknown backbone {name!r}. Known: {sorted(ALIASES)} / {sorted(BACKBONES)}"
        )
    return BACKBONES[key]


def load_backbone(spec: BackboneSpec, dtype: Any, device: str):
    """Instantiate the backbone model and processor for ``spec``.

    Returns ``(model, processor)``. ``dtype`` may be a torch dtype or a string
    accepted by ``from_pretrained``. The model/processor are loaded from the
    original Hugging Face repo id, not the internal spec name.
    """
    if spec.name == "medgemma":
        from transformers import AutoModelForCausalLM, AutoProcessor
        model_cls, processor_cls = AutoModelForCausalLM, AutoProcessor
    else:
        model_cls, processor_cls = spec.model_cls, spec.processor_cls

    processor = processor_cls.from_pretrained(
        spec.repo, trust_remote_code=spec.trust_remote_code
    )
    model = model_cls.from_pretrained(
        spec.repo, dtype=dtype, low_cpu_mem_usage=True,
        trust_remote_code=spec.trust_remote_code,
    )
    return model, processor


def hidden_size_for(spec: BackboneSpec, model) -> int:
    """Return the language-model hidden size for the loaded backbone."""
    config = getattr(model.config, "text_config", None)
    if config is not None:
        return int(config.hidden_size)
    if spec.structure in ("medgemma", "qwen2_5_vl"):
        # MedGemma/Qwen expose .model.language_model; fall back to config.
        return int(getattr(model.config, "hidden_size", 0))
    # internvl3_text: the chat model carries hidden_size directly.
    return int(getattr(model.config, "hidden_size", 0))


def is_multimodal_supported(spec: BackboneSpec) -> bool:
    """Whether image-conditioned rows can be scored for this backbone."""
    return spec.structure in ("medgemma", "qwen2_5_vl")


def collator_for(spec: BackboneSpec, processor, max_length: int):
    """Return a per-backbone collator compatible with the existing model code.

    The default :class:`MedGemmaDirectCollator` works unchanged for MedGemma.
    Qwen2.5-VL's processor rejects ``truncation=True`` on image groups (it
    raises a token-count mismatch), so Qwen gets a subclass that disables
    truncation for image groups and falls back to the parent for text groups.
    """
    from laya_med.medgemma_direct import MedGemmaDirectCollator

    if spec.structure == "qwen2_5_vl":
        from transformers import Qwen2_5_VLProcessor

        class Qwen2_5_VLCollator(MedGemmaDirectCollator):
            """Qwen2.5-VL variant: never truncate image-conditioned groups.

            Qwen's image token count is fixed by the image, so truncating the
            text around it desyncs the image placeholder from the input ids.
            Text-only groups keep truncation (they have no image tokens).
            """

            def __call__(self, rows):
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
                            content = [{"type": "image", "url": image},
                                       {"type": "text", "text": text}]
                        group = grouped[image is not None]
                        group["messages"].append([{"role": "user", "content": content}])
                        group["indices"].append(candidate_index)
                        candidate_index += 1

                encoded_groups = []
                for has_image in (False, True):
                    group = grouped[has_image]
                    if not group["messages"]:
                        continue
                    kwargs = {"padding": True, "truncation": True, "max_length": self.max_length}
                    if has_image:
                        # Never truncate image groups: the image token count is
                        # fixed by the image, so truncating text desyncs it.
                        kwargs["truncation"] = False
                    model_inputs = self.processor.apply_chat_template(
                        group["messages"], tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt", processor_kwargs=kwargs,
                    )
                    encoded_groups.append({
                        "inputs": dict(model_inputs),
                        "candidate_indices": torch.tensor(group["indices"], dtype=torch.long),
                    })
                return {"groups": encoded_groups, "option_counts": option_counts, "rows": list(rows)}

        return Qwen2_5_VLCollator(processor, max_length)
    return MedGemmaDirectCollator(processor, max_length)
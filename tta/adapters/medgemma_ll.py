"""TTA adapter for MedGemma-LL (native candidate-label likelihood scoring).

MedGemma-LL is formulation (b) in Figure 2 -- the same frozen 4B instruction VLM
as MedGemma-AR, but instead of free-form text it scores the allowed answer labels
with a single next-token forward pass, emitting per-candidate log-probabilities.
This adapter wraps the pinned ``jevify`` backend and exposes raw per-option
logits through the TTA ``Adapter`` protocol, so routing and conformal prediction
run on it exactly as they do on the non-AR baselines.

No trainable parameters: the backbone is frozen and there is no decision head,
so RLCD-style test-time fine-tuning is not supported (``requires_grad`` is
False and the RLCD hooks are absent).

Scoring is batch-size 1, matching the AR inference constraint documented in
``scripts/run_trial.py`` ("AR inference and calibration always use batch size 1").
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.extend([str(ROOT / 'src'), str(ROOT / 'training'), str(ROOT / 'tta')])

DEFAULT_MODEL = "google/medgemma-4b-it"


class MedGemmaLLAdapter:
    """Score manifest rows through a MedGemma-LL checkpoint and expose raw logits."""

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto",
                 dtype: str = "auto", no_think: bool = True,
                 manifest: str = "data/prepared/manifest.json",
                 fit_temperature: bool = False):
        # dtype='auto' is resolved by the jevify backend (bfloat16 on CUDA,
        # float32 on CPU). Passing a literal like 'bf16' is rejected by
        # Gemma3Config.__post_init__.
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = model
        self.device = device
        self.dtype = dtype
        self.no_think = no_think
        self.manifest = manifest
        self._fit_temperature = fit_temperature
        self._temp: float | None = None  # lazy; None until first temperature() call
        # ``splits`` is read by the TTA method layer (rows_for) to select
        # calibration/test rows for a requested dataset, mirroring the other
        # adapters (LayaMedAdapter, MedGemmaDirectTtaAdapter).
        from med_jevify.decision_data import load_manifest
        _, self.splits = load_manifest(manifest)

        import jevify
        from med_jevify.evaluate import render_example  # noqa: F401  (sanity import)
        self._render_example = render_example
        # Qwen2.5-VL / InternVL3 use non-standard config classes that
        # AutoModelForCausalLM cannot auto-resolve under transformers 5.x.
        # Temporarily route from_pretrained through the registry for the
        # duration of the load; Gemma/MedGemma fall through unchanged.
        import transformers
        from med_jevify.backbones import resolve_backbone
        _orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained

        def _patched_from_pretrained(model_id, **kw):
            try:
                spec = resolve_backbone(model_id)
                if spec.model_cls is not None:
                    return spec.model_cls.from_pretrained(model_id, **kw)
            except ValueError:
                pass
            return _orig_from_pretrained(model_id, **kw)

        transformers.AutoModelForCausalLM.from_pretrained = _patched_from_pretrained
        try:
            self.jev = jevify.Jevify.from_transformers(
                model, device=device, dtype=dtype, no_think=no_think)
        finally:
            transformers.AutoModelForCausalLM.from_pretrained = _orig_from_pretrained
        self.tokenizer = self.jev.backend.tokenizer

    # --- Adapter protocol -------------------------------------------------

    def temperature(self) -> float:
        """Fitted temperature, or 1.0 (no scaling) to match the baseline protocol.

        MedGemma-LL's published baseline reports ``temperature: null`` -- i.e. raw
        next-token probabilities with no post-hoc scaling -- so the default is 1.0.
        Set ``fit_temperature=True`` to instead fit a shared temperature on the
        held-out calibration split (lazy, cached on first call).
        """
        if self._temp is None:
            if self._fit_temperature:
                self._temp = self._fit_temperature_on_calibration()
            else:
                self._temp = 1.0
        return self._temp

    def requires_grad(self) -> bool:
        return False  # frozen backbone, no decision head -> RLCD unsupported

    def score_logits(self, examples: Sequence[dict], batch_size: int) -> list[np.ndarray]:
        """Return raw per-option logits for each example, one at a time.

        Logits are ``log(matched label mass + 1e-9)`` over the row's options,
        matching the pinned upstream scoring protocol in
        ``med_jevify.evaluate.label_logits``. Batch size is advisory: AR scoring
        is one forward pass per example regardless.
        """
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        from med_jevify.evaluate import label_logits
        return [label_logits(self.jev, ex)[0] for ex in examples]

    # --- optional temperature fitting ------------------------------------

    def _fit_temperature_on_calibration(self) -> float:
        from med_jevify.decision_data import load_manifest
        from med_jevify.calibration import fit_temperature
        _, splits = load_manifest(self.manifest)
        rows = splits.get('calibration', [])
        if not rows:
            return 1.0
        logits = np.asarray(self.score_logits(rows, 1), dtype=float)
        labels = [int(r['label']) for r in rows]
        return float(fit_temperature(logits, labels).temperature)


def build(model: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "auto",
          no_think: bool = True, manifest: str = "data/prepared/manifest.json",
          fit_temperature: bool = False) -> MedGemmaLLAdapter:
    return MedGemmaLLAdapter(model=model, device=device, dtype=dtype, no_think=no_think,
                             manifest=manifest, fit_temperature=fit_temperature)
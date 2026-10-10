"""Non-autoregressive multimodal option-scoring model."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


class LayaMedModel(nn.Module):
    """Encode state/question/option pairs and score all options directly."""

    def __init__(self, text_encoder: nn.Module, vision_encoder: nn.Module | None,
                 text_dim: int, vision_dim: int, hidden_dim: int = 768,
                 fusion: str = 'mlp'):
        super().__init__()
        if fusion not in {'mlp', 'cross_attention'}:
            raise ValueError(f'Unknown fusion mode: {fusion}')
        self.fusion_mode = fusion
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        self.text_projection = nn.Sequential(nn.Linear(text_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.vision_projection = nn.Sequential(nn.Linear(vision_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()) if vision_encoder else None
        if fusion == 'cross_attention' and vision_encoder:
            self.fusion = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=0.1, batch_first=True)
            self.fusion_norm = nn.LayerNorm(hidden_dim)
            self.fusion_ffn = nn.Sequential(nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                                            nn.Dropout(0.1), nn.Linear(hidden_dim * 2, hidden_dim))
        else:
            self.fusion = nn.Sequential(
                nn.Linear(hidden_dim * (2 if vision_encoder else 1), hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            )
        self.option_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Linear(hidden_dim // 2, 1))

    def train(self, mode: bool = True):
        super().train(mode)
        for encoder in (self.text_encoder, self.vision_encoder):
            if encoder is not None and not any(p.requires_grad for p in encoder.parameters()):
                encoder.eval()
        return self

    @staticmethod
    def _pool(output: Any) -> torch.Tensor:
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        if getattr(output, "last_hidden_state", None) is not None:
            return output.last_hidden_state[:, 0]
        if isinstance(output, tuple):
            return output[0][:, 0]
        raise RuntimeError("Encoder output has no pooler or last hidden state.")

    def forward(self, text: dict[str, torch.Tensor], option_counts: list[int], images: dict[str, torch.Tensor] | None = None, image_indices: torch.Tensor | None = None) -> list[torch.Tensor]:
        if any(count < 2 for count in option_counts) or sum(option_counts) != len(text['input_ids']):
            raise ValueError("Candidate counts do not match token batch")
        text_features = self.text_projection(self._pool(self.text_encoder(**text)))
        if self.vision_encoder is not None:
            vision_features = torch.zeros_like(text_features)
            if images is not None:
                if image_indices is None or len(image_indices) != len(text_features):
                    raise ValueError("One image index required per candidate")
                vision_output = self.vision_encoder(**images)
                encoded = self.vision_projection(self._pool(vision_output))
                present = image_indices >= 0
                vision_features[present] = encoded[image_indices[present]]
            features = torch.cat([text_features, vision_features], dim=-1)
        else:
            features = text_features
        if self.fusion_mode == 'cross_attention' and self.vision_encoder is not None:
            tokens = features.reshape(features.shape[0], 2, -1)
            attended, _ = self.fusion(tokens, tokens, tokens, need_weights=False)
            fused = self.fusion_norm(tokens + attended)
            fused = self.fusion_norm(fused + self.fusion_ffn(fused))[:, 0]
        else:
            fused = self.fusion(features)
        scores = self.option_head(fused).squeeze(-1)
        result: list[torch.Tensor] = []
        offset = 0
        for count in option_counts:
            result.append(scores[offset:offset + count])
            offset += count
        return result

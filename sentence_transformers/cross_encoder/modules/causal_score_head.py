from __future__ import annotations

import torch

from sentence_transformers.base.modules import Module


class CausalScoreHead(Module):
    # TODO: Documentation
    config_keys = ["true_token_id", "false_token_id"]

    def __init__(self, true_token_id: int, false_token_id: int | None = None):
        super().__init__()
        self.true_token_id = true_token_id
        self.false_token_id = false_token_id
        self.num_labels = 1  # TODO: More labels? Is that possible?

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Left padding is enforced by Transformer, so the last position is always a real token.
        # With logits_to_keep=1, causal_logits has shape (batch_size, 1, vocab_size), which we
        # convert to (batch_size, vocab_size).
        logits = features["causal_logits"][:, -1]

        if self.false_token_id is None:
            scores = logits[:, self.true_token_id]
        else:
            scores = logits[:, self.true_token_id] - logits[:, self.false_token_id]

        features["scores"] = scores
        return features

    def save(self, output_path) -> None:
        self.save_config(output_path)

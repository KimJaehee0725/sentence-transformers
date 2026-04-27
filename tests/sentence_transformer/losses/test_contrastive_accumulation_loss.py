from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
import torch
from torch import nn
from transformers import set_seed

from sentence_transformers import SentenceTransformer
from sentence_transformers.base.modules.input_module import InputModule
from sentence_transformers.sentence_transformer.losses import (
    CachedMultipleNegativesRankingLoss,
    ContrastiveAccumulationLoss,
)


class _TinyTextEmbedding(InputModule):
    def __init__(self, vocab_size: int = 128, embedding_dim: int = 16) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

    def preprocess(self, inputs: list[str], prompt: str | None = None, **kwargs) -> dict[str, torch.Tensor]:
        if prompt:
            inputs = self._prepend_prompt(inputs, prompt)

        tokenized_inputs = []
        max_length = 1
        for text in inputs:
            token_ids = [ord(char) % self.vocab_size for char in text] or [0]
            tokenized_inputs.append(token_ids)
            max_length = max(max_length, len(token_ids))

        input_ids = torch.zeros((len(inputs), max_length), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for idx, token_ids in enumerate(tokenized_inputs):
            input_ids[idx, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
            attention_mask[idx, : len(token_ids)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def forward(self, features: dict[str, torch.Tensor | Any], **kwargs) -> dict[str, torch.Tensor | Any]:
        token_embeddings = self.embedding(features["input_ids"])
        attention_mask = features["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        sentence_embedding = (token_embeddings * attention_mask).sum(dim=1)
        sentence_embedding = sentence_embedding / attention_mask.sum(dim=1).clamp_min(1.0)
        features["sentence_embedding"] = sentence_embedding
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        return None


def _build_tiny_model() -> SentenceTransformer:
    return SentenceTransformer(modules=[_TinyTextEmbedding()])


@pytest.mark.parametrize(
    "samples",
    [
        [
            ("anchor a", "positive a"),
            ("anchor b", "positive b"),
            ("anchor c", "positive c"),
        ],
        [
            ("anchor a", "positive a", "negative a"),
            ("anchor b", "positive b", "negative b"),
            ("anchor c", "positive c", "negative c"),
        ],
    ],
    ids=["pair", "triplet"],
)
def test_bank_disabled_matches_cached_multiple_negatives_ranking_loss(
    samples: list[tuple[str, ...]],
) -> None:
    set_seed(42)
    base_model = _build_tiny_model()
    model_a = deepcopy(base_model)
    model_b = deepcopy(base_model)
    model_a.train()
    model_b.train()

    loss_kwargs = {
        "scale": 20.0,
        "mini_batch_size": 2,
        "gather_across_devices": False,
        "directions": ("query_to_doc",),
        "partition_mode": "joint",
        "show_progress_bar": False,
        "hardness_mode": None,
        "hardness_strength": 0.0,
    }
    cached_loss = CachedMultipleNegativesRankingLoss(model_a, **loss_kwargs)
    contaccum_loss = ContrastiveAccumulationLoss(model_b, bank_size=0, warmup_steps=0, **loss_kwargs)

    sentence_features = [base_model.preprocess(list(texts)) for texts in zip(*samples)]
    labels = torch.empty(0, dtype=torch.long)

    loss_cached = cached_loss(deepcopy(sentence_features), labels)
    loss_contaccum = contaccum_loss(deepcopy(sentence_features), labels)

    assert torch.allclose(loss_contaccum, loss_cached, atol=1e-5)

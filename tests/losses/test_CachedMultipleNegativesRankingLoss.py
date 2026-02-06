from __future__ import annotations

from sentence_transformers import InputExample, losses
from sentence_transformers.util import batch_to_device


def _pair_batch() -> list[InputExample]:
    return [
        InputExample(texts=["anchor 1", "positive 1"]),
        InputExample(texts=["anchor 2", "positive 2"]),
    ]


def _triplet_batch() -> list[InputExample]:
    return [
        InputExample(texts=["anchor 1", "positive 1", "negative 1"]),
        InputExample(texts=["anchor 2", "positive 2", "negative 2"]),
    ]


def test_cached_mnrl_bank_updates_and_fifo(stsb_bert_tiny_model):
    model = stsb_bert_tiny_model
    model.to("cpu")
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, bank_size=2)
    loss.train()

    features, labels = model.smart_batching_collate(_pair_batch())
    features = [batch_to_device(feature, model.device) for feature in features]
    if labels is not None:
        labels = labels.to(model.device)

    loss(features, labels)
    assert loss._candidate_bank is not None
    assert loss._anchor_bank is not None
    assert len(loss._candidate_bank) == 1
    assert len(loss._anchor_bank) == 1
    first_shape = loss._candidate_bank[0].shape
    first_anchor_shape = loss._anchor_bank[0].shape

    loss(features, labels)
    assert len(loss._candidate_bank) == 2
    assert len(loss._anchor_bank) == 2
    assert loss._candidate_bank[-1].shape == first_shape
    assert loss._anchor_bank[-1].shape == first_anchor_shape

    loss(features, labels)
    assert len(loss._candidate_bank) == 2
    assert len(loss._anchor_bank) == 2

    loss.eval()
    loss(features, labels)
    assert len(loss._candidate_bank) == 2
    assert len(loss._anchor_bank) == 2


def test_cached_mnrl_bank_includes_all_candidates_for_triplets(stsb_bert_tiny_model):
    model = stsb_bert_tiny_model
    model.to("cpu")
    loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=2, bank_size=1)
    loss.train()

    features, labels = model.smart_batching_collate(_triplet_batch())
    features = [batch_to_device(feature, model.device) for feature in features]
    if labels is not None:
        labels = labels.to(model.device)
    batch_size = features[0]["input_ids"].shape[0]

    loss(features, labels)
    assert loss._candidate_bank is not None
    assert loss._anchor_bank is not None
    assert len(loss._candidate_bank) == 1
    assert len(loss._anchor_bank) == 1
    # For triplets, candidates include positive and negative: 2 * batch_size rows.
    assert loss._candidate_bank[0].shape[0] == 2 * batch_size
    assert loss._anchor_bank[0].shape[0] == batch_size

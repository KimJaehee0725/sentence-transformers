from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any

import logging
import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad

logger = logging.getLogger(__name__)


class MultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct=util.cos_sim,
        gather_across_devices: bool = False,
        bank_size: int = 0,
    ) -> None:
        """
        Given a list of (anchor, positive) pairs or (anchor, positive, negative) triplets, this loss optimizes the following:

        1. Given an anchor (e.g. a question), assign the highest similarity to the corresponding positive (i.e. answer)
           out of every single positive and negative (e.g. all answers) in the batch.

        If you provide the optional negatives, they will all be used as extra options from which the model must pick the
        correct positive. Within reason, the harder this "picking" is, the stronger the model will become. Because of
        this, a higher batch size results in more in-batch negatives, which then increases performance (to a point).

        This loss function works great to train embeddings for retrieval setups where you have positive pairs
        (e.g. (query, answer)) as it will sample in each batch ``n-1`` negative docs randomly.

        This loss is also known as InfoNCE loss, SimCSE loss, Cross-Entropy Loss with in-batch negatives, or simply
        in-batch negatives loss.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: scale = 1 / temperature, so
                scale=20.0 is equivalent to temperature=0.05.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to
                dot product (and then set scale to 1)
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            bank_size: Number of previous update steps to use as additional in-batch negatives. If 0, the memory bank
                is disabled. A value of 4 means that the current batch will use negatives from the 4 previous update
                steps. The bank is updated after each forward pass and is global across devices when running with DDP.
                When enabled, the loss also includes an extra term so that cached anchors contribute gradients to the
                current candidates, matching the in-batch negative behavior of prior steps.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - `Training Examples > Natural Language Inference <../../../examples/sentence_transformer/training/nli/README.html>`_
            - `Training Examples > Paraphrase Data <../../../examples/sentence_transformer/training/paraphrases/README.html>`_
            - `Training Examples > Quora Duplicate Questions <../../../examples/sentence_transformer/training/quora_duplicate_questions/README.html>`_
            - `Training Examples > MS MARCO <../../../examples/sentence_transformer/training/ms_marco/README.html>`_
            - `Unsupervised Learning > SimCSE <../../../examples/sentence_transformer/unsupervised_learning/SimCSE/README.html>`_
            - `Unsupervised Learning > GenQ <../../../examples/sentence_transformer/unsupervised_learning/query_generation/README.html>`_

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative) triplets

        Inputs:
            +-------------------------------------------------+--------+
            | Texts                                           | Labels |
            +=================================================+========+
            | (anchor, positive) pairs                        | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative) triplets           | none   |
            +-------------------------------------------------+--------+
            | (anchor, positive, negative_1, ..., negative_n) | none   |
            +-------------------------------------------------+--------+

        Recommendations:
            - Use ``BatchSamplers.NO_DUPLICATES`` (:class:`docs <sentence_transformers.training_args.BatchSamplers>`) to
              ensure that no in-batch negatives are duplicates of the anchor or positive samples.

        Relations:
            - :class:`CachedMultipleNegativesRankingLoss` is equivalent to this loss, but it uses caching that allows for
              much higher batch sizes (and thus better performance) without extra memory usage. However, it is slightly
              slower.
            - :class:`MultipleNegativesSymmetricRankingLoss` is equivalent to this loss, but with an additional loss term.
            - :class:`GISTEmbedLoss` is equivalent to this loss, but uses a guide model to guide the in-batch negative
              sample selection. `GISTEmbedLoss` yields a stronger training signal at the cost of some training overhead.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.MultipleNegativesRankingLoss(model)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.gather_across_devices = gather_across_devices
        self.bank_size = bank_size
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self._candidate_bank = deque(maxlen=bank_size) if bank_size > 0 else None
        self._anchor_bank = deque(maxlen=bank_size) if bank_size > 0 else None
        self._warned_bank_size = False

        if bank_size < 0:
            raise ValueError("bank_size must be >= 0")

    def _maybe_warn_bank_size(
        self,
        step_anchor_count: int,
        num_candidate_columns: int,
        world_size: int,
    ) -> None:
        if self.bank_size <= 0 or self._warned_bank_size:
            return
        step_candidate_count = step_anchor_count * num_candidate_columns
        max_bank_anchors = self.bank_size * step_anchor_count
        max_bank_candidates = self.bank_size * step_candidate_count
        logger.warning(
            "MultipleNegativesRankingLoss: bank_size=%d caches the last k forward passes (micro-batches). "
            "Per step: anchors=%d, candidates=%d (candidate columns=%d, world_size=%d). "
            "At full capacity, the bank holds up to %d anchors and %d candidates. "
            "If you use gradient accumulation, bank_size counts micro-batches, not optimizer steps. "
            "Adjust bank_size accordingly.",
            self.bank_size,
            step_anchor_count,
            step_candidate_count,
            num_candidate_columns,
            world_size,
            max_bank_anchors,
            max_bank_candidates,
        )
        self._warned_bank_size = True

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Compute the embeddings and distribute them to anchor and candidates (positive and optionally negatives)
        embeddings = [self.model(sentence_feature)["sentence_embedding"] for sentence_feature in sentence_features]

        return self.compute_loss_from_embeddings(embeddings, labels)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        """
        Compute the multiple negatives ranking loss from embeddings.

        Args:
            embeddings: List of embeddings

        Returns:
            Loss value
        """
        anchors = embeddings[0]  # (batch_size, embedding_dim)
        candidates = embeddings[1:]  # (1 + num_negatives) tensors of shape (batch_size, embedding_dim)
        batch_size = anchors.size(0)
        offset = 0
        num_candidate_columns = len(embeddings) - 1
        world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1

        if self.training and torch.is_grad_enabled():
            step_anchor_count = batch_size * world_size
            self._maybe_warn_bank_size(step_anchor_count, num_candidate_columns, world_size)

        if self.gather_across_devices:
            # Gather the positives and negatives across all devices, with gradients, but not the anchors. We compute
            # only this device's anchors with all candidates from all devices, such that the backward pass on the document
            # embeddings can flow back to the original devices.
            candidates = [all_gather_with_grad(embedding_column) for embedding_column in candidates]
            # (1 + num_negatives) tensors of shape (batch_size * world_size, embedding_dim)

            # Adjust the offset to account for the gathered candidates
            if torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                offset = rank * batch_size

        candidates = torch.cat(candidates, dim=0)
        # (batch_size * world_size * (1 + num_negatives), embedding_dim)

        current_candidates_len = candidates.size(0)

        total_candidates = candidates
        total_anchors = anchors
        total_labels = [torch.arange(offset, offset + batch_size, device=anchors.device)]

        if self._candidate_bank:
            bank_candidates = torch.cat(list(self._candidate_bank), dim=0)
            total_candidates = torch.cat([total_candidates, bank_candidates], dim=0)

            bank_anchors = torch.cat(list(self._anchor_bank), dim=0)
            total_anchors = torch.cat([total_anchors, bank_anchors], dim=0)

            bank_labels = []
            bank_offset = current_candidates_len
            for step_anchors, step_candidates in zip(self._anchor_bank, self._candidate_bank):
                step_size = step_anchors.size(0)
                # Each cached step stores candidates as [positives..., negatives...], so the positives for that
                # step start at the current offset. We advance by the full step candidate block to align labels.
                bank_labels.append(torch.arange(bank_offset, bank_offset + step_size, device=anchors.device))
                bank_offset += step_candidates.size(0)
            total_labels.append(torch.cat(bank_labels, dim=0))

        total_labels = torch.cat(total_labels, dim=0)

        # For every anchor (current + cached), we compute the similarity to all candidates (current + cached).
        scores = self.similarity_fct(total_anchors, total_candidates) * self.scale
        # (batch_size + cached, world_size * batch_size * (1 + num_negatives) + cached)

        loss = self.cross_entropy_loss(scores, total_labels)

        if self.bank_size > 0 and self.training:
            self._update_banks(embeddings[0], embeddings[1:])

        return loss

    def _update_banks(self, anchors: Tensor, candidates: list[Tensor]) -> None:
        if self.bank_size <= 0:
            return
        with torch.no_grad():
            step_anchors = anchors.detach()
            step_candidates_columns = [candidate.detach() for candidate in candidates]
            if torch.distributed.is_initialized():
                step_anchors = util.all_gather(step_anchors, with_grad=False)
                step_candidates_columns = [
                    util.all_gather(step_candidates_column, with_grad=False)
                    for step_candidates_column in step_candidates_columns
                ]
            step_candidates = torch.cat(step_candidates_columns, dim=0)
            self._candidate_bank.append(step_candidates)
            self._anchor_bank.append(step_anchors)

    def get_config_dict(self) -> dict[str, Any]:
        return {
            "scale": self.scale,
            "similarity_fct": self.similarity_fct.__name__,
            "gather_across_devices": self.gather_across_devices,
            "bank_size": self.bank_size,
        }

    @property
    def citation(self) -> str:
        return """
@misc{henderson2017efficient,
    title={Efficient Natural Language Response Suggestion for Smart Reply},
    author={Matthew Henderson and Rami Al-Rfou and Brian Strope and Yun-hsuan Sung and Laszlo Lukacs and Ruiqi Guo and Sanjiv Kumar and Balint Miklos and Ray Kurzweil},
    year={2017},
    eprint={1705.00652},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

from __future__ import annotations

# TODO: Add the trainer-side ContrastiveAccumulationCommitCallback described in port_design.md §3.

from collections.abc import Callable, Iterable
from functools import partial
from typing import Any, Literal

import torch
import tqdm
from torch import Tensor

from sentence_transformers import util
from sentence_transformers.sentence_transformer.model import SentenceTransformer
from sentence_transformers.sentence_transformer.losses.cached_multiple_negatives_ranking import (
    CachedMultipleNegativesRankingLoss,
    _backward_hook,
)


class ContrastiveAccumulationLoss(CachedMultipleNegativesRankingLoss):
    """Cached InfoNCE loss with an optional stop-gradient dual memory bank as in paper Eq. 5-7."""

    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: Callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        bank_size: int = 0,
        warmup_steps: int = 0,
        gather_across_devices: bool = False,
        directions: tuple[
            Literal["query_to_doc", "query_to_query", "doc_to_query", "doc_to_doc"],
            ...,
        ] = ("query_to_doc",),
        partition_mode: Literal["joint", "per_direction"] = "joint",
        show_progress_bar: bool = False,
        hardness_mode: Literal["in_batch_negatives", "hard_negatives", "all_negatives"] | None = None,
        hardness_strength: float = 0.0,
    ) -> None:
        super().__init__(
            model=model,
            scale=scale,
            similarity_fct=similarity_fct,
            mini_batch_size=mini_batch_size,
            gather_across_devices=gather_across_devices,
            directions=directions,
            partition_mode=partition_mode,
            show_progress_bar=show_progress_bar,
            hardness_mode=hardness_mode,
            hardness_strength=hardness_strength,
        )

        if bank_size < 0:
            raise ValueError("bank_size must be non-negative.")
        if warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative.")
        if gather_across_devices and bank_size > 0:
            raise ValueError(
                "ContrastiveAccumulationLoss does not support gather_across_devices=True together with bank_size > 0 "
                "in v1. Multi-GPU global bank synchronization is planned as a follow-up; for now, either set "
                "bank_size=0 or run single-GPU."
            )

        self.bank_size = bank_size
        self.warmup_steps = warmup_steps

        self.query_bank: Tensor | None = None
        self.passage_bank: Tensor | None = None
        self._num_passage_columns: int | None = None

        self._bank_ptr = 0
        self._bank_fill = 0
        self._optimizer_steps = 0
        self._pending_queries: list[Tensor] = []
        self._pending_passages: list[Tensor] = []

    def _allocate_bank_if_needed(self, queries: Tensor, passages: Tensor) -> None:
        num_passage_columns = passages.size(1)
        if self._num_passage_columns is None:
            self._num_passage_columns = num_passage_columns
        elif self._num_passage_columns != num_passage_columns:
            raise ValueError(
                "ContrastiveAccumulationLoss expected "
                f"{self._num_passage_columns} passage columns from the first batch, but received "
                f"{num_passage_columns}."
            )

        if self.query_bank is not None and self.passage_bank is not None:
            return

        embed_dim = queries.size(-1)
        self.query_bank = torch.empty(
            (self.bank_size, embed_dim),
            device=queries.device,
            dtype=queries.dtype,
        )
        self.passage_bank = torch.empty(
            (self.bank_size, num_passage_columns, embed_dim),
            device=passages.device,
            dtype=passages.dtype,
        )

    def _get_bank_views(self) -> tuple[Tensor | None, Tensor | None]:
        if self._optimizer_steps < self.warmup_steps:
            return None, None
        if self._bank_fill == 0 or self.query_bank is None or self.passage_bank is None:
            return None, None
        return self.query_bank[: self._bank_fill], self.passage_bank[: self._bank_fill]

    def _calculate_loss_and_cache_gradients_with_bank(
        self,
        reps: list[list[Tensor]],
        bank_queries: Tensor | None,
        bank_passages: Tensor | None,
    ) -> Tensor:
        loss = self._calculate_loss_with_bank(reps, bank_queries=bank_queries, bank_passages=bank_passages, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def _calculate_loss_with_bank(
        self,
        reps: list[list[Tensor]],
        bank_queries: Tensor | None,
        bank_passages: Tensor | None,
        with_backward: bool = False,
    ) -> Tensor:
        current_queries = torch.cat(reps[0])
        current_docs = [torch.cat(doc_reps) for doc_reps in reps[1:]]
        batch_size = current_queries.size(0)
        row_device = current_queries.device

        docs_all = torch.cat(current_docs, dim=0)
        all_queries = current_queries
        num_docs = len(current_docs)

        if bank_queries is not None and bank_passages is not None:
            all_queries = torch.cat((all_queries, bank_queries), dim=0)
            bank_docs = [bank_passages[:, doc_idx, :] for doc_idx in range(num_docs)]
            docs_all = torch.cat((docs_all, *bank_docs), dim=0)

        losses: list[Tensor] = []
        for begin in tqdm.trange(
            0,
            batch_size,
            self.mini_batch_size,
            desc="Calculating loss",
            disable=not self.show_progress_bar,
        ):
            end = min(begin + self.mini_batch_size, batch_size)
            local_batch = torch.arange(begin, end, device=row_device)
            row_indices = torch.arange(end - begin, device=row_device)
            local_queries = current_queries[begin:end]
            local_docs = current_docs[0][begin:end]

            sim_matrices = {
                "query_to_doc": self.similarity_fct(local_queries, docs_all),
            }

            if "query_to_query" in self.directions:
                sim_matrices["query_to_query"] = self.similarity_fct(local_queries, all_queries)
                sim_matrices["query_to_query"][row_indices, local_batch] = -torch.inf

            if "doc_to_query" in self.directions:
                sim_matrices["doc_to_query"] = self.similarity_fct(all_queries, local_docs).T

            if "doc_to_doc" in self.directions:
                sim_matrices["doc_to_doc"] = self.similarity_fct(docs_all, local_docs).T
                current_same_query_doc_mask = torch.eye(batch_size, device=row_device, dtype=torch.bool)[local_batch]
                current_same_query_doc_mask = current_same_query_doc_mask.repeat(1, num_docs)
                if docs_all.size(0) > current_same_query_doc_mask.size(1):
                    bank_mask = torch.zeros(
                        (len(local_batch), docs_all.size(0) - current_same_query_doc_mask.size(1)),
                        device=row_device,
                        dtype=torch.bool,
                    )
                    same_query_doc_mask = torch.cat((current_same_query_doc_mask, bank_mask), dim=1)
                else:
                    same_query_doc_mask = current_same_query_doc_mask
                sim_matrices["doc_to_doc"].masked_fill_(same_query_doc_mask, -torch.inf)

            penalties = {}
            if (
                self.hardness_mode in ("in_batch_negatives", "hard_negatives", "all_negatives")
                and self.hardness_strength > 0.0
            ):
                penalty = self.hardness_strength * sim_matrices["query_to_doc"].detach()

                own_doc_mask = torch.eye(batch_size, device=row_device, dtype=torch.bool)[local_batch]
                own_doc_mask = own_doc_mask.repeat(1, num_docs)
                if docs_all.size(0) > own_doc_mask.size(1):
                    bank_mask = torch.zeros(
                        (len(local_batch), docs_all.size(0) - own_doc_mask.size(1)),
                        device=row_device,
                        dtype=torch.bool,
                    )
                    own_doc_mask = torch.cat((own_doc_mask, bank_mask), dim=1)

                if self.hardness_mode == "hard_negatives":
                    penalty_exclusion_mask = ~own_doc_mask
                    penalty_exclusion_mask[:, :batch_size] = True
                elif self.hardness_mode == "in_batch_negatives":
                    penalty_exclusion_mask = own_doc_mask
                else:
                    penalty_exclusion_mask = own_doc_mask
                    penalty_exclusion_mask[:, batch_size:] = False

                penalty[penalty_exclusion_mask] = 0.0
                penalties["query_to_doc"] = penalty

            for key in sim_matrices:
                sim_matrices[key] = sim_matrices[key] * self.scale
            for key, penalty in penalties.items():
                sim_matrices[key] = sim_matrices[key] + penalty

            positive_scores = sim_matrices["query_to_doc"][row_indices, local_batch]
            if self.partition_mode == "joint":
                all_scores = torch.cat(list(sim_matrices.values()), dim=1)
                log_z = torch.logsumexp(all_scores, dim=1)
            else:
                log_z = 0.0
                for sim_matrix in sim_matrices.values():
                    log_z += torch.logsumexp(sim_matrix, dim=1)
                log_z /= len(sim_matrices)

            per_sample_loss = -(positive_scores - log_z)
            loss_mbatch = per_sample_loss.mean() * len(local_batch) / batch_size

            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        return sum(losses)

    def _write_to_bank(self, queries: Tensor, passages: Tensor, start: int) -> None:
        assert self.query_bank is not None
        assert self.passage_bank is not None

        count = queries.size(0)
        first_chunk = min(count, self.bank_size - start)
        if first_chunk > 0:
            self.query_bank[start : start + first_chunk].copy_(queries[:first_chunk])
            self.passage_bank[start : start + first_chunk].copy_(passages[:first_chunk])

        remaining = count - first_chunk
        if remaining > 0:
            self.query_bank[:remaining].copy_(queries[first_chunk:])
            self.passage_bank[:remaining].copy_(passages[first_chunk:])

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor | None = None) -> Tensor:
        if self.bank_size == 0 or not self.model.training or not torch.is_grad_enabled():
            return super().forward(sentence_features, labels)

        sentence_features = list(sentence_features)
        if len(sentence_features) < 2:
            raise ValueError(f"Expected at least 2 inputs, got {len(sentence_features)}")

        reps: list[list[Tensor]] = []
        self.random_states = []
        for sentence_feature in sentence_features:
            reps_mbs = []
            random_state_mbs = []
            for reps_mb, random_state in self.embed_minibatch_iter(
                sentence_feature=sentence_feature,
                with_grad=False,
                copy_random_state=True,
            ):
                reps_mbs.append(reps_mb.detach().requires_grad_())
                random_state_mbs.append(random_state)
            reps.append(reps_mbs)
            self.random_states.append(random_state_mbs)

        current_queries = torch.cat(reps[0]).detach()
        current_passages = torch.stack([torch.cat(doc_reps).detach() for doc_reps in reps[1:]], dim=1)
        self._allocate_bank_if_needed(current_queries, current_passages)
        bank_queries, bank_passages = self._get_bank_views()

        loss = self._calculate_loss_and_cache_gradients_with_bank(reps, bank_queries=bank_queries, bank_passages=bank_passages)
        loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))

        if self._optimizer_steps >= self.warmup_steps:
            self._pending_queries.append(current_queries)
            self._pending_passages.append(current_passages)

        return loss

    def on_optimizer_step(self) -> None:
        if self.bank_size == 0:
            return

        try:
            if not self._pending_queries or self.query_bank is None or self.passage_bank is None:
                return

            staged_queries = torch.cat(self._pending_queries, dim=0)
            staged_passages = torch.cat(self._pending_passages, dim=0)
            total = staged_queries.size(0)

            if total >= self.bank_size:
                write_start = (self._bank_ptr + total) % self.bank_size
                self._write_to_bank(
                    queries=staged_queries[-self.bank_size :],
                    passages=staged_passages[-self.bank_size :],
                    start=write_start,
                )
                self._bank_ptr = write_start
                self._bank_fill = self.bank_size
            else:
                self._write_to_bank(queries=staged_queries, passages=staged_passages, start=self._bank_ptr)
                self._bank_ptr = (self._bank_ptr + total) % self.bank_size
                self._bank_fill = min(self.bank_size, self._bank_fill + total)
        finally:
            self._pending_queries.clear()
            self._pending_passages.clear()
            self._optimizer_steps += 1

    def get_config_dict(self) -> dict[str, Any]:
        config = super().get_config_dict()
        config.update(
            {
                "bank_size": self.bank_size,
                "warmup_steps": self.warmup_steps,
            }
        )
        return config

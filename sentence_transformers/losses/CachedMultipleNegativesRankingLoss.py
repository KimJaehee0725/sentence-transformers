from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator
from contextlib import nullcontext
from functools import partial
from typing import Any

import logging
import torch
import tqdm
from torch import Tensor, nn
from torch.utils.checkpoint import get_device_states, set_device_states

from sentence_transformers import util
from sentence_transformers.models import StaticEmbedding
from sentence_transformers.SentenceTransformer import SentenceTransformer
from sentence_transformers.util import all_gather_with_grad

logger = logging.getLogger(__name__)


class RandContext:
    """
    Random-state context manager class. Reference: https://github.com/luyug/GradCache.

    This class will back up the pytorch's random state during initialization. Then when the context is activated,
    the class will set up the random state with the backed-up one.
    """

    def __init__(self, *tensors) -> None:
        self.fwd_cpu_state = torch.get_rng_state()
        self.fwd_gpu_devices, self.fwd_gpu_states = get_device_states(*tensors)

    def __enter__(self) -> None:
        self._fork = torch.random.fork_rng(devices=self.fwd_gpu_devices, enabled=True)
        self._fork.__enter__()
        torch.set_rng_state(self.fwd_cpu_state)
        set_device_states(self.fwd_gpu_devices, self.fwd_gpu_states)

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self._fork.__exit__(exc_type, exc_val, exc_tb)
        self._fork = None


def _backward_hook(
    grad_output: Tensor,
    sentence_features: Iterable[dict[str, Tensor]],
    loss_obj: CachedMultipleNegativesRankingLoss,
) -> None:
    """A backward hook to backpropagate the cached gradients mini-batch by mini-batch."""
    assert loss_obj.cache is not None
    assert loss_obj.random_states is not None
    with torch.enable_grad():
        for sentence_feature, grad, random_states in zip(sentence_features, loss_obj.cache, loss_obj.random_states):
            for (reps_mb, _), grad_mb in zip(
                loss_obj.embed_minibatch_iter(
                    sentence_feature=sentence_feature,
                    with_grad=True,
                    copy_random_state=False,
                    random_states=random_states,
                ),
                grad,
            ):
                # TODO: This if-statement is for if the model does not require gradients, which may happen if the model
                # contains a Router where one of the routes is frozen. It should be possible to not have to call
                # embed_minibatch_iter in that case, as it's unnecessarily expensive.
                if reps_mb.requires_grad:
                    surrogate = torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    surrogate.backward()


class CachedMultipleNegativesRankingLoss(nn.Module):
    def __init__(
        self,
        model: SentenceTransformer,
        scale: float = 20.0,
        similarity_fct: callable[[Tensor, Tensor], Tensor] = util.cos_sim,
        mini_batch_size: int = 32,
        gather_across_devices: bool = False,
        bank_size: int = 0,
        show_progress_bar: bool = False,
    ) -> None:
        """
        Boosted version of MultipleNegativesRankingLoss (https://huggingface.co/papers/1705.00652) by GradCache (https://huggingface.co/papers/2101.06983).

        Constrastive learning (here our MNRL loss) with in-batch negatives is usually hard to work with large batch sizes due to (GPU) memory limitation.
        Even with batch-scaling methods like gradient-scaling, it cannot work either. This is because the in-batch negatives make the data points within
        the same batch non-independent and thus the batch cannot be broke down into mini-batches. GradCache is a smart way to solve this problem.
        It achieves the goal by dividing the computation into two stages of embedding and loss calculation, which both can be scaled by mini-batches.
        As a result, memory of constant size (e.g. that works with batch size = 32) can now process much larger batches (e.g. 65536).

        In detail:

            (1) It first does a quick embedding step without gradients/computation graphs to get all the embeddings;
            (2) Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings;
            (3) A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain.

        Notes: All steps are done with mini-batches. In the original implementation of GradCache, (2) is not done in mini-batches and
        requires a lot memory when the batch size is large. One drawback is about the speed. Gradient caching will sacrifice
        around 20% computation time according to the paper.

        Args:
            model: SentenceTransformer model
            scale: Output of similarity function is multiplied by scale value. In some literature, the scaling parameter
                is referred to as temperature, which is the inverse of the scale. In short: scale = 1 / temperature, so
                scale=20.0 is equivalent to temperature=0.05.
            similarity_fct: similarity function between sentence embeddings. By default, cos_sim. Can also be set to dot
                product (and then set scale to 1)
            mini_batch_size: Mini-batch size for the forward pass, this denotes how much memory is actually used during
                training and evaluation. The larger the mini-batch size, the more memory efficient the training is, but
                the slower the training will be. It's recommended to set it as high as your GPU memory allows. The default
                value is 32.
            gather_across_devices: If True, gather the embeddings across all devices before computing the loss.
                Recommended when training on multiple GPUs, as it allows for larger batch sizes, but it may slow down
                training due to communication overhead, and can potentially lead to out-of-memory errors.
            bank_size: Number of previous update steps to use as additional in-batch negatives. If 0, the memory bank
                is disabled. A value of 4 means that the current batch will use negatives from the 4 previous update
                steps. The bank is updated after each forward pass and is global across devices when running with DDP.
                When enabled, cached anchors contribute gradients to the current candidates, matching the in-batch
                negative behavior of prior steps.
            show_progress_bar: If True, a progress bar for the mini-batches is shown during training. The default is False.

        References:
            - Efficient Natural Language Response Suggestion for Smart Reply, Section 4.4: https://huggingface.co/papers/1705.00652
            - Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup: https://huggingface.co/papers/2101.06983
            - A Gradient Accumulation Method for Dense Retriever under Memory Constraint: https://arxiv.org/abs/2406.12356

        Requirements:
            1. (anchor, positive) pairs or (anchor, positive, negative pairs)
            2. Should be used with large `per_device_train_batch_size` and low `mini_batch_size` for superior performance, but slower training time than :class:`MultipleNegativesRankingLoss`.

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
            - Equivalent to :class:`MultipleNegativesRankingLoss`, but with caching that allows for much higher batch sizes
              (and thus better performance) without extra memory usage. This loss also trains roughly 2x to 2.4x slower than
              :class:`MultipleNegativesRankingLoss`.

        Example:
            ::

                from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
                from datasets import Dataset

                model = SentenceTransformer("microsoft/mpnet-base")
                train_dataset = Dataset.from_dict({
                    "anchor": ["It's nice weather outside today.", "He drove to work."],
                    "positive": ["It's so sunny.", "He took the car to the office."],
                })
                loss = losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=64)

                trainer = SentenceTransformerTrainer(
                    model=model,
                    train_dataset=train_dataset,
                    loss=loss,
                )
                trainer.train()
        """
        super().__init__()
        if isinstance(model[0], StaticEmbedding):
            raise ValueError(
                "CachedMultipleNegativesRankingLoss is not compatible with a SentenceTransformer model based on a StaticEmbedding. "
                "Consider using MultipleNegativesRankingLoss instead."
            )

        self.model = model
        self.scale = scale
        self.similarity_fct = similarity_fct
        self.mini_batch_size = mini_batch_size
        self.gather_across_devices = gather_across_devices
        self.bank_size = bank_size
        self.show_progress_bar = show_progress_bar

        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cache: list[list[Tensor]] | None = None
        self.random_states: list[list[RandContext]] | None = None
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
            "CachedMultipleNegativesRankingLoss: bank_size=%d caches the last k forward passes (micro-batches). "
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

    def embed_minibatch(
        self,
        sentence_feature: dict[str, Tensor],
        begin: int,
        end: int,
        with_grad: bool,
        copy_random_state: bool,
        random_state: RandContext | None = None,
    ) -> tuple[Tensor, RandContext | None]:
        """Do forward pass on a minibatch of the input features and return corresponding embeddings."""
        grad_context = nullcontext if with_grad else torch.no_grad
        random_state_context = nullcontext() if random_state is None else random_state
        sentence_feature_minibatch = {
            key: value[begin:end] if isinstance(value, torch.Tensor) else value
            for key, value in sentence_feature.items()
        }
        with random_state_context:
            with grad_context():
                random_state = RandContext(*sentence_feature_minibatch.values()) if copy_random_state else None
                reps = self.model(sentence_feature_minibatch)["sentence_embedding"]  # (mbsz, hdim)
        return reps, random_state

    def embed_minibatch_iter(
        self,
        sentence_feature: dict[str, Tensor],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext] | None = None,
    ) -> Iterator[tuple[Tensor, RandContext | None]]:
        """Do forward pass on all the minibatches of the input features and yield corresponding embeddings."""
        input_ids: Tensor = sentence_feature["input_ids"]
        bsz, _ = input_ids.shape
        for i, b in enumerate(
            tqdm.trange(
                0,
                bsz,
                self.mini_batch_size,
                desc="Embed mini-batches",
                disable=not self.show_progress_bar,
            )
        ):
            e = b + self.mini_batch_size
            reps, random_state = self.embed_minibatch(
                sentence_feature=sentence_feature,
                begin=b,
                end=e,
                with_grad=with_grad,
                copy_random_state=copy_random_state,
                random_state=None if random_states is None else random_states[i],
            )
            yield reps, random_state  # reps: (mbsz, hdim)

    def calculate_loss_and_cache_gradients(self, reps: list[list[Tensor]]) -> Tensor:
        """Calculate the cross-entropy loss and cache the gradients wrt. the embeddings."""
        loss = self.calculate_loss(reps, with_backward=True)
        loss = loss.detach().requires_grad_()

        self.cache = [[r.grad for r in rs] for rs in reps]  # e.g. 3 * bsz/mbsz * (mbsz, hdim)

        return loss

    def calculate_loss(self, reps: list[list[Tensor]], with_backward: bool = False) -> Tensor:
        """Calculate the cross-entropy loss. No need to cache the gradients."""
        anchors = torch.cat(reps[0])  # (batch_size, embedding_dim)
        candidates = [torch.cat(r) for r in reps[1:]]  # (1 + num_neg) tensors of shape (batch_size, embedding_dim)
        batch_size = anchors.size(0)
        offset = 0

        if self.gather_across_devices:
            # Gather the positives and negatives across all devices, with gradients, but not the anchors. We compute
            # only this device's anchors with all candidates from all devices, such that the backward pass on the document
            # embeddings can flow back to the original devices.
            candidates = [all_gather_with_grad(embedding_column) for embedding_column in candidates]
            # (1 + num_negatives) tensors of shape (batch_size * world_size, embedding_dim)

            # Adjust the range_labels to account for the gathered candidates
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
                bank_labels.append(torch.arange(bank_offset, bank_offset + step_size, device=anchors.device))
                bank_offset += step_candidates.size(0)
            total_labels.append(torch.cat(bank_labels, dim=0))

        total_labels = torch.cat(total_labels, dim=0)
        total_anchors_len = total_anchors.size(0)

        losses: list[torch.Tensor] = []
        for b in tqdm.trange(
            0,
            total_anchors_len,
            self.mini_batch_size,
            desc="Preparing caches",
            disable=not self.show_progress_bar,
        ):
            e = b + self.mini_batch_size
            scores: Tensor = self.similarity_fct(total_anchors[b:e], total_candidates) * self.scale
            loss_mbatch: torch.Tensor = (
                self.cross_entropy_loss(scores, total_labels[b:e]) * len(scores) / total_anchors_len
            )
            if with_backward:
                loss_mbatch.backward()
                loss_mbatch = loss_mbatch.detach()
            losses.append(loss_mbatch)

        loss = sum(losses)
        return loss

    def forward(self, sentence_features: Iterable[dict[str, Tensor]], labels: Tensor) -> Tensor:
        # Step (1): A quick embedding step without gradients/computation graphs to get all the embeddings
        reps = []
        self.random_states = []  # Copy random states to guarantee exact reproduction of the embeddings during the second forward pass, i.e. step (3)
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

        if self.training and torch.is_grad_enabled():
            local_batch = sum(rep.size(0) for rep in reps[0])
            num_candidate_columns = len(reps) - 1
            world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
            step_anchor_count = local_batch * world_size
            self._maybe_warn_bank_size(step_anchor_count, num_candidate_columns, world_size)

        if torch.is_grad_enabled():
            # Step (2): Calculate the loss, backward up to the embeddings and cache the gradients wrt. to the embeddings
            loss = self.calculate_loss_and_cache_gradients(reps)

            # Step (3): A 2nd embedding step with gradients/computation graphs and connect the cached gradients into the backward chain
            loss.register_hook(partial(_backward_hook, sentence_features=sentence_features, loss_obj=self))
        else:
            # If grad is not enabled (e.g. in evaluation), then we don't have to worry about the gradients or backward hook
            loss = self.calculate_loss(reps)

        if self.bank_size > 0 and self.training:
            self._update_banks(reps)

        return loss

    def _update_banks(self, reps: list[list[Tensor]]) -> None:
        if self.bank_size <= 0:
            return
        anchors = torch.cat(reps[0])
        candidates_columns = [torch.cat(r) for r in reps[1:]]
        with torch.no_grad():
            step_anchors = anchors.detach()
            step_candidates_columns = [candidate.detach() for candidate in candidates_columns]
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
            "mini_batch_size": self.mini_batch_size,
            "gather_across_devices": self.gather_across_devices,
            "bank_size": self.bank_size,
        }

    @property
    def citation(self) -> str:
        return """
@misc{gao2021scaling,
    title={Scaling Deep Contrastive Learning Batch Size under Memory Limited Setup},
    author={Luyu Gao and Yunyi Zhang and Jiawei Han and Jamie Callan},
    year={2021},
    eprint={2101.06983},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
@misc{kim2024gradient,
    title={A Gradient Accumulation Method for Dense Retriever under Memory Constraint},
    author={Sungdong Kim and Hwanhee Lee and Myeongho Jeong and Hoyeon Kim and Minbeom Lee and Kang Min Yoo and Taesup Moon},
    year={2024},
    eprint={2406.12356},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

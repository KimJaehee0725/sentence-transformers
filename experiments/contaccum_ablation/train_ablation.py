#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
import time
from pathlib import Path
from typing import cast


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ST retrieval model for ablations: grad-accum vs cont-accum vs cached+cont-accum."
    )
    parser.add_argument("--model_name", default="answerdotai/ModernBERT-base")
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_train_examples", type=int, default=-1, help="Per-dataset cap. -1 uses full split.")
    parser.add_argument("--seed", type=int, default=12)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=8192)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=512)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lr_scheduler_type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument("--loss_mini_batch_size", type=int, default=128, help="Used by Cached MNRL only.")
    parser.add_argument("--temperature", type=float, default=None, help="If set, uses scale=1/temperature.")
    parser.add_argument("--gather_across_devices", action="store_true")
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--dataloader_num_workers", type=int, default=12)
    parser.add_argument("--dataloader_prefetch_factor", type=int, default=2)
    parser.add_argument("--dataloader_persistent_workers", action="store_true", default=False)
    parser.add_argument("--no_drop_last", action="store_true", help="Disable drop_last (default: True).")
    parser.add_argument(
        "--batch_sampler",
        choices=["batch_sampler", "no_duplicates", "no_duplicates_hashed"],
        default="no_duplicates",
    )
    parser.add_argument(
        "--direction",
        choices=["uni", "bi"],
        default="uni",
        help="uni=query_to_doc, bi=query_to_doc+doc_to_query.",
    )
    parser.add_argument(
        "--strategy",
        choices=["grad_accum", "cont_accum", "cached_cont_accum"],
        default="cached_cont_accum",
    )
    parser.add_argument("--bank_size", type=int, default=0, help="Required for cont_accum/cached_cont_accum.")
    parser.add_argument(
        "--partition_mode",
        choices=["joint", "per_direction"],
        default=None,
        help="Default: uni->joint, bi->per_direction.",
    )
    parser.add_argument("--output_root", default="experiments/contaccum_ablation/output/models")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--no_shuffle", action="store_true")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--wandb_project", default="st-contaccum-ablation")
    parser.add_argument("--wandb_entity", default=None)
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    return parser.parse_args()


def build_output_dir(output_root: Path, run_name: str) -> Path:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return output_root / run_name / timestamp


def resolve_directions(direction: str, partition_mode: str | None) -> tuple[tuple[str, ...], str]:
    if direction == "bi":
        return ("query_to_doc", "doc_to_query"), (partition_mode or "per_direction")
    return ("query_to_doc",), (partition_mode or "joint")


def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_group:
        os.environ.setdefault("WANDB_RUN_GROUP", args.wandb_group)
    os.environ.setdefault("WANDB_MODE", args.wandb_mode)

    import torch
    from datasets import Dataset, DatasetDict, load_dataset
    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
        losses,
    )
    from sentence_transformers.evaluation import NanoBEIREvaluator

    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
    )
    logger = logging.getLogger("train_contaccum_ablation")

    if args.bf16 and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
        logger.warning("bf16 requested but not supported on this device; falling back to bf16=False.")
        args.bf16 = False

    if args.strategy in {"cont_accum", "cached_cont_accum"} and args.bank_size <= 0:
        raise ValueError(f"{args.strategy} requires --bank_size > 0")
    if args.strategy == "grad_accum" and args.bank_size != 0:
        raise ValueError("grad_accum strategy requires --bank_size 0")

    directions, partition_mode = resolve_directions(args.direction, args.partition_mode)

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    max_train_tag = "full" if args.max_train_examples < 0 else str(args.max_train_examples)
    model_tag = args.model_name.rstrip("/").split("/")[-1]
    temp_tag = "tdefault" if args.temperature is None else f"t{args.temperature}".replace(".", "p")
    if args.run_name is None:
        args.run_name = (
            f"{model_tag}_{args.direction}_{args.strategy}_{args.batch_sampler}_{temp_tag}"
            f"_bs{args.per_device_train_batch_size}_ga{args.gradient_accumulation_steps}"
            f"_bank{args.bank_size}_{max_train_tag}"
        )
    output_dir = build_output_dir(output_root, args.run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    final_dir = output_dir / "final"

    logger.info("Loading model: %s", args.model_name)
    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length

    def load_pair_dataset(dataset_id: str, config: str | None, rename_map: dict[str, str]) -> Dataset:
        ds = load_dataset(dataset_id, config, split="train") if config else load_dataset(dataset_id, split="train")
        ds = cast(Dataset, ds)
        if rename_map:
            column_names = ds.column_names or []
            existing = {k: v for k, v in rename_map.items() if k in column_names}
            if existing:
                ds = ds.rename_columns(existing)
        ds = ds.select_columns(["query", "positive"])
        return ds

    logger.info("Loading train datasets (pair only)")
    train_datasets = DatasetDict(
        {
            "msmarco": load_pair_dataset(
                "sentence-transformers/msmarco-co-condenser-margin-mse-sym-mnrl-mean-v1",
                "triplet",
                {"query": "query", "positive": "positive"},
            ),
            "natural_questions": load_pair_dataset(
                "sentence-transformers/natural-questions",
                "pair",
                {"answer": "positive"},
            ),
            "gooaq": load_pair_dataset(
                "sentence-transformers/gooaq",
                "pair",
                {"question": "query", "answer": "positive"},
            ),
            "ccnews": load_pair_dataset(
                "sentence-transformers/ccnews",
                "pair",
                {"title": "query", "article": "positive"},
            ),
            "hotpotqa": load_pair_dataset(
                "sentence-transformers/hotpotqa",
                "triplet",
                {"anchor": "query", "positive": "positive"},
            ),
        }
    )

    for name, ds in train_datasets.items():
        if not args.no_shuffle:
            ds = ds.shuffle(seed=args.seed)
        if args.max_train_examples > 0:
            ds = ds.select(range(min(args.max_train_examples, len(ds))))
        train_datasets[name] = ds
        logger.info("Train examples [%s]: %d", name, len(ds))

    loss_kwargs: dict[str, object] = {
        "directions": directions,
        "partition_mode": partition_mode,
        "bank_size": args.bank_size,
    }
    if args.temperature is not None:
        loss_kwargs["scale"] = 1.0 / args.temperature
    if args.gather_across_devices:
        loss_kwargs["gather_across_devices"] = True
    if args.strategy == "cached_cont_accum":
        loss_kwargs["mini_batch_size"] = args.loss_mini_batch_size

    if args.strategy == "cached_cont_accum":
        loss = losses.CachedMultipleNegativesRankingLoss(model=model, **loss_kwargs)
    else:
        loss = losses.MultipleNegativesRankingLoss(model=model, **loss_kwargs)

    report_to = [] if args.wandb_mode == "disabled" else ["wandb"]

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_strategy="steps",
        save_total_limit=args.save_total_limit,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        bf16=args.bf16,
        fp16=args.fp16,
        dataloader_num_workers=args.dataloader_num_workers,
        dataloader_prefetch_factor=args.dataloader_prefetch_factor,
        dataloader_persistent_workers=args.dataloader_persistent_workers,
        dataloader_drop_last=not args.no_drop_last,
        seed=args.seed,
        max_steps=args.max_steps,
        eval_strategy="no",
        report_to=report_to,
        remove_unused_columns=False,
        batch_sampler=args.batch_sampler,
        disable_tqdm=False,
        run_name=args.run_name,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        loss=loss,
    )

    logger.info(
        "Training start | strategy=%s direction=%s partition_mode=%s bank_size=%d output=%s",
        args.strategy,
        args.direction,
        partition_mode,
        args.bank_size,
        output_dir,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    evaluator = NanoBEIREvaluator(
        ndcg_at_k=[10],
        mrr_at_k=[10],
        accuracy_at_k=[10],
        precision_recall_at_k=[10],
        map_at_k=[10],
        batch_size=args.per_device_eval_batch_size,
        show_progress_bar=False,
        write_csv=False,
    )
    results = evaluator(
        model,
        output_path=str(output_dir / "eval"),
        epoch=0,
        steps=trainer.state.global_step,
    )
    ndcg_key = evaluator.primary_metric
    print(f"NDCG@10: {results[ndcg_key]:.6f} ({ndcg_key})")

    final_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_dir))
    model.save(str(final_dir), create_model_card=True)
    logger.info("Saved model to: %s", final_dir)


if __name__ == "__main__":
    main()


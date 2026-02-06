# ContAccum Ablation

This folder contains:

- `train_ablation.py`: single-run training script
- `run_2gpu.sh`: launcher for 6 runs (uni/bi x grad_accum/cont_accum/cached_cont_accum) with 2 runs in parallel on 2 GPUs

## Quick Start

```bash
wandb login
GPUS=0,1 \
WANDB_PROJECT=st-contaccum-ablation \
bash experiments/contaccum_ablation/run_2gpu.sh
```

## Main Env Knobs

- `TARGET_BSZ` (default: `8192`)
- `GRADACCUM_MICRO_BSZ` (default: `1024`)
- `CONT_MICRO_BSZ` (default: `1024`)
- `CACHED_BSZ` (default: `8192`)
- `CONT_BANK_SIZE` (default: `TARGET_BSZ/CONT_MICRO_BSZ - 1`)
- `CACHED_BANK_SIZE` (default: `CONT_BANK_SIZE`)
- `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE` (`online|offline|disabled`)
- `MODEL_NAME` (default: `answerdotai/ModernBERT-base`)

Logs are written under `experiments/contaccum_ablation/logs/<RUN_TAG>/`.

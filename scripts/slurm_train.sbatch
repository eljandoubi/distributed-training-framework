#!/bin/bash
#SBATCH --job-name=deepseekv3-train
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%x-%j.out
#SBATCH --error=slurm-%x-%j.err

# ---------------------------------------------------------------------------
# SLURM launch script for the DeepSeek-V3 distributed training framework.
#
# Usage:
#   export src_CONFIG=fsdp   # one of the presets in src/config/default_configs.py
#   sbatch scripts/slurm_train.sbatch
#
# Adjust --nodes / --ntasks-per-node / --gpus-per-node to match your cluster
# and the parallelism degrees configured for src_CONFIG.
# ---------------------------------------------------------------------------

set -euo pipefail

: "${src_CONFIG:?Set src_CONFIG to one of: ddp, fsdp, hsdp, pp_tp, fsdp_tp, fsdp_cp, hsdp_ep, fsdp_ep_tp, fsdp_ep_etp}"
export src_CONFIG

NPROC_PER_NODE="${SLURM_GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-8}}"

# Rendezvous endpoint: use the first node in the allocation.
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT="${MASTER_PORT:-29500}"
export MASTER_ADDR MASTER_PORT

srun --cpu-bind=none \
    torchrun \
        --nnodes="$SLURM_NNODES" \
        --nproc_per_node="$NPROC_PER_NODE" \
        --node_rank="$SLURM_NODEID" \
        --rdzv_id="$SLURM_JOB_ID" \
        --rdzv_backend=c10d \
        --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        -m src.train

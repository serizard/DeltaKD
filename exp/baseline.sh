#!/bin/bash

if [[ $# -eq 2 ]]; then
    GPU_IDS=$1
    MASTER_PORT=$2
else
    echo "Usage: $0 GPU_IDS (example: 0,1,2,3) MASTER_PORT (example: 29501)"
    exit 1
fi

NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

# CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
#     --student-model deit_tiny_patch16_224 \
#     --teacher-model deit_small_distilled_patch16_224 \
#     --dataset cifar-100 \
#     --data-path /root/workspace/AAAKD/dataset \
#     --epochs 500 \
#     --batch-size 256 \
#     --lr 5e-4 \
#     --alpha 0.5 \
#     --gpus $GPU_IDS \
#     --opt adamw \
#     --distillation-type vitkd \
#     --log-file logs/baseline-deit-tiny-cifar100.log \
#     --save-dir checkpoints/vitkd-deit-tiny-cifar100 \
#     --amp \
#     --wandb \
#     --wandb-project AAAKD_vitkd_128 \

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
    --student-model deit_tiny_patch16_224 \
    --teacher-model deit_small_distilled_patch16_224 \
    --dataset cifar-100 \
    --data-path /root/workspace/AAAKD/dataset \
    --epochs 300 \
    --batch-size 256 \
    --lr 4e-4 \
    --gpus $GPU_IDS \
    --opt adamw \
    --distillation-type vitkd \
    --log-file logs/vitkd-deit-tiny-cifar100.log \
    --save-dir checkpoints/vitkd-deit-tiny-cifar100 \
    --amp \
    --wandb \
    --wandb-project AAAKD_vitkd_256 \
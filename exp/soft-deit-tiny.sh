#!/bin/bash

if [[ $# -eq 1 ]]; then
    GPU_IDS=$1
else
    echo "Usage: $0 GPU_IDS (example: 0,1,2,3)"
    exit 1
fi

NUM_GPUS=$(echo $GPU_IDS | tr ',' '\n' | wc -l)

# soft distillation은 student model에 distillation 토큰이 필요
CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS tools/train.py \
    --student_model deit_tiny_distilled_patch16_224 \
    --teacher_model deit_small_distilled_patch16_224 \
    --dataset imagenet-1k \
    --data_path /root/datasets/imagenet \
    --epochs 300 \
    --batch_size 1024 \
    --lr 5e-4 \
    --gpus $GPU_IDS \
    --weight_decay 0.05 \
    --opt adamw \
    --drop_path_rate 0.1 \
    --distillation_type soft \
    --log_file logs/baseline-deit-tiny.log \
    --save_dir checkpoints/baseline-deit-tiny \
    --amp \
    # --wandb \
    # --wandb_project AAAKD

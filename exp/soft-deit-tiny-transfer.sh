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
#     --dataset stanford_cars \
#     --data-path /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/dataset \
#     --finetune \
#     --checkpoint /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/checkpoints/soft-deit-tiny-cifar100/checkpoint.pth \
#     --epochs 1000 \
#     --batch-size 512 \
#     --lr 5e-4 \
#     --weight-decay 1e-4 \
#     --gpus $GPU_IDS \
#     --distillation-type soft \
#     --log-file logs/soft-deit-tiny-stanford-cars.log \
#     --save-dir checkpoints/soft-deit-tiny-stanford-cars \
#     --wandb \
#     --wandb-project AAAKD

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
    --student-model deit_tiny_patch16_224 \
    --teacher-model deit_small_distilled_patch16_224 \
    --dataset flowers \
    --data-path /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/dataset \
    --finetune \
    --checkpoint /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/checkpoints/soft-deit-tiny-cifar100/checkpoint.pth \
    --epochs 1000 \
    --batch-size 512 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --gpus $GPU_IDS \
    --distillation-type soft \
    --log-file logs/soft-deit-tiny-flowers.log \
    --save-dir checkpoints/soft-deit-tiny-flowers \
    --wandb \
    --wandb-project AAAKD

CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=$NUM_GPUS --master_port $MASTER_PORT tools/train.py \
    --student-model deit_tiny_patch16_224 \
    --teacher-model deit_small_distilled_patch16_224 \
    --dataset caltech256 \
    --data-path /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/dataset \
    --finetune \
    --checkpoint /root/workspace/tactile_llava/dataset/gpt_instruction/misc/AAAKD/checkpoints/soft-deit-tiny-cifar100/checkpoint.pth \
    --epochs 1000 \
    --batch-size 512 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --gpus $GPU_IDS \
    --distillation-type soft \
    --log-file logs/soft-deit-tiny-caltech256.log \
    --save-dir checkpoints/soft-deit-tiny-caltech256 \
    --wandb \
    --wandb-project AAAKD

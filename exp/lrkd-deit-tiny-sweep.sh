#!/bin/bash


LRKD_RANK=${lrkd_rank:-32}
LRKD_ALPHA=${lrkd_alpha:-0.1}
LRKD_BETA=${lrkd_beta:-0.1}
LRKD_GAMMA=${lrkd_gamma:-0.1}


CUDA_VISIBLE_DEVICES=7 torchrun --nproc_per_node=1 --master_port 29502 tools/train.py \
    --student-model deit_tiny_patch16_224 \
    --teacher-model deit_small_distilled_patch16_224 \
    --dataset cifar-100 \
    --data-path /root/workspace/AAAKD/dataset \
    --epochs 20 \
    --batch-size 128 \
    --lr 5e-4 \
    --weight-decay 1e-4 \
    --alpha 0.5 \
    --lrkd-rank $LRKD_RANK \
    --lrkd-alpha $LRKD_ALPHA \
    --lrkd-beta $LRKD_BETA \
    --lrkd-gamma $LRKD_GAMMA \
    --gpus 7 \
    --distillation-type lrkd \
    --log-file logs/lrkd-deit-tiny-cifar100.log \
    --save-dir checkpoints/lrkd-deit-tiny-cifar100 \
    --wandb \
    --wandb-project AAAKD_LRKD

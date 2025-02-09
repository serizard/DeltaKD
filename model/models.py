import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from functools import partial
import timm
from dataset.datasets import DATASET_STATS
import types



def get_teacher_student_model(teacher_model_name, student_model_name, drop_path_rate=0.1, num_classes=1000, args=None):
    teacher_model = timm.create_model(teacher_model_name, 
                              pretrained=True, 
                              drop_path_rate=drop_path_rate, 
                              num_classes=DATASET_STATS[args.dataset]['num_classes'])

    student_model = timm.create_model(student_model_name, 
                              pretrained=False, 
                              drop_path_rate=drop_path_rate, 
                              num_classes=DATASET_STATS[args.dataset]['num_classes'])

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    distillation_type = args.distillation_type
    
    if distillation_type.lower() == 'vitkd':
        student_dims = student_model.embed_dim
        teacher_dims = teacher_model.embed_dim

        student_model.align2 = nn.ModuleList([
            nn.Linear(student_dims, teacher_dims, bias=True)
            for i in range(2)])
        student_model.align = nn.Linear(student_dims, teacher_dims, bias=True)
        student_model.mask_token = nn.Parameter(torch.zeros(1, 1, teacher_dims))
        student_model.generation = nn.Sequential(
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1),
            nn.ReLU(inplace=True), 
            nn.Conv2d(teacher_dims, teacher_dims, kernel_size=3, padding=1))

    elif distillation_type.lower() == 'lrkd':
        student_dims = student_model.embed_dim
        student_model.align = nn.ModuleList([
            nn.Linear(student_dims, args.lrkd_rank, bias=True)
            for i in range(3)])

    elif 'deit' in student_model_name and distillation_type.lower() in ['soft', 'hard']:
        student_model.set_distilled_training(enable=True)
    
    elif distillation_type.lower() == 'diffkd':
        student_dims = student_model.embed_dim
        teacher_dims = teacher_model.embed_dim

        # Create a custom denoising module instead of Sequential
        class DenoisingNetwork(nn.Module):
            def __init__(self, dims):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(dims, dims * 2),
                    nn.GELU(),
                    nn.Linear(dims * 2, dims),
                    nn.Dropout(0.1)
                )
                self.time_embed = nn.Sequential(
                    nn.Linear(1, dims),
                    nn.GELU(),
                    nn.Linear(dims, dims)
                )
            
            def forward(self, x, t):
                # Time embedding
                t_emb = self.time_embed(t.float().view(-1, 1))
                
                # Add time information to features
                x = x + t_emb.unsqueeze(1)
                
                # Predict noise
                return self.net(x)

        # Replace sequential with custom module
        student_model.denoise_fn = DenoisingNetwork(teacher_dims)

        # Feature alignment layers
        student_model.align = nn.ModuleList([
            nn.Linear(student_dims, teacher_dims, bias=True)
            for _ in range(3)  # 3개의 주요 레이어에 대해 정렬
        ])

    return teacher_model, student_model


def forward_with_features(model, x):
    if not hasattr(model, "blocks"):
        return None, None

    blocks = [block for block in model.blocks if hasattr(block, "mlp")]
    ffn_outputs = [None] * len(blocks)

    hook_handles = [
        block.mlp.register_forward_hook(
            lambda module, inp, out, idx=i: ffn_outputs.__setitem__(idx, out)
        )
        for i, block in enumerate(blocks)
    ]

    output = model(x)
    for handle in hook_handles:
        handle.remove()

    return output, ffn_outputs

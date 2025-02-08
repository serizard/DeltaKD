import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
from functools import partial
import timm
from dataset.datasets import DATASET_STATS



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

    elif 'deit' in student_model_name and distillation_type.lower() in ['soft', 'hard']:
        student_model.set_distilled_training(enable=True)
    
    
    return teacher_model, student_model


def forward_with_features(model, x):
    ffn_outputs = [] 
    if hasattr(model, 'blocks'):
        for block in model.blocks:
            if hasattr(block, 'mlp'):
                def hook_fn(module, input, output):
                    ffn_outputs.append(output)
                handle = block.mlp.register_forward_hook(hook_fn)
    
        output = model(x)
        
        for block in model.blocks:
            if hasattr(block, 'mlp'):
                for hook in block.mlp._forward_hooks.values():
                    hook.remove()
        
        return output, ffn_outputs 
    
    return None, None 

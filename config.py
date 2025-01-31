from multiprocessing import cpu_count


'''
Available Models
- deit_base_patch16_224
- deit_small_patch16_224
- deit_tiny_patch16_224
- swin_tiny_patch4_window7_224
- swin_small_patch4_window7_224
- swin_base_patch4_window7_224
'''


class Config:
    teacher_model = 'deit_small_patch16_224'
    student_model = 'deit_tiny_patch16_224'
    
    # Training
    batch_size = 1024
    lr = 5e-4
    epochs = 300
    weight_decay = 0.05
    mixup_alpha = 0.8
    cutmix_alpha = 1.0
    random_erase_prob = 0.25
    randaug_num_ops = 9
    randaug_magnitude = 0.5
    switch_prob = 0.5
    mixup_prob = 1.0
    num_workers = cpu_count()
    warmup_epochs = 5
    label_smoothing = 0.1
    drop_path_rate = 0.1

    # Distillation
    distillation_type = 'none'
    alpha = 0.5
    tau = 1.0

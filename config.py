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


    LOSS_WEIGHT = {
        'attn_weight': 0.5,
        'token_weight': 1.0,
        'pos_weight': 0.1,
        'ent_align_weight': 0.2
    }
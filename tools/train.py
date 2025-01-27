from config import Config
from model.models import get_model
from model.wrapper import DistillWrapper
from loss import *
import argparse
from logs.logger import setup_logger
import torch
import wandb
from timm.data import Mixup
from dataset.datasets import DatasetBuilder
from schedules import OptimizerFactory, Scheduler, ScaledGradNorm
from timm.utils import ModelEma
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import setup_distributed, setup_device, seed_everything
from engine import train_one_epoch, evaluate, validate
from utils import save_checkpoint



def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for ViT knowledge distillation training')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--teacher_model', type=str, default=Config.teacher_model,
                            help='Teacher model architecture')
    model_group.add_argument('--student_model', type=str, default=Config.student_model,
                            help='Student model architecture')
    
    # Training hyperparameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batch_size', type=int, default=Config.batch_size)
    train_group.add_argument('--lr', type=float, default=Config.lr)
    train_group.add_argument('--epochs', type=int, default=Config.epochs)
    train_group.add_argument('--num_workers', type=int, default=Config.num_workers)
    train_group.add_argument('--amp', action='store_true', 
                            help='Use mixed precision training')
    
    # Distillation weights
    distill_group = parser.add_argument_group('Distillation')
    distill_group.add_argument('--weight_attn', type=float, default=Config.weight_attn,
                              help='Attention distillation weight')
    distill_group.add_argument('--weight_token', type=float, default=Config.weight_token,
                              help='Token distillation weight')
    distill_group.add_argument('--weight_pos', type=float, default=Config.weight_pos,
                              help='Position embedding distillation weight')
    distill_group.add_argument('--weight_ent_align', type=float, default=Config.weight_ent_align,
                              help='Entropy alignment weight')
    
    # Threshold arguments
    threshold_group = parser.add_argument_group('Thresholds')
    threshold_group.add_argument('--alpha_threshold', type=float, default=Config.alpha_threshold,
                                help='Dynamic entropy threshold')
    threshold_group.add_argument('--local_head_threshold', type=float, default=Config.local_head_threshold,
                                help='Local/Global head classification threshold')
    
    # Saving and logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log_file', type=str, default='train.log')
    log_group.add_argument('--save_dir', type=str, default='checkpoints')
    log_group.add_argument('--wandb', action='store_true', 
                          help='Use Weights & Biases logging')
    
    # Miscellaneous settings
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--resume', action='store_true',
                           help='Resume training from checkpoint')
    misc_group.add_argument('--checkpoint', type=str, default=None,
                           help='Path to checkpoint file')
    misc_group.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility')
    misc_group.add_argument('--device', type=str, 
                           default='cuda' if torch.cuda.is_available() else 'cpu',
                           help='Device to use for training')
    misc_group.add_argument('--dataset', type=str, default='cifar-100',
                           help='Dataset to use for training')
    misc_group.add_argument('--mixup', action='store_true',
                           help='Use Mixup data augmentation')
    
    return parser.parse_args()



def main():
    args = parse_args()
    print(args)

    config = Config()

    logger = setup_logger('train.log')
    logger.info(f"Training started with {args.teacher_model} as teacher and {args.student_model} as student")

    setup_distributed(args)
    device = setup_device(args) 
    seed_everything(args.seed)

    teacher_model = get_model(args.teacher_model, pretrained=True)
    student_model = get_model(args.student_model, pretrained=False)

    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    teacher_model = DistillWrapper(teacher_model, is_teacher=True)
    student_model = DistillWrapper(student_model, is_teacher=False)

    if args.wandb:
        wandb.init(project='distill-vit', config=args)

    dataset_builder = DatasetBuilder(args)
    train_loader, train_sampler = dataset_builder.build_loader(is_train=True)
    val_loader, _ = dataset_builder.build_loader(is_train=False)   

    optimizer = OptimizerFactory.create(student_model, args)
    scheduler = Scheduler(optimizer, args)
    grad_scaler = ScaledGradNorm(args)

    if args.mixup:
        mixup_fn = dataset_builder.build_mixup_fn()
    else:
        mixup_fn = None

    if args.ema:
        model_ema = ModelEma(student_model, decay=args.ema_decay, device=device)
    else:
        model_ema = None

    if args.distributed:
        student_model = DDP(student_model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
        teacher_model = DDP(teacher_model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)

    logger.info(f"Training started with {args.teacher_model} as teacher and {args.student_model} as student")

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(student_model, teacher_model, train_loader, optimizer, scheduler, grad_scaler, mixup_fn, model_ema, device, args)
        val_metrics = validate(student_model, val_loader, device, args)
        if args.wandb:
            wandb.log(train_metrics, step=epoch)
            wandb.log(val_metrics, step=epoch)

        logger.info(f"Epoch {epoch} - Train: {train_metrics} - Val: {val_metrics}")

        if args.save_dir:
            save_checkpoint(student_model, optimizer, epoch, args)

    logger.info("Training completed")
    logger.info("Final validation metrics:")
    logger.info(val_metrics)

    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()


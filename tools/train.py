from config import Config
from model.models import VisionModelWrapper
from model.loss import DistillationLoss, call_base_loss
import argparse
from logs.logger import setup_logger
import torch
import wandb
import os
from dataset.datasets import DatasetBuilder
from schedules import OptimizerFactory, Scheduler
from timm.utils import ModelEma
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from utils import setup_distributed, setup_device, seed_everything
from engine import train_one_epoch, validate
from utils import save_checkpoint, remove_module_prefix



def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for ViT knowledge distillation training')
    
    # Model arguments
    model_group = parser.add_argument_group('Model')
    model_group.add_argument('--teacher_model', type=str, default=Config.teacher_model,
                            help='Teacher model architecture')
    model_group.add_argument('--student_model', type=str, default=Config.student_model,
                            help='Student model architecture')
    model_group.add_argument('--fp16', action='store_true', help='Use FP16 training')

    # Training hyperparameters
    train_group = parser.add_argument_group('Training')
    train_group.add_argument('--batch_size', type=int, default=Config.batch_size)
    train_group.add_argument('--lr', type=float, default=Config.lr)
    train_group.add_argument('--epochs', type=int, default=Config.epochs)
    train_group.add_argument('--num_workers', type=int, default=Config.num_workers)
    train_group.add_argument('--amp', action='store_true', 
                            help='Use mixed precision training')
    train_group.add_argument('--weight_decay', type=float, default=Config.weight_decay)
    train_group.add_argument('--opt', type=str, default='adamw')
    train_group.add_argument('--warmup_epochs', type=int, default=Config.warmup_epochs)
    train_group.add_argument('--drop_path_rate', type=float, default=Config.drop_path_rate)
    train_group.add_argument('--label_smoothing', type=float, default=Config.label_smoothing)
    train_group.add_argument('--ema_decay', type=float, default=None)

    # Distributed training
    dist_group = parser.add_argument_group('Distributed')
    dist_group.add_argument('--gpus', type=str, default=None,
                            help='Comma-separated list of GPU indices to use for distributed training. '
                                 'If not specified, defaults to using all available GPUs or CPU if no GPUs are available.')
    dist_group.add_argument('--dist_url', default='env://',
                            help='url used to set up distributed training')


    # Distillation
    distill_group = parser.add_argument_group('Distillation')
    distill_group.add_argument('--distillation_type', type=str, choices=['none', 'soft', 'hard', 'vitkd', 'aaakd', 'vitkd_w_logit', 'aaakd_w_logit'], default='none',
                              help='Distillation type')
    distill_group.add_argument('--alpha', type=float, default=Config.alpha,
                              help='Alpha for distillation')
    distill_group.add_argument('--tau', type=float, default=Config.tau,
                              help='Tau for distillation')
    
    # Saving and logging
    log_group = parser.add_argument_group('Logging')
    log_group.add_argument('--log_file', type=str, default='logs/train.log')
    log_group.add_argument('--save_dir', type=str, default='checkpoints')
    log_group.add_argument('--wandb', action='store_true', 
                          help='Use Weights & Biases logging')
    log_group.add_argument('--wandb_project', type=str, default='distill-vit',
                          help='Weights & Biases project name')
    
    # Data
    data_group = parser.add_argument_group('Data')
    data_group.add_argument('--data_path', type=str, default='dataset')
    data_group.add_argument('--dataset', type=str, default='imagenet-1k',
                           help='Dataset to use for training')
    
    # Miscellaneous settings
    misc_group = parser.add_argument_group('Miscellaneous')
    misc_group.add_argument('--resume', action='store_true',
                           help='Resume training from checkpoint')
    misc_group.add_argument('--checkpoint', type=str, default=None,
                           help='Path to checkpoint file')
    misc_group.add_argument('--seed', type=int, default=42,
                           help='Random seed for reproducibility')
    misc_group.add_argument('--device', type=str, default=None,
                           help='Device to use for training')
    misc_group.add_argument('--mixup', action='store_true', default=True,
                           help='Use Mixup data augmentation')
    
    return parser.parse_args()


def main():
    args = parse_args()
    setup_distributed(args)
    device = setup_device(args) 
    seed_everything(args.seed)

    if args.rank == 0: 
        print(args)

    teacher_model = VisionModelWrapper(args.teacher_model, pretrained=True, drop_path_rate=args.drop_path_rate, args=args)
    student_model = VisionModelWrapper(args.student_model, pretrained=False, drop_path_rate=args.drop_path_rate, args=args)
    teacher_model = teacher_model.freeze_model()

    logger = setup_logger(args.log_file)
    logger.info(f"Training started with {args.teacher_model} as teacher and {args.student_model} as student")

    if args.wandb and dist.get_rank() == 0: 
        logger.info("Wandb init")
        wandb.init(project=args.wandb_project, config=args, name="baseline_deit")

    dataset_builder = DatasetBuilder(args)
    train_loader, train_sampler = dataset_builder.build_loader(is_train=True)
    val_loader, _ = dataset_builder.build_loader(is_train=False)   

    optimizer = OptimizerFactory.create_optimizer(student_model, args)
    scheduler = Scheduler(optimizer, args)
    # grad_scaler = ScaledGradNorm(args)
    grad_scaler = None

    criterion_task = call_base_loss(args)
    criterion_distillation = DistillationLoss(base_criterion=criterion_task, teacher_model=teacher_model, distillation_type=args.distillation_type, alpha=args.alpha, tau=args.tau)

    if args.resume:
        checkpoint = torch.load(args.checkpoint)
        state_dict = checkpoint['model_state_dict']
        # "module." 접두사 제거 => key 맞추기
        state_dict = remove_module_prefix(state_dict)
        student_model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if grad_scaler is not None and 'scaler_state_dict' in checkpoint:
            grad_scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0


    if args.mixup:
        mixup_fn = dataset_builder.build_mixup_fn()
    else:
        mixup_fn = None

    if args.ema_decay:
        model_ema = ModelEma(student_model, decay=args.ema_decay, device=device)
    else:
        model_ema = None

    student_model.to(device)
    teacher_model.to(device)
    if args.distributed:
        student_model = DDP(student_model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=False)
        # teacher_model = DDP(teacher_model, device_ids=[args.gpu], output_device=args.gpu, find_unused_parameters=True)
    else:
        student_model = student_model.to(device)
        teacher_model = teacher_model.to(device)
    best_val_acc = 0.0 

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)

    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(student_model = student_model,
                                        teacher_model = teacher_model,
                                        train_loader = train_loader,
                                        criterion = criterion_distillation,
                                        optimizer = optimizer,
                                        scheduler = scheduler,
                                        grad_scaler = grad_scaler,
                                        mixup_fn = mixup_fn,
                                        model_ema = model_ema,
                                        device = device,
                                        epoch = epoch,
                                        args = args)
        val_metrics = validate(student_model, val_loader, criterion_distillation, device, args)
        if args.wandb and wandb.run is not None:
            wandb.log(train_metrics, step=epoch)
            wandb.log(val_metrics, step=epoch)

        logger.info(f"Epoch {epoch} - Train: {train_metrics} - Val: {val_metrics}")
        
        is_best = False
        current_val_acc = val_metrics.get('val_acc1', 0.0)
        print(f"Current val acc: {current_val_acc}")
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            is_best = True
        
        # if epoch % 10 == 0:  
        if not args.distributed or (args.distributed and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'model_state_dict': student_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler_state_dict': grad_scaler.state_dict() if grad_scaler is not None else None,
            }, is_best=is_best, filename=f'{args.save_dir}/checkpoint.pth') 
            # 더 나은 파일 이름 있으면 그걸로 변경 및 tools/utils.py save_checkpoint 함수 수정할 거 있으면 같이 수정.

    logger.info("Training completed")
    logger.info("Final validation metrics:")
    logger.info(val_metrics)

    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()


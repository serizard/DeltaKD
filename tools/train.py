from model.models import VisionModelWrapper
from model.loss import DistillationLoss, call_base_loss
import argparse
from logs.logger import setup_logger, get_unique_log_file_path
import torch
import wandb
import os
from timm.data import Mixup
from dataset.datasets import DatasetBuilder
from tools.opt_sched import OptimizerFactory, Scheduler
from timm.utils import ModelEma
from torch.nn.parallel import DistributedDataParallel as DDP
from utils import setup_distributed, setup_device, seed_everything
from engine import train_one_epoch, validate
from utils import save_checkpoint, remove_module_prefix, enable_finetune_mode, get_model_state
from tools.augment import new_data_aug_generator


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for ViT knowledge distillation training')
    
    # Model arguments
    parser.add_argument('--teacher-model', type=str, default='deit_small_distilled_patch16_224',
                        help='Name of the teacher model architecture to use')
    parser.add_argument('--student-model', type=str, default='deit_tiny_patch16_224',
                        help='Name of the student model architecture to use')
    parser.add_argument('--fp16', action='store_true', 
                        help='Enable FP16 (half-precision) training')
    parser.add_argument('--input-size', default=224, type=int, 
                        help='Input image size for training')

    # Training hyperparameters
    parser.add_argument('--batch-size', type=int, default=256,
                        help='Batch size for training')
    parser.add_argument('--amp', action='store_true', 
                        help='Enable Automatic Mixed Precision training')
    parser.add_argument('--ema-decay', type=float, default=None,
                        help='Decay factor for model EMA (Exponential Moving Average)')
    parser.add_argument('--label-smoothing', type=float, default=0.1,
                        help='Label smoothing factor for loss calculation')
    parser.add_argument('--drop-path-rate', type=float, default=0.1,
                        help='Drop path rate for stochastic depth')
    parser.add_argument('--num-workers', type=int, default=10,
                        help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=300,
                        help='Number of epochs to train')
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin memory for faster data transfer to GPU')
    parser.set_defaults(pin_mem=True)

    # Optimizer
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: None, use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    # Scheduler
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')

    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Distributed training
    parser.add_argument('--gpus', type=str, default=None,
                        help='Comma-separated list of GPU indices to use for distributed training. '
                                 'If not specified, defaults to using all available GPUs or CPU if no GPUs are available.')
    parser.add_argument('--dist-url', default='env://',
                        help='url used to set up distributed training')

    # Distillation
    parser.add_argument('--distillation-type', type=str, 
                        choices=['none', 'soft', 'hard', 'vitkd', 'aaakd', 'vitkd_w_logit', 'aaakd_w_logit'], 
                        default='none',
                        help='Type of knowledge distillation to use')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='Weight factor for distillation loss')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='Temperature parameter for distillation')
    
    # Saving and logging
    parser.add_argument('--log-file', type=str, default='logs/train.log',
                        help='Path to save training logs')
    parser.add_argument('--save-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--wandb', action='store_true',
                        help='Enable Weights & Biases logging')
    parser.add_argument('--wandb-project', type=str, default='distill-vit',
                        help='Project name for Weights & Biases')
    
    # Data
    parser.add_argument('--data-path', type=str, default='dataset',
                        help='Root directory of the dataset')
    parser.add_argument('--dataset', type=str, default='imagenet-1k',
                        help='Name of the dataset to use')
    parser.add_argument('--eval-crop-ratio', default=0.875, type=float, 
                        help="Crop ratio for evaluation")

    # Augmentation parameters
    parser.add_argument('--mixup', type=float, default=0.8,
                        help='mixup alpha, mixup enabled if > 0. (default: 0.8)')
    parser.add_argument('--cutmix', type=float, default=1.0,
                        help='cutmix alpha, cutmix enabled if > 0. (default: 1.0)')
    parser.add_argument('--cutmix-minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup-prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup-switch-prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup-mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # (2) Random erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # (3) Other Augmentation params
    parser.add_argument('--color-jitter', type=float, default=0.3, metavar='PCT',
                        help='Color jitter factor (default: 0.3)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + \
                             "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1, help='Label smoothing (default: 0.1)')
    parser.add_argument('--interpolation', type=str, default='bicubic',
                        help='Training interpolation (random, bilinear, bicubic default: "bicubic")')

    parser.add_argument('--repeated-aug', action='store_true')
    parser.add_argument('--no-repeated-aug', action='store_false', dest='repeated_aug')
    parser.set_defaults(repeated_aug=True)
    
    parser.add_argument('--ThreeAugment', action='store_true')
    parser.add_argument('--src', action='store_true')
    parser.set_defaults(ThreeAugment=False)
    parser.set_defaults(src=False)

    
    # Miscellaneous settings
    parser.add_argument('--resume', action='store_true',
                        help='Resume training from checkpoint')
    parser.add_argument('--finetune', action='store_true',
                        help='Finetune from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint file for resume/finetune')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use for training (cpu/cuda)')
    
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

    if os.path.exists(args.log_file):
        args.log_file = get_unique_log_file_path(args.log_file)
    logger = setup_logger(args.log_file)
    logger.info(f"Training started with {args.teacher_model} as teacher and {args.student_model} as student")

    if args.wandb and (not args.distributed or args.rank == 0):
        logger.info("Wandb init")
        wandb.init(project=args.wandb_project, config=args, name="baseline_deit")

    dataset_builder = DatasetBuilder(args)
    train_loader = dataset_builder.build_loader(is_train=True)
    val_loader = dataset_builder.build_loader(is_train=False)  

    if args.ThreeAugment:
        train_loader.dataset.transform = new_data_aug_generator(args)

    optimizer = OptimizerFactory.create_optimizer(student_model, args)
    scheduler = Scheduler(optimizer, args)
    # grad_scaler = ScaledGradNorm(args)
    grad_scaler = None

    start_epoch = 0
    if args.checkpoint:
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        if args.resume:
            start_epoch = checkpoint['epoch']
            print(f"Starting from epoch: {start_epoch}")
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            if grad_scaler is not None and checkpoint.get('scaler') is not None:
                grad_scaler.load_state_dict(checkpoint['scaler'])

        student_state = remove_module_prefix(checkpoint['model'])
        if args.finetune:
            enable_finetune_mode(student_model, student_state)
        else:
            student_model.load_state_dict(student_state, strict=False)

    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=dataset_builder.num_classes)
    else:
        mixup_fn = None

    criterion_task = call_base_loss(args)
    criterion_distillation = DistillationLoss(base_criterion=criterion_task, teacher_model=teacher_model, distillation_type=args.distillation_type, alpha=args.alpha, tau=args.tau)

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
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            is_best = True
        print(f"Current val acc: {current_val_acc}")
        print(f"Best val acc: {best_val_acc}")
        
        if not args.distributed or (args.distributed and args.rank == 0):
            save_checkpoint({
                'epoch': epoch + 1,
                'model': get_model_state(student_model),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': grad_scaler.state_dict() if grad_scaler is not None else None,
            }, is_best=is_best, filename=f'{args.save_dir}/checkpoint.pth') 

    logger.info("Training completed")
    logger.info("Final validation metrics:")
    logger.info(val_metrics)

    if args.wandb:
        wandb.finish()

if __name__ == '__main__':
    main()


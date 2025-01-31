import os
import torch
import torch.distributed as dist
import numpy as np
import random
import shutil
import datetime

def setup_distributed(args):
    args.distributed = False
    if 'WORLD_SIZE' in os.environ:
        args.world_size = int(os.environ['WORLD_SIZE'])
        if args.world_size > 1:
            args.rank = int(os.environ['RANK'])
            args.gpu = int(os.environ['LOCAL_RANK'])
            args.distributed = True
        else:
            args.distributed = False
            args.rank = 0
            args.gpu = 0
    elif args.gpus is not None: # 사용자 지정 GPU가 있을 때 (torchrun 미사용)
    # Legacy multi-GPU using CUDA_VISIBLE_DEVICES, single node only
        gpu_list = [int(gpu) for gpu in args.gpus.split(',')]
        num_gpus = len(gpu_list)
        if num_gpus > 1:
            args.distributed = True
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus # Set visible GPUs
            args.rank = 0
            args.world_size = num_gpus
            args.gpu = 0
        else:
            args.distributed = False
            args.gpu = gpu_list[0] if gpu_list else 0
    else:
        print('Not using distributed mode')
        args.distributed = False
        args.rank = 0
        args.gpu = 0

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        args.dist_backend = 'nccl'
        print(f'| distributed init (rank {args.rank}): {args.dist_url}')
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
            timeout=datetime.timedelta(0, 1800),
        )
        dist.barrier()


def setup_device(args):
    if torch.cuda.is_available():
        if args.distributed:
            device = torch.device('cuda', args.gpu)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    return device


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(state, is_best, filename):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, filename.replace('pth', 'best.pth'))


def load_checkpoint(model, optimizer, scheduler, scaler, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return model, optimizer, scheduler, scaler


def load_model(model, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model


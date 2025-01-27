import os
import torch
import torch.distributed as dist
import numpy as np
import random
import shutil

def setup_distributed(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True
    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}')
    dist.init_process_group(
        backend=args.dist_backend, init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)
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


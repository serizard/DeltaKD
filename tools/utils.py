import os
import torch
import torch.distributed as dist
import numpy as np
import random
import shutil
import datetime

def remove_module_prefix(state_dict): # resume할 때.
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value
    return new_state_dict

def get_model_state(model):
    if hasattr(model, 'module'):
        return model.module.state_dict()
    return model.state_dict()

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
    elif args.gpus is not None: 
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
    else:
        print('Not using distributed mode')


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
    epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint['scaler'])
    return epoch, model, optimizer, scheduler, scaler


def load_model(model, filename):
    checkpoint = torch.load(filename, map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    return model


def enable_finetune_mode(model, model_ckpt):
    current_state_dict = model.state_dict()
    for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
        if k in model_ckpt and model_ckpt[k].shape != current_state_dict[k].shape:
            print(f"Removing key {k} from pretrained checkpoint")
            del model_ckpt[k]

    # interpolate position embedding
    pos_embed_checkpoint = model_ckpt['pos_embed']
    embedding_size = pos_embed_checkpoint.shape[-1]
    num_patches = model.patch_embed.num_patches
    num_extra_tokens = model.pos_embed.shape[-2] - num_patches
    # height (== width) for the checkpoint position embedding
    orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
    # height (== width) for the new position embedding
    new_size = int(num_patches ** 0.5)
    # class_token and dist_token are kept unchanged
    extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
    # only the position tokens are interpolated
    pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
    pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
    pos_tokens = torch.nn.functional.interpolate(
        pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
    new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
    model_ckpt['pos_embed'] = new_pos_embed

    model.load_state_dict(model_ckpt, strict=False)
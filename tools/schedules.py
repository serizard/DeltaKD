import math
import torch
from timm.scheduler import CosineLRScheduler

class Scheduler:
    def __init__(self, optimizer, args):
        self.args = args
        self.optimizer = optimizer
        self.base_lr = args.lr * args.batch_size / 512.0  # Learning rate scaling
        self._create_scheduler()

    def _create_scheduler(self):
        self.lr_scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.args.epochs,
            lr_min=1e-5,
            warmup_lr_init=0.0,
            warmup_t=self.args.warmup_epochs,
            cycle_limit=1,
            t_in_epochs=True,
        )

    def step(self, epoch, metric=None):
        self.lr_scheduler.step(epoch)

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']

class OptimizerFactory:
    @staticmethod
    def create_optimizer(model, args):
        # Parameter groups
        param_groups = OptimizerFactory._get_parameter_groups(model)
        
        if args.opt.lower() == 'adamw':
            optimizer = torch.optim.AdamW(
                param_groups,
                lr=args.lr,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=args.weight_decay
            )
        elif args.opt.lower() == 'sgd':
            optimizer = torch.optim.SGD(
                param_groups,
                lr=args.lr,
                momentum=0.9,
                weight_decay=args.weight_decay
            )
        else:
            raise ValueError(f"Unsupported optimizer: {args.opt}")
            
        return optimizer

    @staticmethod
    def _get_parameter_groups(model):
        decay = []
        no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
                
            if len(param.shape) == 1 or name.endswith(".bias") or \
                'layer_norm' in name or 'embedding' in name:
                no_decay.append(param)
            else:
                decay.append(param)
                
        return [
            {'params': decay},
            {'params': no_decay, 'weight_decay': 0.}
        ]


### Gradient Clipping ###

# class ScaledGradNorm:
#     def __init__(self, args):
#         self.args = args
#         self.grad_clip = args.grad_clip if hasattr(args, 'grad_clip') else None
#         self.grad_accum_steps = args.grad_accum_steps if hasattr(args, 'grad_accum_steps') else 1

#     def __call__(self, parameters):
#         if self.grad_clip is None:
#             return
            
#         # Scale gradients by grad accumulation steps
#         if self.grad_accum_steps > 1:
#             for p in parameters:
#                 if p.grad is not None:
#                     p.grad.data.div_(self.grad_accum_steps)
                    
#         # Clip gradients
#         torch.nn.utils.clip_grad_norm_(parameters, self.grad_clip)

# class AverageMeter:
#     def __init__(self):
#         self.reset()

#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0

#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
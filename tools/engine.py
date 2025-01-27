import torch
from logs.logger import MetricLogger
from timm.utils import accuracy, ModelEma
from loss import AttentionAwareLoss
import torch.nn as nn


criterion_task = nn.CrossEntropyLoss()
criterion_attn = AttentionAwareLoss()


def train_one_epoch(student_model, teacher_model, train_loader, optimizer, scheduler, grad_scaler, mixup_fn, model_ema, device, epoch, config, args):
    student_model.train()
    teacher_model.eval()
    metric_logger = MetricLogger()
    
    header = f'Epoch: [{epoch+1}/{args.epochs}]'
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(train_loader, 10, header)):
        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast(enabled=True):
            student_logits, student_feats = student_model(samples)

        with torch.no_grad():
            teacher_logits, teacher_feats = teacher_model(samples)

        loss_task = criterion_task(student_logits, targets)
        loss_attn = criterion_attn(teacher_feats['attn'], student_feats['attn'])
        
        loss = loss_task + config.LOSS_WEIGHT['attn_weight'] * loss_attn
        
        optimizer.zero_grad()
        loss.backward()
        grad_scaler(student_model.parameters())
        optimizer.step()
        
        if model_ema is not None:
            model_ema.update(student_model)
            
        scheduler.step_update(epoch * len(train_loader) + data_iter_step)
        
        batch_size = samples.shape[0]
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_loss_task=loss_task.item())
        metric_logger.update(train_loss_attn=loss_attn.item())
        metric_logger.update(train_lr=scheduler.get_lr())
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(student_model, val_loader, device, config, args):
    student_model.eval()
    criterion_task = nn.CrossEntropyLoss()
    metric_logger = MetricLogger()

    header = 'Test:'
    for samples, targets in metric_logger.log_every(val_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            student_logits, student_feats = student_model(samples)

        loss_task = criterion_task(student_logits, targets)
        loss_attn = criterion_attn(student_feats['attn'])

        loss = loss_task + config.LOSS_WEIGHT['attn_weight'] * loss_attn

        acc1, acc5 = accuracy(student_logits, targets, topk=(1, 5))

        batch_size = samples.shape[0]
        metric_logger.update(val_loss=loss.item())
        metric_logger.update(val_loss_task=loss_task.item())
        metric_logger.update(val_loss_attn=loss_attn.item())
        metric_logger.update(val_acc1=acc1.item())
        metric_logger.update(val_acc5=acc5.item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, val_loader, device, args):
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    
    metric_logger = MetricLogger()
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(val_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(samples)
            loss = criterion(outputs, targets)
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(acc5=acc5.item())
        
    # Return average stats
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
import torch
from logs.logger import MetricLogger
from timm.utils import accuracy, ModelEma
import torch.nn as nn

def train_one_epoch(student_model, teacher_model, train_loader, criterion, optimizer, loss_scaler, clip_grad, mixup_fn, model_ema, device, epoch, args):
# def train_one_epoch(student_model, teacher_model, train_loader, criterion, optimizer, grad_scaler, mixup_fn, model_ema, device, epoch, args):
    student_model.train()
    teacher_model.eval()
    metric_logger = MetricLogger()

    header = f'Epoch: [{epoch+1}/{args.epochs}]'
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(train_loader, 10, header, args.rank)):
        if mixup_fn is not None:
            original_targets = targets.to(device, non_blocking=True)
            samples, targets = mixup_fn(samples, targets)
            
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if args.amp:    
            with torch.cuda.amp.autocast(enabled=True):
                student_logits, student_feats = student_model(samples)
        else:
            student_logits, student_feats = student_model(samples)

        loss = criterion(samples, student_logits, student_feats, targets) # inputs, outputs, student_features, labels
        
        if not isinstance(student_logits, torch.Tensor):
            student_logits, _ = student_logits
        
        if mixup_fn is not None:
            acc1, acc5 = accuracy(student_logits, original_targets, topk=(1, 5))
        else:
            acc1, acc5 = accuracy(student_logits, targets, topk=(1, 5))
        
        optimizer.zero_grad()
        # loss.backward()
        # if grad_scaler is not None:
        #     grad_scaler(student_model.parameters())
        # optimizer.step()
        
        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad,
                    parameters=student_model.parameters(), create_graph=is_second_order)
        
        if model_ema is not None:
            model_ema.update(student_model)
        
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_acc1=acc1.item())
        metric_logger.update(train_acc5=acc5.item())
        metric_logger.update(train_lr=optimizer.param_groups[0]['lr'])
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def validate(student_model, val_loader, device, args):
    criterion = torch.nn.CrossEntropyLoss()
    student_model.eval()
    metric_logger = MetricLogger()

    header = 'Val:'
    for samples, targets in metric_logger.log_every(val_loader, 10, header, args.rank):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=True):
            student_logits, student_feats = student_model(samples)

        if not isinstance(student_logits, torch.Tensor):
            student_logits, _ = student_logits

        loss = criterion(student_logits, targets)

        acc1, acc5 = accuracy(student_logits, targets, topk=(1, 5))

        metric_logger.update(val_loss=loss.item())
        metric_logger.update(val_acc1=acc1.item())
        metric_logger.update(val_acc5=acc5.item())

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

@torch.no_grad()
def evaluate(model, val_loader, criterion, device, args):
    model.eval()
    
    metric_logger = MetricLogger()
    header = 'Test:'
    
    for samples, targets in metric_logger.log_every(val_loader, 10, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Forward
        with torch.cuda.amp.autocast(enabled=True):
            outputs, features = model(samples)
            loss = criterion(outputs, targets)
        
        # Measure accuracy
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        
        batch_size = samples.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.update(acc1=acc1.item())
        metric_logger.update(acc5=acc5.item())
        
    # Return average stats
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
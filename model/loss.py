import torch
from timm.loss import LabelSmoothingCrossEntropy
import torch.nn.functional as F

class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        assert distillation_type in ['none', 'soft', 'hard']
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, student_features, labels):
        outputs_kd = None
        if not isinstance(outputs, torch.Tensor):
            # assume that the model outputs a tuple of [outputs, outputs_kd]
            outputs, outputs_kd = outputs
        base_loss = self.base_criterion(outputs, labels)
        if self.distillation_type == 'none':
            return base_loss

        if outputs_kd is None:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            teacher_logits, teacher_features = self.teacher_model(inputs)

        if self.distillation_type == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()
        elif self.distillation_type == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_logits.argmax(dim=1))
        elif self.distillation_type == 'ViTKD':
            pass
        elif self.distillation_type == 'AAAKD':
            pass
        elif self.distillation_type == 'TokenKD':
            pass
        elif self.distillation_type == 'PositionKD':
            pass
        elif self.distillation_type == 'EntropyKD':
            pass
        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss



def call_base_loss(args):
    if args.dataset in ['cifar-100', 'cifar-10']:
        return LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    elif args.dataset in ['imagenet-1k', 'imagenet-21k']:
        return LabelSmoothingCrossEntropy(smoothing=args.label_smoothing)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Supported datasets are {['cifar-100', 'cifar-10', 'imagenet-1k', 'imagenet-21k']}")

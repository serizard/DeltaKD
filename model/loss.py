import torch
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
import torch.nn.functional as F
import torch.nn as nn
from model.dist_loss import ViTKDLoss
from model.vitkd_loss import vitkd_loss

"""
!!student & teacher model 간의 관계!!

각 layer 별 token 수는 동일해야 함.
임베딩 사이즈는 달라도 됨. 어차피 projection 이용해서 맞추면 되는 일.
layer 수는 동일한 것이 좋지만 아니어도 대처는 가능하지 않을까.
(ViT-T, S, B 모두 layer 12개, Large만 24개)

"""
class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, student_model: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.student_model = student_model
        self.teacher_model = teacher_model
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

        if outputs_kd is None and self.distillation_type in ['soft', 'hard']:
            raise ValueError("When knowledge distillation is enabled, the model is "
                             "expected to return a Tuple[Tensor, Tensor] with the output of the "
                             "class_token and the dist_token")
        # don't backprop throught the teacher
        with torch.no_grad():
            # teacher feature 같은 경우에는 원하는 layer의 output 저장. 
            # layer 같은 경우에는 사전에 지정할 수 있음.
            teacher_logits, teacher_features = self.teacher_model(inputs)

        """ 

        student_features, teacher_features 예상 format
        : [batch_size, num_tokens, embed_dim]

        """
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
        elif self.distillation_type == 'vitkd':

            # student_model이 DDP로 감싸져 있으면 .module을 사용하여 원본 모델 접근
            student_model = self.student_model.module if isinstance(self.student_model, torch.nn.parallel.DistributedDataParallel) else self.student_model
            
            student_dims = student_features['block_0'].shape[-1]
            teacher_dims = teacher_features['block_0'].shape[-1]
            # vitkd 설정을 forward()에서 수행
            # student_model.setup_vitkd_loss(student_dims=student_dims, teacher_dims=teacher_dims)
            student_model.to(student_features['block_0'].device)
            # ViTKD Loss 계산
            loss_lr_gen = vitkd_loss(student_model, student_features, teacher_features, 
                        alpha_vitkd=0.00003, beta_vitkd=0.000003, lambda_vitkd=0.5)
            loss_origin = base_loss
            vitkd_total_loss = loss_lr_gen + loss_origin
            return vitkd_total_loss
        
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
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax
    if mixup_active:
        return SoftTargetCrossEntropy()
    else:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)


from timm.scheduler import create_scheduler

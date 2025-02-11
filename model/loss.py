import torch
from timm.loss import SoftTargetCrossEntropy, LabelSmoothingCrossEntropy
import torch.nn.functional as F
import torch.nn as nn
from model.models import forward_with_features
from model.misc import random_masking, saliency_masking
import math
"""
!!student & teacher model 간의 관계!!

각 layer 별 token 수는 동일해야 함.
임베딩 사이즈는 달라도 됨. 어차피 projection 이용해서 맞추면 되는 일.
layer 수는 동일한 것이 좋지만 아니어도 대처는 가능하지 않을까.
(ViT-T, S, B 모두 layer 12개, Large만 24개)

"""
class DistillationLoss(torch.nn.Module):
    def __init__(self, base_criterion: torch.nn.Module, teacher_model: torch.nn.Module,
                 distillation_type: str, alpha: float, tau: float):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.distillation_type = distillation_type
        self.alpha = alpha
        self.tau = tau

    def forward(self, inputs, outputs, student_model, student_features, labels, args):
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

        with torch.no_grad():
            if self.distillation_type.lower() in ['soft', 'hard']:
                teacher_logits = self.teacher_model(inputs)
            else:
                teacher_logits, teacher_features = forward_with_features(self.teacher_model, inputs)
        """ 
        student_features, teacher_features 예상 format
        : [batch_size, num_tokens, embed_dim]
        """
        if self.distillation_type.lower() == 'soft':
            T = self.tau
            distillation_loss = F.kl_div(
                F.log_softmax(outputs_kd / T, dim=1),
                F.log_softmax(teacher_logits / T, dim=1),
                reduction='sum',
                log_target=True
            ) * (T * T) / outputs_kd.numel()

        elif self.distillation_type.lower() == 'hard':
            distillation_loss = F.cross_entropy(outputs_kd, teacher_logits.argmax(dim=1))

        elif self.distillation_type.lower() == 'vitkd':
            student_model = student_model.module if isinstance(student_model, torch.nn.parallel.DistributedDataParallel) else student_model
            
            student_model.to(inputs.device)
            distillation_loss = vitkd_loss(student_model, student_features, teacher_features, 
                        alpha_vitkd=0.00003, beta_vitkd=0.000003, lambda_vitkd=0.5)
            return base_loss + distillation_loss

        elif self.distillation_type.lower() == 'lrkd':
            student_model = student_model.module if isinstance(student_model, torch.nn.parallel.DistributedDataParallel) else student_model
            student_model.to(inputs.device)

            rank = args.lrkd_rank
            # CLS token 제거 및 projection
            student_features = [
                student_model.align[0](student_features[0][:, 1:]),
                student_model.align[1](student_features[1][:, 1:]),
                student_model.align[2](student_features[-1][:, 1:])
            ]

            # CLS, DIST token 제거
            teacher_features = [teacher_features[0][:, 2:], teacher_features[1][:, 2:], teacher_features[11][:, 2:]]
            distillation_loss = lrkd_loss(teacher_features, student_features, rank, alpha=args.lrkd_alpha, beta=args.lrkd_beta, gamma=args.lrkd_gamma)

        elif self.distillation_type.lower() == 'diffkd':
            student_model = student_model.module if isinstance(student_model, torch.nn.parallel.DistributedDataParallel) else student_model
            student_model.to(inputs.device)

            # CLS 토큰과 패치 임베딩 결합
            student_features = [
                student_model.align[0](student_features[0][:, 1:]),
                student_model.align[1](student_features[1][:, 1:]),
                student_model.align[2](student_features[-1][:, 1:])
            ]
            # teacher_features = [feat[:, 2:, :] for feat in teacher_features]  # CLS, DIST 토큰 제거
            teacher_features = [teacher_features[0][:, 2:], teacher_features[1][:, 2:], teacher_features[-1][:, 2:]]

            # Phase 1: Diffusion-Driven Feature Perturbation
            T = 8  # diffusion steps
            t = torch.randint(0, T, (inputs.shape[0],), device=inputs.device)
            
            # Adaptive noise scheduling
            sigma_max = torch.where(t < T//2, 
                                  torch.tensor(0.3, device=inputs.device),
                                  torch.tensor(0.7, device=inputs.device))
            sigma_t = (1 - torch.cos(math.pi * t.float() / T)) * sigma_max
            
            # Feature matching loss with noise-aware weighting
            feat_loss = 0
            for s_feat, t_feat in zip(student_features, teacher_features):
                # Add noise to teacher features
                t_feat = t_feat / torch.norm(t_feat, p=2, dim=-1, keepdim=True)
                s_feat = s_feat / torch.norm(s_feat, p=2, dim=-1, keepdim=True)
                
                noise = torch.randn_like(t_feat) * sigma_t.view(-1, 1, 1)
                noisy_t_feat = t_feat + noise
                # Noise prediction by student
                pred_noise = student_model.denoise_fn(noisy_t_feat, t)
                feat_loss += F.mse_loss(pred_noise, noise)
                
                # Feature matching with adaptive weights
                w_t = 1 / (sigma_t ** 2 + 1e-8)
                feat_loss += w_t.mean() * F.mse_loss(s_feat, t_feat)

            feat_loss = feat_loss / len(student_features)

            # Combine losses
            lambda_feat = 5e-5
            distillation_loss = feat_loss * lambda_feat
    
        elif self.distillation_type.lower() == 'saliency_mgd':
            student_model = student_model.module if isinstance(student_model, torch.nn.parallel.DistributedDataParallel) else student_model
            student_model.to(inputs.device)

            distillation_loss = saliency_mgd_loss(student_model, student_features, teacher_features, args)
            print(f"distillation_loss: {distillation_loss}")
            print(f"base_loss: {base_loss}")
            return base_loss + distillation_loss


        loss = base_loss * (1 - self.alpha) + distillation_loss * self.alpha
        return loss


def call_base_loss(args):
    mixup_active = args.mixup > 0 or args.cutmix > 0. or args.cutmix_minmax
    if mixup_active:
        return SoftTargetCrossEntropy()
    else:
        return LabelSmoothingCrossEntropy(smoothing=args.smoothing)


def vitkd_loss(student_model, student_features, teacher_features, 
               alpha_vitkd=0.00003, beta_vitkd=0.000003, lambda_vitkd=0.5):
    
    """Forward function.
        0, 1, 11th layer 사용 (맨 앞 2개, 맨 뒤 1개)
        student_features block_0: torch.Size([256, 197, 192]) CLS token 포함
        teacher_features block_0 torch.Size([256, 198, 384]) CLS, DIST token 포함

        preds_S(List): [B*2*N*D, B*N*D], student's feature map
        preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
    """
    block_0_s = student_features[0][:, 1:, :]
    block_0_t = teacher_features[0][:, 2:, :]
    block_1_s = student_features[1][:, 1:, :]
    block_1_t = teacher_features[1][:, 2:, :]
    block_11_s = student_features[-1][:, 1:, :]
    block_11_t = teacher_features[-1][:, 2:, :]

    low_s = torch.stack([block_0_s, block_1_s], dim=1)
    low_t = torch.stack([block_0_t, block_1_t], dim=1)
    high_s = block_11_s
    high_t = block_11_t

    B = low_s.shape[0]
    loss_mse = nn.MSELoss(reduction='sum')

    '''ViTKD: Mimicking'''
    if student_model.align2 is not None:
        for i in range(2):
            if i == 0:
                xc = student_model.align2[i].to(low_s.device)(low_s[:,i]).unsqueeze(1)
            else:
                xc = torch.cat((xc, student_model.align2[i].to(low_s.device)(low_s[:,i]).unsqueeze(1)),dim=1)
    else:
        xc = low_s

    loss_lr = loss_mse(xc, low_t) / B * alpha_vitkd

    '''ViTKD: Generation'''
    if student_model.align is not None:
        x = student_model.align(high_s)
    else:
        x = high_s

    # Mask tokens
    B, N, D = x.shape
    x, mat, ids, ids_masked = random_masking(x, lambda_vitkd)
    mask_tokens = student_model.mask_token.repeat(B, N - x.shape[1], 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
    mask = mat.unsqueeze(-1)

    hw = int(N**0.5)
    x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
    x = student_model.generation(x).flatten(2).transpose(1,2)

    loss_gen = loss_mse(torch.mul(x, mask), torch.mul(high_t, mask))
    loss_gen = loss_gen / B * beta_vitkd / lambda_vitkd
        
    return loss_lr + loss_gen


def lrkd_loss(teacher_features, student_features, rank=10, alpha=0.1, beta=0.1, gamma=0.1):
    loss_mse = nn.MSELoss(reduction='mean')
    losses = []
    for t_feat, s_feat in zip(teacher_features, student_features):
        t_feat = t_feat.reshape(-1, t_feat.size(-1))
        s_feat = s_feat.reshape(-1, s_feat.size(-1))
        
        U, S, _ = torch.linalg.svd(t_feat, full_matrices=False)
        U_k = U[:, :rank]
        S_k = torch.diag(S[:rank])
        aligned_t_feat = torch.mm(U_k, S_k)

        loss = loss_mse(aligned_t_feat, s_feat)
        losses.append(loss)

    return losses[0] * alpha + losses[1] * beta + losses[2] * gamma

# 1. [CLS] 토큰 제외 + Self-Attention → Attention Weight 사용
# 2. [CLS] 포함 + Self-Attention → [CLS] 부분의 Attention Weight 추출
# 3. Cross-Attention: [CLS]를 Query, 패치들을 Key/Value로 → Attention Weight 사용

def saliency_mgd_loss(student_model, student_features, teacher_features, args):
    loss_mse = nn.MSELoss(reduction='mean')

    student_features = student_model.align(student_features[-1][:, 1:])
    teacher_features = teacher_features[-1]
    B, N, D = student_features.shape

    x, mat, ids = saliency_masking(student_model, teacher_features, student_features, args.saliency_mask_ratio, args.saliency_method)

    mask_tokens = student_model.mask_token.repeat(B, N - x.shape[1], 1)
    x_ = torch.cat([x, mask_tokens], dim=1)
    x = torch.gather(x_, dim=1, index=ids.unsqueeze(-1).repeat(1, 1, D))
    mask = mat.unsqueeze(-1)

    hw = int(N**0.5)
    x = x.reshape(B, hw, hw, D).permute(0, 3, 1, 2)
    x = student_model.generation(x).flatten(2).transpose(1,2)

    teacher_features = teacher_features[:, 2:]
    loss_gen = loss_mse(torch.mul(x, mask), torch.mul(teacher_features, mask))
        
    return loss_gen
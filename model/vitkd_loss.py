import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def vitkd_loss(student_model, student_features, teacher_features, 
               alpha_vitkd=0.00003, beta_vitkd=0.000003, lambda_vitkd=0.5):
    
    """Forward function.
        0, 1, 11th layer 사용 (맨 앞 2개, 맨 뒤 1개)
        student_features block_0: torch.Size([256, 197, 192]) CLS token 포함
        teacher_features block_0 torch.Size([256, 198, 384]) CLS, DIST token 포함

        preds_S(List): [B*2*N*D, B*N*D], student's feature map
        preds_T(List): [B*2*N*D, B*N*D], teacher's feature map
    """
    block_0_s = student_features['block_0'][:, 1:, :]
    block_0_t = teacher_features['block_0'][:, 2:, :]
    block_1_s = student_features['block_1'][:, 1:, :]
    block_1_t = teacher_features['block_1'][:, 2:, :]
    block_11_s = student_features['block_11'][:, 1:, :]
    block_11_t = teacher_features['block_11'][:, 2:, :]

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

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    ids_masked = ids_shuffle[:, len_keep:L]

    x_keep = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_keep, mask, ids_restore, ids_masked

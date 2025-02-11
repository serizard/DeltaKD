import torch
import torch.nn as nn


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


# 1. [CLS] 토큰 제외 + Self-Attention → Attention Weight 사용
# 2. [CLS] 포함 + Self-Attention → [CLS] 부분의 Attention Weight 추출
# 3. Cross-Attention: [CLS]를 Query, 패치들을 Key/Value로 → Attention Weight 사용
def saliency_masking(student_model, teacher_feat, student_feat, mask_ratio, method):
    """
    Saliency masking 함수
    ----------------------
    입력:
      student_model: 학생 모델 (saliency_attn 모듈 포함)
      teacher_feat: 교사 feature. method에 따라 [CLS] 토큰 포함 여부가 달라짐.
      student_feat: 학생 feature (보통 패치 토큰만 포함)
      mask_ratio: 마스킹할 비율
      method: masking 방식을 선택 (1, 2, 3)
         1: [CLS] 토큰 제외 + Self-Attention → 각 토큰의 self-attention diagonal 사용
            - teacher_feat와 student_feat는 이미 [CLS] 토큰 없이 패치 토큰만 포함
         2: [CLS] 포함 + Self-Attention → [CLS] 토큰에서 패치 토큰에 대한 attention weight 추출
            - teacher_feat는 [CLS] 토큰과 패치 토큰 모두 포함 (CLS 토큰은 index 0)
         3: Cross-Attention: [CLS]를 Query, 패치들을 Key/Value로 사용하여 attention weight 계산
            - teacher_feat는 [CLS] 토큰과 패치 토큰 모두 포함.
    
    반환:
      x_keep: masking 되지 않은, 선택된 student feature
      mask: 바이너리 마스크 (0: keep, 1: masked)
      ids_restore: 원본 토큰 순서 복원을 위한 인덱스
    """
    import torch

    if method == 1:
        # Method 1: [CLS] 토큰 제외 + Self-Attention
        # teacher_feat와 student_feat는 이미 [CLS] 토큰 제거 후 패치 토큰만 포함한다고 가정.
        teacher_feat = teacher_feat[:, 2:] # [CLS], [DIST] 토큰 제거
        N, L, D = teacher_feat.shape  # L: 패치 토큰 수
        len_keep = int(L * (1 - mask_ratio))
        
        # self-attention을 통해 각 패치의 "자기 자신에 대한" attention 강도를 구함.
        attn_weights = student_model.saliency_attn(teacher_feat)  # shape: [N, L]
        
        ids_shuffle = torch.argsort(attn_weights, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        
        x_keep = torch.gather(student_feat, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([N, L], device=student_feat.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_keep, mask, ids_restore

    elif method == 2:
        # Method 2: [CLS] 포함 + Self-Attention → [CLS] 토큰의 attention weight 사용
        # teacher_feat는 [CLS] 토큰과 패치 토큰 모두 포함 (shape: [B, 1 + 패치 수, D])
        teacher_feat = torch.cat([teacher_feat[:, :1], teacher_feat[:, 2:]], dim=1) # [DIST] 토큰 제거
        N, L, D = teacher_feat.shape
        L_patch = L - 1  # 패치 토큰 수
        len_keep = int(L_patch * (1 - mask_ratio))
        B = N

        # student_model.saliency_attn는 SimpleAttention 인스턴스로, 내부에 qk 레이어가 있음.
        num_heads = student_model.saliency_attn.num_heads
        head_dim = D // num_heads
        scale = head_dim ** -0.5

        # teacher_feat로부터 q와 k 계산 (qk linear layer 이용)
        qk = student_model.saliency_attn.qk(teacher_feat)  # shape: [B, L, 2*D]
        q, k = torch.chunk(qk, 2, dim=-1)  # 각각 shape: [B, L, D]

        # 헤드별로 분리
        q = q.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)  # [B, num_heads, L, head_dim]
        k = k.reshape(B, L, num_heads, head_dim).permute(0, 2, 1, 3)

        # [CLS] 토큰에서의 query를 사용하여 전체 토큰에 대한 attention을 계산.
        # [CLS] 토큰은 teacher_feat의 첫 번째 토큰 (index 0)
        q_cls = q[:, :, 0:1, :]  # [B, num_heads, 1, head_dim]
        attn_logits = (q_cls @ k.transpose(-2, -1)) * scale  # [B, num_heads, 1, L]
        attn = attn_logits.softmax(dim=-1)  # [B, num_heads, 1, L]

        # 헤드 평균을 취함
        attn_weights_cls = attn.mean(dim=1).squeeze(1)  # [B, L]
        # CLS 토큰 외의, 패치 토큰에 대한 attention weight 추출
        attn_patch = attn_weights_cls[:, 1:]  # [B, L_patch]
        
        ids_shuffle = torch.argsort(attn_patch, dim=1)  # 낮은 값부터 정렬
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        
        # student_feat는 [B, L_patch, D]라고 가정.
        x_keep = torch.gather(student_feat, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([B, L_patch], device=student_feat.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        return x_keep, mask, ids_restore

    elif method == 3:
        # Method 3: Cross-Attention: [CLS]를 Query, 패치들을 Key/Value로 사용
        # teacher_feat는 [CLS] 토큰과 패치 토큰 모두 포함 (shape: [B, 1 + 패치 수, D])
        teacher_feat = torch.cat([teacher_feat[:, :1], teacher_feat[:, 2:]], dim=1) # [DIST] 토큰 제거
        N, L, D = teacher_feat.shape
        L_patch = L - 1
        len_keep = int(L_patch * (1 - mask_ratio))
        B = N

        # [CLS] 토큰과 패치 토큰 분리
        cls_token = teacher_feat[:, :1, :]    # [B, 1, D]
        patch_tokens = teacher_feat[:, 1:, :]   # [B, L_patch, D]

        # Cross-Attention을 이용하여 [CLS]가 각 패치에 주는 attention 계산
        attn_weights = student_model.saliency_attn(cls_token, patch_tokens)  # 예상 output shape: [B, L_patch] 또는 [B, 1, L_patch]
        if attn_weights.dim() == 3 and attn_weights.size(1) == 1:
            attn_weights = attn_weights.squeeze(1)  # [B, L_patch]
        
        ids_shuffle = torch.argsort(attn_weights, dim=1)  # 낮은 값 순 정렬
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_keep = ids_shuffle[:, :len_keep]
        
        # student_feat는 [B, L_patch, D]라고 가정.
        x_keep = torch.gather(student_feat, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        
        mask = torch.ones([B, L_patch], device=student_feat.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_keep, mask, ids_restore

    else:
        raise ValueError(f"Invalid saliency masking method: {method}")
    


# def saliency_masking(student_model, teacher_feat, student_feat, mask_ratio, method):
#     N, L, D = teacher_feat.shape  # batch, length, dim
#     L -= 1 # exclude CLS token
#     len_keep = int(L * (1 - mask_ratio))

#     attn_weights = student_model.saliency_attn(teacher_feat)
#     attn_weights= attn_weights[:, 1:] 

#     # Sort based on attention weights
#     ids_shuffle = torch.argsort(attn_weights, dim=1)  # ascend: small is keep, large is remove
#     ids_restore = torch.argsort(ids_shuffle, dim=1)
    
#     # Keep the first subset (lowest attention weights)
#     ids_keep = ids_shuffle[:, :len_keep]
#     ids_masked = ids_shuffle[:, len_keep:L]
    
#     # Select the kept features from student_feat
#     x_keep = torch.gather(student_feat, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
    
#     # Generate binary mask: 0 is keep, 1 is remove
#     mask = torch.ones([N, L], device=student_feat.device)
#     mask[:, :len_keep] = 0
#     # Unshuffle to get the binary mask
#     mask = torch.gather(mask, dim=1, index=ids_restore)
    
#     return x_keep, mask, ids_restore, ids_masked

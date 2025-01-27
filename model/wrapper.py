import torch.nn as nn 


class DistillWrapper(nn.Module):
    def __init__(self, model, is_teacher=False):
        super().__init__()
        self.model = model
        self.is_teacher = is_teacher
        self.features = {}
        
        for idx, block in enumerate(self.model.blocks):
            block.register_forward_hook(self._get_hook(idx))

        if self.is_teacher:
            for param in self.model.parameters():
                param.requires_grad = False
            
    def _get_hook(self, layer_idx):
        def hook(module, input, output):
            self.features[f"attn_{layer_idx}"] = output[1] 
        return hook
    
    def forward(self, x):
        self.features.clear()
        x = self.model.patch_embed(x)
        x = self.model.pos_drop(x + self.model.pos_embed)
        
        # Save positional embeddings
        self.features['pos_embed'] = self.model.pos_embed
        
        for blk in self.model.blocks:
            x = blk(x)
            
        x = self.model.norm(x)
        return self.model.head(x[:, 0]), self.features
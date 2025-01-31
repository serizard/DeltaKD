import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union
from collections import OrderedDict
from functools import partial
import timm

class DistilledModel(nn.Module):
    def __init__(self, model_name: str, pretrained: bool = True, args: dict = None):
        super().__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, drop_path_rate=args.drop_path_rate)
        self.features = {}
        self._register_hooks()

        if args.fp16:
            self.model = self.model.half()
        else:
            self.model = self.model.float()
        
    def get_features(self, name: str, module: nn.Module, input: torch.Tensor, output: torch.Tensor):
        if isinstance(output, (list, tuple)):
            self.features[name] = output[0]
        else:
            self.features[name] = output
            
    def _register_hooks(self):
        if 'swin' in self.model.__class__.__name__.lower():
            self._register_swin_hooks()
        else:
            self._register_vit_hooks()
    
    def _register_swin_hooks(self):
        """Register hooks for Swin Transformer layers"""
        # Stage outputs
        for i, stage in enumerate(self.model.layers):
            stage.register_forward_hook(
                partial(self.get_features, f'stage_{i}')
            )
            
            # Block outputs in each stage
            for j, block in enumerate(stage.blocks):
                # MHSA output
                block.attn.register_forward_hook(
                    partial(self.get_features, f'stage_{i}_block_{j}_attn')
                )
                # FFN output
                block.mlp.register_forward_hook(
                    partial(self.get_features, f'stage_{i}_block_{j}_ffn')
                )
    
    def _register_vit_hooks(self):
        """Register hooks for ViT layers"""
        for i, block in enumerate(self.model.blocks):
            # Full block output
            block.register_forward_hook(
                partial(self.get_features, f'block_{i}')
            )
            # MHSA output
            block.attn.register_forward_hook(
                partial(self.get_features, f'block_{i}_attn')
            )
            # FFN output
            block.mlp.register_forward_hook(
                partial(self.get_features, f'block_{i}_ffn')
            )
    
    @torch.no_grad()
    def get_layer_outputs(
        self,
        layer_type: str = 'block',
        indices: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in self.features.items()
                if layer_type in k and (indices is None or int(k.split('_')[-2]) in indices)}
    
    @torch.no_grad()
    def get_feature_types(self) -> Dict[str, List[str]]:
        types = {}
        for key in self.features.keys():
            feat_type = key.split('_')[-1]
            if feat_type not in types:
                types[feat_type] = []
            types[feat_type].append(key)
        return types

    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def train(self, mode: bool = True):
        super().train(mode)
        self.model.train(mode)
        return self
        
    def eval(self):
        super().eval()
        self.model.eval()
        return self
    
    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)
    
    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        return self.model.load_state_dict(state_dict, strict)
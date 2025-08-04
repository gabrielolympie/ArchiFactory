import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops import einsum
from einops import repeat

from copy import deepcopy

def count_parameters(stack):
    total_parameters = sum(p.numel() for p in stack.parameters())
    ffn_parameters = sum(p.numel() for layer in stack.layers for p in layer.ffn_module.parameters())
    mixin_parameters = sum(p.numel() for layer in stack.layers for p in layer.mixin_module.parameters())
    
    print(f"Total parameters: {total_parameters:,}")
    print(f"Mixin parameters: {mixin_parameters:,}")
    print(f"FFN parameters: {ffn_parameters:,}")

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    
class ResidualBlock(nn.Module):
    def __init__(self, module, hidden_size):
        super().__init__()
        self.module = module
        self.rms_norm = RMSNorm(hidden_size, eps = 1e-6)
        
    def forward(self, hidden_states):
        residual = hidden_states
        hidden_states = self.rms_norm(hidden_states)
        hidden_states = self.module(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states
    
class MixinBlock(nn.Module):
    def __init__(self, hidden_size, mixin_module, ffn_module, ):
        super().__init__()
        self.mixin_module = ResidualBlock(mixin_module, hidden_size)
        self.ffn_module = ResidualBlock(ffn_module, hidden_size)
        
    def forward(self, hidden_states):
        
        if self.mixin_module is not None:
            hidden_states = self.mixin_module(hidden_states)
            
        if self.ffn_module is not None:
            hidden_states = self.ffn_module(hidden_states)
        return hidden_states

class StackedMixinBlock(nn.Module):
    def __init__(self, num_layers, hidden_size, initializer_range=1e-2, mixin_module=None, ffn_module=None, positionnal_module=None):
        super().__init__()
        self.layers = nn.ModuleList([MixinBlock(hidden_size, deepcopy(mixin_module), deepcopy(ffn_module)) for _ in range(num_layers)])
        
        for i, layer in enumerate(self.layers):
            layer.layer_id = i
            layer.mixin_module.layer_id = i
            layer.ffn_module.layer_id = i
            
        self.positionnal_module = positionnal_module
            
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.bias is not None:
                    module.bias.data.zero_()
                    
            elif isinstance(module, nn.Embedding):
                module.weight.data.normal_(mean=0.0, std=initializer_range)
                if module.padding_idx is not None:
                    module.weight.data[module.padding_idx].zero_()   
        
        
    def forward(self, hidden_states):
        if self.positionnal_module is not None:
            hidden_states = hidden_states + self.positionnal_module(hidden_states)
            
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return hidden_states
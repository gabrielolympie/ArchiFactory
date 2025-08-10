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
    embedding_parameters = sum(p.numel() for p in stack.embedding_module.parameters())
    ffn_parameters = sum(p.numel() for layer in stack.stacked_mixin_block.layers for p in layer.ffn_module.parameters())
    mixin_parameters = sum(p.numel() for layer in stack.stacked_mixin_block.layers for p in layer.mixin_module.parameters())
    lm_head_parameters = sum(p.numel() for p in stack.lm_head_module.parameters())
    
    print(f"Total parameters: {total_parameters:,}")
    print(f"Embedding parameters: {embedding_parameters:,}")
    print(f"Mixin parameters: {mixin_parameters:,}")
    print(f"FFN parameters: {ffn_parameters:,}")
    print(f"LM Head parameters: {lm_head_parameters:,}")

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

def init_weight(module, initializer_range=1e-2):
    for name, named_module in module.named_modules():
        if isinstance(named_module, nn.Linear):
            named_module.weight.data.normal_(mean=0.0, std=initializer_range)
            if named_module.bias is not None:
                named_module.bias.data.zero_()
                
        elif isinstance(named_module, nn.Embedding):
            named_module.weight.data.normal_(mean=0.0, std=initializer_range)
            if named_module.padding_idx is not None:
                named_module.weight.data[named_module.padding_idx].zero_()  
        

class StackedMixinBlock(nn.Module):
    def __init__(self, num_layers, hidden_size, initializer_range=1e-2, mixin_module=None, ffn_module=None, positionnal_module=None):
        super().__init__()
        self.layers = nn.ModuleList([MixinBlock(hidden_size, deepcopy(mixin_module), deepcopy(ffn_module)) for _ in range(num_layers)])
        
        for i, layer in enumerate(self.layers):
            layer.layer_id = i
            layer.mixin_module.layer_id = i
            layer.ffn_module.layer_id = i
            
        ## Init
        init_weight(self.layers, initializer_range)

        self.positionnal_module = positionnal_module
        if self.positionnal_module is not None:
            init_weight(self.positionnal_module, initializer_range)
        
    def forward(self, hidden_states):
        if self.positionnal_module is not None:
            hidden_states = hidden_states + self.positionnal_module(hidden_states)
            
        for layer in self.layers:
            hidden_states = layer(hidden_states)
            
        return hidden_states
    
class StackedMixinForCausalLM(nn.Module):
    def __init__(
        self,
        num_layers,
        hidden_size,
        initializer_range=1e-2,
        embedding_module=None,
        final_norm_module=None,
        lm_head_module=None,
        mixin_module=None,
        ffn_module=None,
        positionnal_module=None,
        freeze_lm_modules=True,
        vocab_size=None,
    ):
        super().__init__()
        self.embedding_module = deepcopy(embedding_module)
        self.final_norm_module = deepcopy(final_norm_module)
        self.lm_head_module = deepcopy(lm_head_module)
        self.stacked_mixin_block = StackedMixinBlock(num_layers, hidden_size, initializer_range, mixin_module, ffn_module, positionnal_module)
            
        # Init modules if null
        if self.embedding_module is None:
            assert vocab_size is not None, "vocab_size must be provided if embedding_module is None"
            self.embedding_module = nn.Embedding(vocab_size, hidden_size)
            init_weight(self.embedding_module, initializer_range)
        if self.final_norm_module is None:
            self.final_norm_module = RMSNorm(hidden_size)
            init_weight(self.final_norm_module, initializer_range)
        if self.lm_head_module is None:
            assert vocab_size is not None, "vocab_size must be provided if lm_head_module is None"
            self.lm_head_module = nn.Linear(hidden_size, vocab_size)
            init_weight(self.lm_head_module, initializer_range)
        
        if freeze_lm_modules:
            for name, parameters in self.embedding_module.named_parameters():
                parameters.requires_grad = False
                
            for name, parameters in self.final_norm_module.named_parameters():
                parameters.requires_grad = False
            
            for name, parameters in self.lm_head_module.named_parameters():
                parameters.requires_grad = False
            
        self.vocab_size = self.embedding_module.weight.shape[0]
        self.loss_fn = torch.nn.CrossEntropyLoss()
        
    def forward(self, input_ids, return_loss=False):
        hidden_states = self.embedding_module(input_ids)
        hidden_states = self.stacked_mixin_block(hidden_states)
        hidden_states = self.final_norm_module(hidden_states)
        logits = self.lm_head_module(hidden_states)
        
        if return_loss:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            
            shift_labels = shift_labels.to(shift_logits.device)
            loss = self.loss_fn(shift_logits, shift_labels)
            return loss
        
        return logits 
            
        
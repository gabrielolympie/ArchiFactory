import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange, einsum, repeat

from fla.layers import MultiScaleRetention, Mamba2, RWKV6Attention
    
class RNNMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=None, num_key_value_heads=None):
        super().__init__()
        self.rnn = nn.RNN(hidden_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False)
        
    def forward(self, x):
        output, _ = self.rnn(x)
        return output

class LSTMMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=None, num_key_value_heads=None):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=1, dropout=0.0, bidirectional=False)
        
    def forward(self, x):
        output, _ = self.lstm(x)
        return output
    
class MultiScaleRetentionMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None):
        super().__init__()
        
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        self.msr = MultiScaleRetention(
            hidden_size=hidden_size,
            num_heads=num_attention_heads,
            num_kv_heads=num_key_value_heads
        )
        
    def forward(self, x):
        output, _, _ = self.msr(x)
        return output
   
class Mamba2Mixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None):
        super().__init__()
        self.layer_idx = 0
        head_dim = hidden_size // num_attention_heads
        
        self.mamba2 = Mamba2(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            head_dim=head_dim,
            state_size=8,
            expand=1,
            n_groups=1,
            chunk_size=8,
            
        )
        
    def forward(self, x):
        self.mamba2.layer_idx = self.layer_idx
        output = self.mamba2(hidden_states =x)
        return output

class RWKV6Mixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None):
        super().__init__()
        self.layer_idx = 0
        self.rwkv6 = RWKV6Attention(
            hidden_size = hidden_size,
            num_heads = num_attention_heads,
        )
        
    def forward(self, x):
        self.rwkv6.layer_idx = self.layer_idx
        
        output, _, _ = self.rwkv6(x)
        return output
 
class GroupedQuerySelfAttentionMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, num_key_value_heads=None, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads is not None else num_attention_heads
        self.dropout = dropout
        
        assert self.num_attention_heads % self.num_key_value_heads == 0, (
            f"num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
        )
        
        self.head_dim = hidden_size // num_attention_heads
        assert self.head_dim * num_attention_heads == self.hidden_size, (
            f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size} "
            f"and `num_attention_heads`: {num_attention_heads})."
        )
        
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden_size, self.num_key_value_heads * self.head_dim, bias=bias)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
    def forward(self, x, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x)  
        k = self.k_proj(x)  
        v = self.v_proj(x)  
        
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        attn_output = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
            enable_gqa=(self.num_key_value_heads != self.num_attention_heads)
        )
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, self.hidden_size)
        output = self.out_proj(attn_output)
        
        return output
    
class MultiHeadLatentAttentionMixin(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, latent_size=None, dropout=0.0, bias=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.latent_size = latent_size if latent_size is not None else hidden_size // 4  # Default to 1/4 of hidden size
        
        self.dropout = dropout
        
        self.head_dim = hidden_size // num_attention_heads
        
        assert self.head_dim * num_attention_heads == self.hidden_size, (
            f"hidden_size must be divisible by num_attention_heads (got `hidden_size`: {self.hidden_size} "
            f"and `num_attention_heads`: {num_attention_heads})."
        )
        
        # Projections for input tokens
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Projections for latent tokens
        self.latent_q_proj = nn.Linear(self.latent_size, hidden_size, bias=bias)
        self.latent_k_proj = nn.Linear(self.latent_size, hidden_size, bias=bias)
        self.latent_v_proj = nn.Linear(self.latent_size, hidden_size, bias=bias)
        
        # Output projection
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=bias)
        
        # Learnable latent tokens (initialized randomly)
        self.latent_tokens = nn.Parameter(torch.randn(1, self.latent_size, self.latent_size))
        
    def forward(self, x, attn_mask=None, is_causal=False):
        batch_size, seq_len, _ = x.shape
        
        # Get latent tokens for this batch
        latent = self.latent_tokens.expand(batch_size, -1, -1)
        
        # Project input tokens
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Project latent tokens
        latent_q = self.latent_q_proj(latent)
        latent_k = self.latent_k_proj(latent)
        latent_v = self.latent_v_proj(latent)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        latent_q = latent_q.view(batch_size, self.latent_size, self.num_attention_heads, self.head_dim).transpose(1, 2)
        latent_k = latent_k.view(batch_size, self.latent_size, self.num_attention_heads, self.head_dim).transpose(1, 2)
        latent_v = latent_v.view(batch_size, self.latent_size, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # Compute cross attention between latent queries and input keys/values
        attn_output = F.scaled_dot_product_attention(
            latent_q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal
        )
        
        # Compute cross attention between input queries and latent keys/values
        latent_attn_output = F.scaled_dot_product_attention(
            q, latent_k, latent_v,
            attn_mask=None,  # Typically no mask for latent attention
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False
        )
        
        # Combine the attention outputs
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, self.latent_size, self.hidden_size)
        latent_attn_output = latent_attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        
        # You can choose how to combine these - here we just return the latent-attended input tokens
        # Alternatively, you could concatenate or add them
        output = self.out_proj(latent_attn_output)
        
        return output

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange, einsum, repeat

from fla.layers import MultiScaleRetention,Mamba2,RWKV6Attention
from modules.archi_modules import RMSNorm
    
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
            num_kv_heads=num_key_value_heads,
            expand_k=1.0,
            expand_v=1.0
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
            state_size=32,
            expand=1,
            n_groups=1,
            chunk_size=128,
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
            expand_k=0.5,
            expand_v=0.5,
            proj_low_rank_dim=64,
            gate_low_rank_dim=64
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
        
    def forward(self, x, attn_mask=None, is_causal=True):
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
    def __init__(
        self, 
        hidden_size,
        num_attention_heads,
        num_key_value_heads = None,
        q_latent_dim = None,
        kv_lora_dim = None,
        dropout=0.0
    ):
        super().__init__()
        
        # assert config.v_head_dim is not None , f"v_head_dim is not defined {config.v_head_dim=}"
        # assert config.q_lora_rank is not None , f"q_lora_rank is not defined {config.q_lora_rank=}"
        # assert config.kv_lora_rank is not None , f"kv_lora_rank is not defined {config.kv_lora_rank=}"
        # assert config.rope_head_dim is not None , f"rope_head_dim is not defined {config.rope_head_dim=}"
        
        # self.config = config
        
        ## Example from deepseek v2 lite config
        # "hidden_size": 2048,
        # "kv_lora_rank": 512, // hidden_size / 4
        # "num_attention_heads" : 16, 
        # "num_key_value_heads": 16,
        # "qk_nope_head_dim": 128, // hidden_size / 16
        # "qk_rope_head_dim": 64, // hidden_size / 32
        
        self.dim = hidden_size
        self.num_heads = num_attention_heads
        self.v_head_dim = hidden_size // num_attention_heads
        
        self.nope_head_dim = max(hidden_size // num_attention_heads, 64) ## avoid getting too small
        self.rope_head_dim = max(hidden_size // (num_attention_heads * 2), 32) ## avoid getting too small
        
        if q_latent_dim is None:
            q_latent_dim = hidden_size // 2
            
        if kv_lora_dim is None:
            kv_lora_dim = hidden_size // 4
        
        self.q_lora_rank = q_latent_dim
        self.kv_lora_rank = kv_lora_dim
        
        self.dropout = dropout
        
        # note: head dim of query and key if different from head dim of value
        
        # (attention_dim == num_head*head_dim) > d_model in deepseekv2
        # this is dim between wV and wQ
        self.value_dim = self.num_heads * self.v_head_dim
        
        # this is dims between wQ and wK
        self.nope_dim = self.num_heads * self.nope_head_dim
        self.rope_dim = self.num_heads * self.rope_head_dim  
        
        # query compression
        self.compress_q_linear = nn.Linear(self.dim, self.q_lora_rank, bias=False)  # W_DQ
        
        self.decompress_q_nope = nn.Linear(self.q_lora_rank, self.nope_dim, bias=False)
        self.decompress_q_rope = nn.Linear(self.q_lora_rank, self.rope_dim, bias=False)
        
        self.q_norm = RMSNorm(self.q_lora_rank)
        
        
        # key and value compression
        self.compress_kv_linear = nn.Linear(self.dim, self.kv_lora_rank, bias=False)  # W_DKV
        self.decompress_k_nope = nn.Linear(self.kv_lora_rank, self.nope_dim, bias=False)
        self.decompress_v_linear = nn.Linear(self.kv_lora_rank, self.value_dim, bias=False)
        self.kv_norm = RMSNorm(self.kv_lora_rank)
        
        
        self.k_rope_linear = nn.Linear(self.dim, self.rope_head_dim  , bias=False)
        # self.rope_norm = RMSNorm(self.rope_dim) # not in deepseekv2

        self.proj = nn.Linear(self.value_dim , self.dim, bias=False)
        self.res_dropout = nn.Dropout(p=dropout)
        
        
    def forward(self, x: Tensor):
        batch_size, seq_len, _ = x.shape

        compressed_q = self.compress_q_linear(x)
        norm_q = self.q_norm(compressed_q)
        query_nope:Tensor = self.decompress_q_nope(norm_q)
        query_rope:Tensor = self.decompress_q_rope(norm_q)

        compressed_kv = self.compress_kv_linear(x)
        norm_kv = self.kv_norm(compressed_kv)
        key_nope: Tensor = self.decompress_k_nope(norm_kv)
        value: Tensor = self.decompress_v_linear(norm_kv)
        
        key_rope:Tensor = self.k_rope_linear(x)
        # norm_rope = self.rope_norm(key_rope)

        query_nope = query_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        query_rope = query_rope.view(batch_size, seq_len, self.num_heads, self.rope_head_dim).transpose(1,2)
        
        key_rope = key_rope.view(batch_size, seq_len, 1, self.rope_head_dim).transpose(1,2)
        key_nope = key_nope.view(batch_size, seq_len, self.num_heads, self.nope_head_dim).transpose(1,2)
        
        value = value.view(batch_size, seq_len, self.num_heads, self.v_head_dim).transpose(1,2)
        
        # *** the line that fixes MLA :) ***
        # key_rope = key_rope/self.num_heads 

        # q_rope,k_rope = apply_rope(query_rope,key_rope, cis=freqs_cis)
        
        q_recombined = torch.empty((batch_size,self.num_heads,seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device).to(x.dtype)
        k_recombined = torch.empty((batch_size, self.num_heads, seq_len, self.rope_head_dim + self.nope_head_dim), device=x.device).to(x.dtype)
        
        q_recombined[:,:,:,:self.nope_head_dim] = query_nope
        # q_recombined[:,:,:,self.nope_head_dim:] = q_rope
        
        # k_rope = torch.repeat_interleave(k_rope, self.num_heads, dim=1) # >> you dont need to do this <<
        # 👇 broadcasting will do replication krope to all heads automagically
        k_recombined[:,:,:,:self.nope_head_dim] = key_nope
        # k_recombined[:,:,:,self.nope_head_dim:] = k_rope

        output = F.scaled_dot_product_attention(q_recombined, k_recombined, value, is_causal=True, dropout_p=self.dropout)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.v_head_dim)

        output = self.proj(output)
        output = self.res_dropout(output)
        return output
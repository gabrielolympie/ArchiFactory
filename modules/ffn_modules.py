import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops import einsum
from einops import repeat
    
class FFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size):
        super().__init__()
        self.hidden_size=hidden_size
        self.intermediate_size=intermediate_size
        
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj        
    
class SparseMoeFFN(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_experts, num_experts_per_tok, norm_topk_prob):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # gating
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FFN(hidden_size, intermediate_size) for _ in range(self.num_experts)]
        )

        # Expert balancing
        self.expert_count = torch.zeros(num_experts)
        self.expert_ema = torch.zeros(num_experts)
        self.expert_ema_alpha = 0.01  # Smoothing factor for EMA
        self.expert_beta = 0.1  # Weight for the balancing term

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # Training-time expert balancing
        if self.training:
            # Add expert balancing offset to router_logits
            expert_freq = self.expert_count / self.expert_count.sum()
            expert_offset = self.expert_beta * (1 / (expert_freq + 1e-6) - 1)
            router_logits = router_logits + expert_offset[None, :]

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)

        if self.norm_topk_prob:
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)

        # Update expert counts with exponential moving average
        if self.training:
            expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts)
            current_counts = expert_mask.sum(dim=(0, 1))
            self.expert_count = self.expert_ema_alpha * current_counts + (1 - self.expert_ema_alpha) * self.expert_count

        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states

        
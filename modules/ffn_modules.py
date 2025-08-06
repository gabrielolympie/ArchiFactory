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
    def __init__(self, hidden_size, intermediate_size, num_experts, num_experts_per_tok, norm_topk_prob,
                 expert_balancer_alpha: float = 0.999, expert_balancer_loss_weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

        # gating
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [FFN(hidden_size, intermediate_size) for _ in range(self.num_experts)]
        )

        # Expert balancing parameters
        self.expert_balancer_alpha = expert_balancer_alpha
        self.expert_balancer_loss_weight = expert_balancer_loss_weight
        # Using register_buffer so it's saved with state_dict and moved with model.to(device)
        self.register_buffer("expert_counts_ema", torch.zeros(num_experts))
        self.register_buffer("expert_counts_ema_steps", torch.tensor(0, dtype=torch.long))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        # Expert Balancing Logic (only applied in training mode)
        if self.training and self.expert_balancer_loss_weight > 0:
            # Determine which experts would be selected based on initial logits
            # This is done *before* applying softmax for the actual routing weights
            _, initial_selected_experts = torch.topk(router_logits, self.top_k, dim=-1)

            # Calculate expert selection counts for the current batch
            # One-hot encode selected_experts: (batch*seq_len, top_k, num_experts)
            selected_experts_one_hot = F.one_hot(initial_selected_experts, num_classes=self.num_experts)
            # Sum over the top_k dimension to get a mask for each token: (batch*seq_len, num_experts)
            selected_experts_mask = selected_experts_one_hot.sum(dim=1)
            # Sum over tokens to get total activations per expert in the current batch: (num_experts,)
            expert_activations_current_batch = selected_experts_mask.sum(dim=0).float()

            # Update Exponential Moving Average (EMA) of expert counts
            # Using mul_ and add_ for in-place operations on buffer
            self.expert_counts_ema.mul_(self.expert_balancer_alpha).add_(
                expert_activations_current_batch, alpha=(1 - self.expert_balancer_alpha)
            )
            # Increment step counter for EMA debiasing
            self.expert_counts_ema_steps.add_(1)
            
            # Apply debiasing to EMA, especially important during early training steps
            debias_factor = 1 - (self.expert_balancer_alpha ** self.expert_counts_ema_steps.item())
            expert_counts_ema_debiased = self.expert_counts_ema / (debias_factor + 1e-8)

            # Calculate average probability for each expert from EMA
            total_activations_ema = expert_counts_ema_debiased.sum()
            expert_probs_ema = expert_counts_ema_debiased / (total_activations_ema + 1e-8)

            # Calculate an offset to encourage exploration:
            # Experts with lower `expert_probs_ema` will have a larger (less negative) log,
            # so `offset` will be larger, boosting their logits.
            offset = -self.expert_balancer_loss_weight * torch.log(expert_probs_ema + 1e-8)
            
            # Add the calculated offset to the router_logits
            router_logits = router_logits + offset.unsqueeze(0)

        routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
        
        if self.norm_topk_prob:  # only diff with mixtral sparse moe block!
            routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
            
        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        expert_hitted = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hitted:
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx].squeeze(0))

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))

        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states #, router_logits
    

        
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from einops import rearrange
from einops import einsum
from einops import repeat

class NaivePositionnalEmbedding(nn.Module):
    def __init__(self, hidden_size, max_length = 512):
        super().__init__()
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
    def forward(self, hidden_size):
        batch_size, seq_len, _ = hidden_size.shape
        position_ids = torch.arange(seq_len, dtype=torch.long, device=hidden_size.device)
        position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        return self.position_embedding(position_ids)
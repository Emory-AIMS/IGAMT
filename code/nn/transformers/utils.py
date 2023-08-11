import torch
from torch import Tensor
import torch.nn.functional as f


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
    temp = query.bmm(key.transpose(1, 2))
    scale = query.size(-1) ** 0.5
    softmax = f.softmax(temp / scale, dim=-1)
    return softmax.bmm(value)


# different records may have different seq_len, but same dim_model
def position_embedding(seq_len: int, dim_model: int) -> Tensor:
    pos = torch.arange(seq_len, dtype=torch.float).reshape(1, -1, 1)
    dim = torch.arange(dim_model, dtype=torch.float).reshape(1, 1, -1)
    phase = pos / 1e4 ** (dim // dim_model) # extendable, infinity space
    return torch.where(dim.long() % 2 == 0, torch.sin(phase), torch.cos(phase)).cuda() # lookup rule


def feed_forward(dim_input: int = 512, dim_feedforward: int = 2048) -> torch.nn.Module:
    return torch.nn.Sequential(
        torch.nn.Linear(dim_input, dim_feedforward),
        torch.nn.ReLU(),
        torch.nn.Linear(dim_feedforward, dim_input),
    )


class AttentionHead(torch.nn.Module):
    def __init__(self, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.q = torch.nn.Linear(dim_in, dim_k)
        self.k = torch.nn.Linear(dim_in, dim_k)
        self.v = torch.nn.Linear(dim_in, dim_v)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return scaled_dot_product_attention(self.q(query), self.k(key), self.v(value))


class MultiHeadAttention(torch.nn.Module):
    def __init__(self, num_heads: int, dim_in: int, dim_k: int, dim_v: int):
        super().__init__()
        self.heads = torch.nn.ModuleList(
            [AttentionHead(dim_in, dim_k, dim_v) for _ in range(num_heads)]
        )
        self.linear = torch.nn.Linear(num_heads * dim_v, dim_in)

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return self.linear(
            torch.cat([h(query, key, value) for h in self.heads], dim=-1)
        )

class Residual(torch.nn.Module):
    def __init__(self, sublayer: torch.nn.Module, dimension: int, dropout: float = 0.1):
        super().__init__()
        self.sublayer = sublayer
        self.norm = torch.nn.LayerNorm(dimension)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, *tensors: Tensor) -> Tensor:
        # Assume that the "value" tensor is given last, so we can compute the
        # residual.  This matches the signature of 'MultiHeadAttention'.
        return self.norm(tensors[-1] + self.dropout(self.sublayer(*tensors)))

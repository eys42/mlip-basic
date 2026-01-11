from torch import nn, Tensor
import torch.nn.functional as F

class SDPABlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, batch_first: bool = True) -> None:
        super().__init__()
        self.d_model: int = d_model
        self.nhead: int = nhead
        self.dim_feedforward: int = dim_feedforward
        self.batch_first: bool = batch_first
        self.head_dim: int = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj: nn.Linear = nn.Linear(d_model, d_model)

        self.norm1: nn.LayerNorm = nn.LayerNorm(d_model)
        self.norm2: nn.LayerNorm = nn.LayerNorm(d_model)
        self.feedforward: nn.Sequential = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )

    def reshape_heads(self, t: Tensor, batch_size: int) -> Tensor:
        return t.reshape(batch_size, -1, self.nhead, self.head_dim).transpose(1, 2)

    def forward(self, x: Tensor) -> Tensor:
        result = x
        x = self.norm1(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        batch_size = x.size(0)
        q = self.reshape_heads(q, batch_size)
        k = self.reshape_heads(k, batch_size)
        v = self.reshape_heads(v, batch_size)
        
        attn_out = F.scaled_dot_product_attention(q, k, v)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, -1, self.d_model)
        x = result + self.out_proj(attn_out)
        x += self.feedforward(self.norm2(x))
        return x
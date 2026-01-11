from torch import Tensor, nn
from sdpa_block import SDPABlock

class Model(nn.Module):
    def __init__(self, in_features: int = 12, nhead: int = 4, d_model: int = 64, dim_feedforward: int = 128, num_layers: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Linear(in_features, d_model)
        self.transformer_layer_list = [SDPABlock(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, batch_first=True) for _ in range(num_layers)]
        self.transformer = nn.Sequential(*self.transformer_layer_list)
        self.regressor = nn.Linear(d_model, 1)
    def forward(self, x: Tensor) -> Tensor:
        if not x.is_contiguous():
            x = x.contiguous()
        h = self.embedding(x)
        h_out = self.transformer(h)
        h_out_padded = h_out.to_padded_tensor(padding=0.0)
        # sum across the atom dimension (dim=1)
        pooled = h_out_padded.sum(dim=1)
        return self.regressor(pooled)
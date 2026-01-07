import torch

class Model(torch.nn.Module):
    def __init__(self, in_features: int, nhead: int, hidden_features: int, pool_features: int, out_features: int=1) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.TransformerEncoderLayer(in_features, nhead=nhead, dim_feedforward=hidden_features),
        )
        self.linear = torch.nn.Linear(in_features, out_features)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    


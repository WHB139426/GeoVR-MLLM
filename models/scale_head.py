import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaleHead(nn.Module):
    def __init__(self, dim_in=2048):
        super().__init__()
        output_dim = 512
        self.backbone = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, 1),
        )

    def forward(self, x):
        # x shape: [B, dim_in]
        x = self.backbone(x).squeeze(-1)
        x = torch.exp(x) 
        return x
    
# x = torch.randn(16, 2048)
# model = ScaleHead(dim_in=2048)
# print(model(x).shape)
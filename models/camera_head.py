import torch
import torch.nn as nn
import torch.nn.functional as F

class CameraDec(nn.Module):
    def __init__(self, dim_in=2048):
        super().__init__()
        output_dim = 1536
        self.backbone = nn.Sequential(
            nn.LayerNorm(dim_in),
            nn.Linear(dim_in, output_dim),
            nn.GELU(),
            nn.Linear(output_dim, output_dim),
            nn.GELU(),
        )
        self.fc_t = nn.Linear(output_dim, 3)
        self.fc_qvec = nn.Linear(output_dim, 4)
        self.fc_fov = nn.Sequential(nn.Linear(output_dim, 2), nn.ReLU())

    def forward(self, feat):
        # feat shape: [B, T, dim_in]
        B, T, C = feat.shape
        feat_flat = feat.reshape(B * T, -1) # [B * T, dim_in]
        feat_flat = self.backbone(feat_flat)
        out_t = self.fc_t(feat_flat).reshape(B, T, 3)
        out_qvec = self.fc_qvec(feat_flat).reshape(B, T, 4)
        out_fov = self.fc_fov(feat_flat).reshape(B, T, 2)
        pose_enc = torch.cat([out_t, out_qvec, out_fov], dim=-1)
        
        return pose_enc
    
# model = CameraDec(dim_in=2048)
# feature = torch.randn(8, 16, 2048)
# out = model(feature)
# print(out.shape)
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class DistillHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        """
        :param in_channels: Qwen-VL feature's hidden_size
        :param out_channels: Geometry feature's hidden_size
        """
        super().__init__()

        self.norm = nn.LayerNorm(in_channels)
        self.fc1 = nn.Linear(in_channels, out_channels, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(out_channels, out_channels, bias=True)

    def forward(self, qwen_feat, target_spatial_shape=None):
        """
        :param qwen_feat: (B, T, H_q, W_q, D_q)
        :param target_spatial_shape: tuple (H_v, W_v)
        :return: (B, T, H_v, W_v, D_v)
        """
        B, T, H_q, W_q, D_q = qwen_feat.shape

        if target_spatial_shape is not None:
            H_v, W_v = target_spatial_shape
            if H_q!=H_v or W_q!=W_v:
                x = rearrange(qwen_feat, 'b t h w d -> (b t) d h w')
                x = F.interpolate(x, size=(H_v, W_v), mode='bilinear', align_corners=False)
                x = rearrange(x, '(b t) d h w -> b t h w d', b=B)
            else:
                x = qwen_feat
        else:
            x = qwen_feat

        x = self.norm(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

# model = DistillHead(in_channels=4096, out_channels=2048)
# qwen_feature = torch.randn(16, 15, 20, 4096)
# geo_feature = torch.randn(16, 30, 40, 2048)
# print(model(qwen_feature.unsqueeze(0), (geo_feature.shape[1], geo_feature.shape[2]))[0].shape)
# print(model(qwen_feature.unsqueeze(0), None)[0].shape)
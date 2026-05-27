# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Inspired by https://github.com/DepthAnything/Depth-Anything-V2

import math
from typing import Tuple
import einops

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Union

def position_grid_to_embed(pos_grid: torch.Tensor, embed_dim: int, omega_0: float = 100) -> torch.Tensor:
    """
    Convert 2D position grid (HxWx2) to sinusoidal embeddings (HxWxC)

    Args:
        pos_grid: Tensor of shape (H, W, 2) containing 2D coordinates
        embed_dim: Output channel dimension for embeddings

    Returns:
        Tensor of shape (H, W, embed_dim) with positional embeddings
    """
    H, W, grid_dim = pos_grid.shape
    assert grid_dim == 2
    pos_flat = pos_grid.reshape(-1, grid_dim)  # Flatten to (H*W, 2)

    # Process x and y coordinates separately
    emb_x = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 0], omega_0=omega_0)  # [1, H*W, D/2]
    emb_y = make_sincos_pos_embed(embed_dim // 2, pos_flat[:, 1], omega_0=omega_0)  # [1, H*W, D/2]

    # Combine and reshape
    emb = torch.cat([emb_x, emb_y], dim=-1)  # [1, H*W, D]

    return emb.view(H, W, embed_dim)  # [H, W, D]

def make_sincos_pos_embed(embed_dim: int, pos: torch.Tensor, omega_0: float = 100) -> torch.Tensor:
    """
    This function generates a 1D positional embedding from a given grid using sine and cosine functions.

    Args:
    - embed_dim: The embedding dimension.
    - pos: The position to generate the embedding from.

    Returns:
    - emb: The generated 1D positional embedding.
    """
    assert embed_dim % 2 == 0
    device = pos.device
    omega = torch.arange(embed_dim // 2, dtype=torch.float32 if device.type == "mps" else torch.double, device=device)
    omega /= embed_dim / 2.0
    omega = 1.0 / omega_0**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out)  # (M, D/2)
    emb_cos = torch.cos(out)  # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.float()

def create_uv_grid(
    width: int, height: int, aspect_ratio: float = None, dtype: torch.dtype = None, device: torch.device = None
) -> torch.Tensor:
    """
    Create a normalized UV grid of shape (width, height, 2).

    The grid spans horizontally and vertically according to an aspect ratio,
    ensuring the top-left corner is at (-x_span, -y_span) and the bottom-right
    corner is at (x_span, y_span), normalized by the diagonal of the plane.

    Args:
        width (int): Number of points horizontally.
        height (int): Number of points vertically.
        aspect_ratio (float, optional): Width-to-height ratio. Defaults to width/height.
        dtype (torch.dtype, optional): Data type of the resulting tensor.
        device (torch.device, optional): Device on which the tensor is created.

    Returns:
        torch.Tensor: A (width, height, 2) tensor of UV coordinates.
    """
    # Derive aspect ratio if not explicitly provided
    if aspect_ratio is None:
        aspect_ratio = float(width) / float(height)

    # Compute normalized spans for X and Y
    diag_factor = (aspect_ratio**2 + 1.0) ** 0.5
    span_x = aspect_ratio / diag_factor
    span_y = 1.0 / diag_factor

    # Establish the linspace boundaries
    left_x = -span_x * (width - 1) / width
    right_x = span_x * (width - 1) / width
    top_y = -span_y * (height - 1) / height
    bottom_y = span_y * (height - 1) / height

    # Generate 1D coordinates
    x_coords = torch.linspace(left_x, right_x, steps=width, dtype=dtype, device=device)
    y_coords = torch.linspace(top_y, bottom_y, steps=height, dtype=dtype, device=device)

    # Create 2D meshgrid (width x height) and stack into UV
    uu, vv = torch.meshgrid(x_coords, y_coords, indexing="xy")
    uv_grid = torch.stack((uu, vv), dim=-1)

    return uv_grid

def _make_dense_resize_layer(channels: int, resize_scale: float) -> nn.Module:
    if resize_scale == 1.0:
        return nn.Identity()

    if resize_scale == 0.5:
        return nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=3,
            stride=2,
            padding=1,
        )

    upsample_scale = int(resize_scale)
    return nn.ConvTranspose2d(
        in_channels=channels,
        out_channels=channels,
        kernel_size=upsample_scale,
        stride=upsample_scale,
        padding=0,
    )

def _make_prediction_head(in_channels: int, out_channels: int) -> nn.Module:
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=True)

def _init_small_conf_prediction_head(proj: nn.Module) -> None:
    if not isinstance(proj, nn.Conv2d):
        raise TypeError(f"Unsupported confidence projection layer: {type(proj)}")

    nn.init.zeros_(proj.weight)
    if proj.bias is None:
        raise ValueError("Small confidence init requires a bias term for proj_conf")

    # With expp1 confidence activation this starts from conf ~= 1.05.
    nn.init.constant_(proj.bias, math.log(1.05 - 1.0))

def _make_fusion_block(features: int, has_residual: bool = True) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=False),
        has_residual=has_residual,
    )

def _make_scratch(in_shape: list[int], out_shape: int) -> nn.Module:
    scratch = nn.Module()
    scratch.layer1_rn = nn.Conv2d(in_shape[0], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer2_rn = nn.Conv2d(in_shape[1], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer3_rn = nn.Conv2d(in_shape[2], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    scratch.layer4_rn = nn.Conv2d(in_shape[3], out_shape, kernel_size=3, stride=1, padding=1, bias=False)
    return scratch

class ResidualConvUnit(nn.Module):
    def __init__(self, features: int, activation: nn.Module) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        return out + x

class FeatureFusionBlock(nn.Module):
    def __init__(self, features: int, activation: nn.Module, has_residual: bool = True) -> None:
        super().__init__()
        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)
        self.has_residual = has_residual
        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation)
        self.resConfUnit2 = ResidualConvUnit(features, activation)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None = None, size: Tuple[int, int] | None = None) -> torch.Tensor:
        output = x
        if self.has_residual:
            if residual is None:
                raise ValueError("FeatureFusionBlock requires a residual tensor when has_residual=True")
            output = output + self.resConfUnit1(residual)

        output = self.resConfUnit2(output)
        output = custom_interpolate(output, size=size, mode="bilinear", align_corners=True)
        return self.out_conv(output)

def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] | None = None,
    scale_factor: float | None = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    if size is None:
        if scale_factor is None:
            raise ValueError("custom_interpolate requires either size or scale_factor")
        size = (
            int(x.shape[-2] * scale_factor),
            int(x.shape[-1] * scale_factor),
        )

    if tuple(x.shape[-2:]) == tuple(size):
        return x

    int_max = 1610612736
    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]
    if input_elements <= int_max:
        return F.interpolate(x, size=size, mode=mode, align_corners=align_corners)

    chunks = torch.chunk(x, chunks=(input_elements // int_max) + 1, dim=0)
    interpolated_chunks = [
        F.interpolate(
            chunk,
            size=size,
            mode=mode,
            align_corners=align_corners,
        )
        for chunk in chunks
    ]
    return torch.cat(interpolated_chunks, dim=0).contiguous()

class DenseHead(nn.Module):
    """Dense prediction head used by the released VGGT-Omega checkpoints."""

    def __init__(
        self,
        dim_in: int = 2048,
        patch_size: int = 16*2,
        target_patch_size: int = 14*2,
        features: int = 256,
        out_channels: list[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: list[int] = [4, 11, 17, 23],
    ) -> None:
        super().__init__()

        if patch_size % 4 != 0:
            raise ValueError(
                "DenseHead expects patch_size divisible by 4 because the fused feature is decoded "
                f"from 1/4 scale. Got patch_size={patch_size}."
            )

        self.patch_size = patch_size
        self.target_patch_size = target_patch_size

        self.intermediate_layer_idx = intermediate_layer_idx
        self.final_shuffle_factor = target_patch_size // 4
        self.norm = nn.LayerNorm(dim_in, eps=1e-5)

        self.projects = nn.ModuleList(
            [nn.Conv2d(in_channels=dim_in, out_channels=oc, kernel_size=1, stride=1, padding=0) for oc in out_channels]
        )
        self.resize_layers = nn.ModuleList(
            [
                _make_dense_resize_layer(channels=out_channels[0], resize_scale=4.0),
                _make_dense_resize_layer(channels=out_channels[1], resize_scale=2.0),
                _make_dense_resize_layer(channels=out_channels[2], resize_scale=1.0),
                _make_dense_resize_layer(channels=out_channels[3], resize_scale=0.5),
            ]
        )

        self.scratch = _make_scratch(out_channels, features)
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        self.proj = _make_prediction_head(
            features,
            self.final_shuffle_factor**2,
        )

    def _apply_pos_embed(self, x: torch.Tensor, width: int, height: int, ratio: float = 0.1) -> torch.Tensor:
        patch_w = x.shape[-1]
        patch_h = x.shape[-2]
        pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=width / height, dtype=x.dtype, device=x.device)
        pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        pos_embed = pos_embed * ratio
        pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        return x + pos_embed

    def scratch_forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        return self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])

    def forward(
        self,
        Original_H,
        Original_W,
        Target_H,
        Target_W,
        all_hidden_states: List[torch.Tensor],
    ):
        H, W = Original_H, Original_W
        T_H, T_W = Target_H, Target_W
        B = all_hidden_states[0].shape[0]
        T = all_hidden_states[0].shape[1]
        assert all_hidden_states[0].shape[2] == int(H*W/((self.patch_size)**2))

        patch_h, patch_w = H // (self.patch_size), W // (self.patch_size)
        target_patch_h, target_patch_w = T_H // (self.target_patch_size), T_W // (self.target_patch_size)

        multi_scale_features = []
        for feature_idx, layer_idx in enumerate(self.intermediate_layer_idx):
            x = all_hidden_states[layer_idx] # [B, T, patch_h*patch_w, dim_in] 
            x = self.norm(x)
            x = einops.rearrange(x, 'b t (h w) d -> (b t) d h w', h=patch_h, w=patch_w)
            x = self.projects[feature_idx](x)
            x = self._apply_pos_embed(x, W, H)
            x = self.resize_layers[feature_idx](x)
            multi_scale_features.append(x)

        fused = self.scratch_forward(multi_scale_features)

        if patch_h!=target_patch_h or patch_w!=target_patch_w:
            fused = custom_interpolate(
                fused,
                size = (target_patch_h*4, target_patch_w*4),
                mode="bilinear",
                align_corners=True,
            )

        fused = self._apply_pos_embed(fused, T_W, T_H)

        depth_logits = self.proj(fused)
        depth_logits = F.pixel_shuffle(depth_logits, self.final_shuffle_factor)
        depth_logits = depth_logits.permute(0, 2, 3, 1)
        
        depth = torch.exp(depth_logits)
        depth = depth.view(B, T, *depth.shape[1:]).squeeze(-1)

        return depth, None


# all_hidden_states = [torch.randn(1, 16, 300, 2048) for i in range(28+1)]
# print(len(all_hidden_states), all_hidden_states[0].shape)
# print("-------------------------------------")
# model = DenseHead(
#     dim_in = 2048,
#     patch_size = 16*2,
#     target_patch_size = 14*2,
#     features = 256,
#     out_channels = [256, 512, 1024, 1024],
#     intermediate_layer_idx = [4, 11, 17, 23],
#     )
# def get_parameter_number(model):
#     total_num = sum(p.numel() for p in model.parameters())
#     trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     return {'Total': total_num, 'Trainable': trainable_num} 
# print(get_parameter_number(model))
# pred_depth, pred_conf = model(Original_H=480, Original_W=640, Target_H=392, Target_W=532, all_hidden_states=all_hidden_states)
# print("-------------------------------------")
# print(pred_depth.shape, pred_conf)


# Copyright (c) 2026 EF-SyncNet Authors.
# Licensed under the Apache License, Version 2.0.
# For full attribution, please refer to the README.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.device import get_device

class EFCA(nn.Module):
    """
        Edge-Frequency Cross Attention
        Utilize the edge features (F_edge) to guide the wavelet enhanced features (F_wave)
    """
    def __init__(self, in_channels, reduction=4, num_heads=4):
        super(EFCA, self).__init__()
        inter_channels = in_channels // reduction

        # ensure multihead attention valid
        assert inter_channels % num_heads == 0, \
            f"inter_channels ({inter_channels}) must be divisible by num_heads ({num_heads})"



        self.q_proj = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.k_proj = nn.Conv2d(in_channels, inter_channels, kernel_size=1)
        self.v_proj = nn.Conv2d(in_channels, inter_channels, kernel_size=1)


        self.cross_att = nn.MultiheadAttention(embed_dim=inter_channels, num_heads=num_heads, batch_first=True)


        self.out_proj = nn.Sequential(
            nn.Conv2d(inter_channels, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

        self.alpha = nn.Parameter(torch.ones(1) * 0.5)

        self.out_proj_activation = nn.Identity()

    def forward(self, F_edge, F_wave):
        """
            F_edge: Edge features from EGMC (B, C, H, W)
            F_wave: Frequency domain features from the wavelet enhancement module (B, C, H, W)
        """
        B, C, H, W = F_wave.shape

        assert F_edge.shape[1] == C, f"F_edge channels ({F_edge.shape[1]}) != F_wave channels ({C})"
 
        inter_channels = self.q_proj.out_channels


        Q = self.q_proj(F_edge)
        K = self.k_proj(F_wave)
        V = self.v_proj(F_wave)

        # Flatten  (B, C, H*W) → (B, H*W, C)
        Q = Q.flatten(2).permute(0, 2, 1)
        K = K.flatten(2).permute(0, 2, 1)
        V = V.flatten(2).permute(0, 2, 1)

        # Cross Attention
        out, _ = self.cross_att(Q, K, V)  # (B, HW, C')


        attn_map = out.permute(0, 2, 1).view(B, inter_channels, H, W)  # (B, C', H, W)
        attn_to_in = self.out_proj(attn_map)  # (B, in_channels, H, W)


        F_out = F_wave + self.alpha * attn_to_in
        F_out = self.out_proj_activation(F_out) if hasattr(self, 'out_proj_activation') else F_out


        return F_out

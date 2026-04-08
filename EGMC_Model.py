# Copyright (c) 2026 EF-SyncNet Authors.
# Licensed under the Apache License, Version 2.0.
# For full attribution, please refer to the README.md.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.device import get_device

class EGMC(nn.Module):
    """Dual-Gated Edge-guided Multi-Context Module (Optimized for Noise Reduction)"""

    def __init__(self, in_channels):
        super().__init__()

        self.edge_3x3 = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
        self.edge_5x5 = nn.Conv2d(in_channels, 1, kernel_size=5, padding=2)
        self.edge_dilated = nn.Conv2d(in_channels, 1, kernel_size=3, padding=2, dilation=2)

        gaussian = torch.tensor([[1, 2, 1],
                                 [2, 4, 2],
                                 [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3) / 16.0
        

        sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
        laplace = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)


        self.register_buffer("gaussian", gaussian)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)
        self.register_buffer("laplace", laplace)

        self.to_gray = nn.Conv2d(in_channels, 1, kernel_size=1)


        self.sobel_threshold = nn.Parameter(torch.tensor(0.1)) 
        self.laplace_threshold = nn.Parameter(torch.tensor(0.1))


        self.edge_fuse = nn.Sequential(
            nn.Conv2d(5, 1, kernel_size=3, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True)
        )


        self.spatial_gate = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, 1),
            nn.Sigmoid()
        )
        self.context = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, 3, padding=3, dilation=3, groups=in_channels)
        ])
        self.project = nn.Conv2d(in_channels * 3, in_channels, 1)
        self.fuse = nn.Sequential(
            nn.Conv2d(in_channels + 1, in_channels, 3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )

    def high_freq(self, x_gray):

        x_smooth = F.conv2d(x_gray, self.gaussian, padding=1)


        sx = F.conv2d(x_smooth, self.sobel_x, padding=1)
        sy = F.conv2d(x_smooth, self.sobel_y, padding=1)
        sobel = torch.sqrt(sx**2 + sy**2 + 1e-6)


        laplace = F.conv2d(x_smooth, self.laplace, padding=1)

        laplace = torch.abs(laplace)


        sobel = F.relu(sobel - self.sobel_threshold)
        laplace = F.relu(laplace - self.laplace_threshold)

        return sobel, laplace

    def forward(self, x):

        e1 = torch.sigmoid(self.edge_3x3(x))
        e2 = torch.sigmoid(self.edge_5x5(x))
        e3 = torch.sigmoid(self.edge_dilated(x))

        x_gray = self.to_gray(x)
        sobel, laplace = self.high_freq(x_gray)


        edge_map = self.edge_fuse(torch.cat([e1, e2, e3, sobel, laplace], dim=1))


        feat = self.fuse(torch.cat([x, edge_map], dim=1))

        multi = [conv(feat) for conv in self.context]
        multi = self.project(torch.cat(multi, dim=1))


        alpha = self.spatial_gate(edge_map) * self.channel_gate(x)
        

        out = x + alpha * multi

        return out, edge_map

    def forward_viz(self, x):

        e1 = torch.sigmoid(self.edge_3x3(x))
        e2 = torch.sigmoid(self.edge_5x5(x))
        e3 = torch.sigmoid(self.edge_dilated(x))
        f_conv = (e1 + e2 + e3) / 3.0 


        x_gray = self.to_gray(x)

        sobel, laplace = self.high_freq(x_gray) 
        f_grad = (sobel + laplace) / 2.0


        raw_concat = torch.cat([e1, e2, e3, sobel, laplace], dim=1)
        f_edge = self.edge_fuse(raw_concat) 

        feat_temp = self.fuse(torch.cat([x, f_edge], dim=1))

        multi_outs = [conv(feat_temp) for conv in self.context]
        f_ctx = self.project(torch.cat(multi_outs, dim=1)) 

        # 返回字典
        return {
            "f_grad": f_grad,   
            "f_conv": f_conv,   
            "f_edge": f_edge,   
            "f_ctx": f_ctx      
        }
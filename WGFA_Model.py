# Copyright (c) 2026 EF-SyncNet Authors.
# Licensed under the Apache License, Version 2.0.
# For full attribution, please refer to the README.md.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.backbones import wavelet

class convbnrelu(nn.Module):
    def __init__(self, in_channel, out_channel, k=3, s=1, p=1, g=1, d=1, bias=False, bn=True, relu=True):
        super(convbnrelu, self).__init__()
        conv = [nn.Conv2d(in_channel, out_channel, k, s, p, dilation=d, groups=g, bias=bias)]
        if bn:
            conv.append(nn.BatchNorm2d(out_channel))
        if relu:
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)
    def forward(self, x):
        return self.conv(x)


class DSConv3x3(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1, dilation=1, relu=True):
        super(DSConv3x3, self).__init__()
        self.conv = nn.Sequential(
            convbnrelu(in_channel, in_channel, k=3, s=stride, p=dilation, d=dilation, g=in_channel),
            convbnrelu(in_channel, out_channel, k=1, s=1, p=0, relu=relu),
        )
    def forward(self, x):
        return self.conv(x)


class GradSemanticFusion(nn.Module):
    """
        Gradient + Semantic Dual-Guided High-Frequency and Low-Frequency Fusion Module.
        Input:
        high_sp: [B, C, H2, W2]  (High-frequency synthesis HL + LH + HH)
        low_sp:  [B, C, H2, W2]  (Low-frequency LL)
        Output:
        fre_flat: [B, N, C]  (Consistent with the format used by the subsequent self-attention, N = H2 * W2)
        Design Concept:
        1) Calculate the spatial gradient map (the gradient is weak in weak textures, but can be amplified when there are slight differences)
        2) Concatenate the gradient map with the semantic features of high/low to generate a position-level gate
        3) Apply sigmoid to the gate to compress it into [0, 1], and perform weighted fusion: fre = gate * high + (1 - gate) * low
        4) You can also choose to expand the gate to the channel dimension (optional, commented out)
    """
    def __init__(self, channels, hidden=128, per_channel=False):
        super(GradSemanticFusion, self).__init__()
        self.channels = channels
        self.per_channel = per_channel


        self.grad_conv = nn.Sequential(  
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False), #groups=channels 表示逐通道卷积
            nn.ReLU(inplace=True),
            # reduce to single map
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
            nn.Sigmoid()  
        )


        self.gate_conv = nn.Sequential(
            nn.Conv2d(channels * 2 + 1, hidden, kernel_size=1, bias=False), 
            nn.ReLU(inplace=True), 
            nn.Conv2d(hidden, 1, kernel_size=1, bias=True)  
        )


        if self.per_channel:
            self.channel_gate = nn.Sequential(
                nn.Linear(channels * 2 + 1, channels),
                nn.Sigmoid()
            )

    def forward(self, high_sp, low_sp):
        """
            high_sp, low_sp: [B, C, H2, W2]
            Return fre_flat: [B, N, C] (N = H2*W2)
            Therefore, the gradient provides: image gradient = intensity of pixel change; gradient guidance = guiding where to enhance high frequency;
            Spatial position signal (Where), the true indication of structural position;
            High frequency should play a role in "positions with structure".
            Semantic provides: which high frequency to enhance, which high frequency to suppress;
        """
        B, C, H2, W2 = high_sp.shape

        avg = 0.5 * (high_sp + low_sp)  # [B,C,H2,W2]
        grad_map = self.grad_conv(avg)  # [B,1,H2,W2],
                                        

        cat = torch.cat([high_sp, low_sp, grad_map], dim=1)  


        gate_logits = self.gate_conv(cat)  # [B,1,H2,W2]
        gate = torch.sigmoid(gate_logits)  # [B,1,H2,W2] 


        if self.per_channel:

            cat_flat = cat.view(B, -1, H2 * W2).permute(0, 2, 1)  # [B, N, 2C+1]
            channel_gates = self.channel_gate(cat_flat)  # [B, N, C]
 
            high_flat = high_sp.view(B, C, -1).permute(0, 2, 1)  # [B, N, C]
            low_flat = low_sp.view(B, C, -1).permute(0, 2, 1)   # [B, N, C]
            fre_flat = channel_gates * high_flat + (1 - channel_gates) * low_flat
            return fre_flat  # [B, N, C]


        fre_sp = gate * high_sp + (1.0 - gate) * low_sp  # [B,C,H2,W2]


        fre_flat = fre_sp.flatten(2).transpose(1, 2)  # [B, N, C]
        return fre_flat


class MultiheadAttention(nn.Module):
    def __init__(self, d_model, h, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert d_model % h == 0
        self.norm1 = nn.LayerNorm(d_model)
        self.pool = wavelet.WavePool(d_model)
        self.self_attn1 = nn.MultiheadAttention(d_model, h, dropout=dropout, batch_first=True)
        self.conv3x3 = DSConv3x3(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.fusion = GradSemanticFusion(channels=d_model, hidden=128, per_channel=False)

    def forward(self, query, key, value, H, W, attn_mask=None):
        b, n1, c = value.size()
        assert n1 == H * W, f"Input features do not match H and W: n1={n1}, H={H}, W={W}"

        feat = key.transpose(1, 2).view(b, c, H, W)
        LL, HL, LH, HH = self.pool(feat) 
        high_fre = HL + LH + HH
        low_fre = LL

        high_fre = high_fre.flatten(2).transpose(1, 2)
        low_fre = low_fre.flatten(2).transpose(1, 2)


        H2 = H // 2
        W2 = W // 2
        high_sp = high_fre.transpose(1, 2).view(b, c, H2, W2) 
        low_sp = low_fre.transpose(1, 2).view(b, c, H2, W2)
 
        fre = self.fusion(high_sp, low_sp)  
        
        x1 = self.self_attn1(query=query, key=fre, value=fre, attn_mask=None)[0]
        x1 = self.norm1(x1 + query)

        feat = self.conv3x3(feat).flatten(2).transpose(1, 2)
        x = x1 + feat
        x = self.out_norm(x)
        
        return x

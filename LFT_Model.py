# Copyright (c) 2026 EF-SyncNet Authors.
# Licensed under the Apache License, Version 2.0.
# For full attribution, please refer to the README.md.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmengine.device import get_device
import math

from mmseg.registry import MODELS
from ..utils import resize
from .decode_head import BaseDecodeHead

class LocalFrequencyTuning(nn.Module):
    """
        Local Frequency Tuning (LFT)
        - Perform local frequency adaptation at the decoder (local frequency tuning), applying gentle and learnable gains/attenuations for low/high frequencies.
        - Ensure interpretability: parameters have clear physical meanings (gamma controls the amplitude of high-frequency attenuation, beta controls low-frequency enhancement).
        - Focus on stability: small residual scaling, parameter boundedness, EMA smoothing statistics, warmup, protection against NaN/Inf.
        - Lightweight: use depthwise conv + 1x1 proj, with low computational cost, facilitating the insertion of multi-scale decoders.
        - Interface compatibility: forward(x, global_step=None), output has the same shape as the input, facilitating the replacement of FCB.
    """

    def __init__(self,
                 #参数组二
                 channels,
                 reduction=8,
                 gamma_init=0.02,       
                 gamma_max=0.035,       
                 per_channel_gamma=False,
                 beta=0.04,             
                 max_corr=0.35,          
                 warmup_steps=3000,
                 residual_scale_init=0.03,  
                 local_ws=3,            
                 ema_momentum=0.95,
                 use_gn=True,
                 eps=1e-6,
                 enable_wavelet_pool=True,
                 debug=False):

        super().__init__()

        self.debug = debug
        self.channels = channels
        self.gamma_max = float(gamma_max)
        self.beta = float(beta)
        self.max_corr = float(max_corr)
        self.warmup_steps = int(warmup_steps)
        self.eps = float(eps)
        self.per_channel = bool(per_channel_gamma)
        self.local_ws = int(local_ws)
        self.ema_momentum = float(ema_momentum)
        self.enable_wavelet_pool = bool(enable_wavelet_pool)


        if self.enable_wavelet_pool:
            try:
                from mmseg.models.backbones import wavelet
                self.pool = wavelet.WavePool(channels)
            except Exception:

                self.pool = None
        else:
            self.pool = None


        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(1, channels // reduction), 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(1, channels // reduction), channels, 1, bias=True),
            nn.Sigmoid()
        )


        if self.per_channel:
            self._gamma_param = nn.Parameter(torch.ones(channels) * gamma_init)
        else:
            self._gamma_param = nn.Parameter(torch.tensor(gamma_init, dtype=torch.float32))



        self._alpha_raw = nn.Parameter(torch.tensor(math.log(residual_scale_init + 1e-6), dtype=torch.float32))


        self.lf_pool = nn.AvgPool2d(kernel_size=self.local_ws, stride=self.local_ws, ceil_mode=True)
        self.lf_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.lf_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)


        self.hf_dw = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        self.hf_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)


        if use_gn:
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=max(1, min(32, channels // 8)), num_channels=channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True)
            )


        self.register_buffer('low_energy_ema', torch.zeros(1), persistent=False)
        self.register_buffer('high_energy_ema', torch.zeros(1), persistent=False)
        self.register_buffer('eps_buf', torch.tensor(self.eps))

   
        self._init_weights()

    def _init_weights(self):
      
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
     
        for n in self.fuse_conv:
            if isinstance(n, nn.Conv2d):
                nn.init.constant_(n.weight, 0.0)  
                if getattr(n, 'bias', None) is not None:
                    nn.init.zeros_(n.bias)

    def forward(self, x, global_step=None):
        """
        x: (B, C, H, W)
        global_step: int or None
        return: (B, C, H, W)
        """
        B, C, H, W = x.shape
        device = x.device


        pad_h = 0 if H % 2 == 0 else 1
        pad_w = 0 if W % 2 == 0 else 1
        if pad_h or pad_w:
            x_in = F.pad(x, (0, pad_w, 0, pad_h))
        else:
            x_in = x


        if self.pool is not None:
            try:
                LL, HL, LH, HH = self.pool(x_in)
                high = HL + LH + HH
                low = LL
            except Exception:

                low = F.avg_pool2d(x_in, kernel_size=2, stride=2)
                up_low = F.interpolate(low, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
                high = x_in - up_low
        else:
            low = F.avg_pool2d(x_in, kernel_size=2, stride=2)
            up_low = F.interpolate(low, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
            high = x_in - up_low


        low_energy = torch.mean(torch.abs(low), dim=(-1, -2))  
        high_energy = torch.mean(torch.abs(high), dim=(-1, -2)) # B x C

        low_mean = torch.mean(low_energy) 
        high_mean = torch.mean(high_energy)


        if self.low_energy_ema.item() == 0.0 and self.high_energy_ema.item() == 0.0:

            self.low_energy_ema = low_mean.detach()
            self.high_energy_ema = high_mean.detach()
        else:
           
            if self.training:
                self.low_energy_ema = self.ema_momentum * self.low_energy_ema + (1.0 - self.ema_momentum) * low_mean.detach()
                self.high_energy_ema = self.ema_momentum * self.high_energy_ema + (1.0 - self.ema_momentum) * high_mean.detach()


        freq_ratio = (torch.log1p(self.high_energy_ema) - torch.log1p(self.low_energy_ema)).clamp(min=-10.0, max=10.0)

        freq_ratio_b = freq_ratio * torch.ones((B, C, 1, 1), device=device)


        ch_protect = self.se(x)  # B x C x 1 x 1


        if self.per_channel:
            sig = torch.sigmoid(self._gamma_param).view(1, C)
        else:
            sig = torch.sigmoid(self._gamma_param)

        warm = 1.0
        if (self.warmup_steps > 0 and global_step is not None and global_step < self.warmup_steps):
          
            warm = (global_step / float(self.warmup_steps)) ** 2

        if self.per_channel:
            gamma = (self.gamma_max * sig * warm).view(1, C, 1, 1).to(device)
        else:
            gamma = (self.gamma_max * sig * warm) * torch.ones((1, C, 1, 1), device=device)

 
        freq_corr = (gamma * ch_protect) * freq_ratio_b
        freq_corr = torch.clamp(freq_corr, min=-self.max_corr, max=self.max_corr)

 
        denom = 1.0 + torch.abs(freq_corr)
        high_corrected = high / (denom + self.eps)

   
        low_corrected = low * (1.0 + self.beta)


        if high_corrected.shape[2] != x_in.shape[2] or high_corrected.shape[3] != x_in.shape[3]:
            high_up = F.interpolate(high_corrected, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
        else:
            high_up = high_corrected

        if low_corrected.shape[2] != x_in.shape[2] or low_corrected.shape[3] != x_in.shape[3]:
            low_up = F.interpolate(low_corrected, size=(x_in.shape[2], x_in.shape[3]), mode='bilinear', align_corners=False)
        else:
            low_up = low_corrected

        if pad_h or pad_w:
            high_up = high_up[:, :, :H, :W]
            low_up = low_up[:, :, :H, :W]

        # ---------------- Lo-Fi / Hi-Fi-lite  ----------------
        lf = self.lf_pool(x) if self.local_ws > 1 else x
        lf = self.lf_dw(lf)
        lf = self.lf_proj(lf)
        lf = F.interpolate(lf, size=(H, W), mode='bilinear', align_corners=False)

        hf = self.hf_dw(x)
        hf = self.hf_proj(hf)

        # recon 
        recon = high_up + low_up

        
        fused = hf + lf + recon

        # fuse conv + residual 
        recon_fused = self.fuse_conv(fused)


        alpha = torch.sigmoid(self._alpha_raw)

        out = x + alpha * recon_fused

        # 最后安全防护：如果有 NaN/Inf 则回退为输入（并打印 debug）
        if torch.isnan(out).any() or torch.isinf(out).any():
            if self.debug:
                print("[LFT] NaN/Inf detected in out -> fallback to x")
            out = x

        return out

    def flops(self, H, W):
        C = self.channels
        hf = H * W * C * 3
        lf = (H // max(1, self.local_ws)) * (W // max(1, self.local_ws)) * C * 3
        fuse = H * W * C * 1
        return hf + lf + fuse

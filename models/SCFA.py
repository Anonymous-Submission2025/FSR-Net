import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2d_BN(nn.Sequential):
    def __init__(self, in_ch, out_ch, ks, stride=1, padding=None):
        if padding is None:
            padding = ks // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, ks, stride, padding),
            nn.BatchNorm2d(out_ch)
        )

class SCFA(nn.Module):
   
    def __init__(self, in_channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  
        
        
        self.channel_excitation = nn.Sequential(
            nn.Linear(in_channels * 2, in_channels // 2),  
            nn.ReLU(),
            nn.Linear(in_channels // 2, in_channels),      
            nn.Sigmoid()
        )
        
       
        self.spatial_gate = nn.Sequential(
            Conv2d_BN(in_channels * 2, in_channels // 2, ks=1),
            nn.ReLU(),
            Conv2d_BN(in_channels // 2, 1, ks=1),
            nn.Sigmoid()
        )

    def forward(self, d, e):
        
        if d.shape[2:] != e.shape[2:]:
            d = F.interpolate(d, size=e.shape[2:], mode='bilinear', align_corners=True)
        
        
        b, c, _, _ = e.shape
        
        
        d_pool = self.avg_pool(d).view(b, c)
        e_pool = self.avg_pool(e).view(b, c)
        
        
        cat_pool = torch.cat([d_pool, e_pool], dim=1)  # [B, 2C]
        
        
        freq_weight = self.channel_excitation(cat_pool).view(b, c, 1, 1)  # [B, C, 1, 1]
        
        
        e_freq_refined = e * freq_weight
        
        
        cat_spatial = torch.cat([d, e_freq_refined], dim=1)
        spatial_weight = self.spatial_gate(cat_spatial)
        
        out = d + e_freq_refined * spatial_weight
        
        return out



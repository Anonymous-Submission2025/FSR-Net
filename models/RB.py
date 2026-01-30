import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Optional


class RB(nn.Module):
    
    
    def __init__(self, 
                 input_channels: int = 1,
                 num_clusters: int = 3,
                 frequency_analysis_samples: int = 100):
        super().__init__()
        
        self.input_channels = input_channels
        self.num_clusters = num_clusters
        
        self.frequency_analysis_samples = min(frequency_analysis_samples, 500)
        
        
        self.freq_topology = None 
        
        
        self.initialized = False
    
    def _analyze_frequency_energy(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, H, W = x.shape
        
       
        freq_H = H
        freq_W = W // 2 + 1
        
        avg_energy = torch.zeros(C, freq_H, freq_W, device=x.device)
        
        with torch.no_grad():
            samples_to_use = min(B, self.frequency_analysis_samples)
            for i in range(samples_to_use):
                # 使用 rfft2 处理实数图像
                x_fft = torch.fft.rfft2(x[i])
                # 计算能量 (幅度平方)
                energy = torch.abs(x_fft) ** 2
                avg_energy += energy
            avg_energy /= samples_to_use
       
        avg_energy = torch.log(avg_energy + 1e-8)
        
        
        for c in range(C):
            c_min = avg_energy[c].min()
            c_max = avg_energy[c].max()
            avg_energy[c] = (avg_energy[c] - c_min) / (c_max - c_min + 1e-8)
            
        return avg_energy

    def _initialize_topology(self, x: torch.Tensor):
        
        # print(f"[SILR-Block] Initializing frequency topology using K-Means on input shape {x.shape}...")
        
        B, C, H, W = x.shape
        freq_H = H
        freq_W = W // 2 + 1
        
        
        energy_map = self._analyze_frequency_energy(x) # [C, freq_H, freq_W]
        
        masks = []
        for c in range(C):
           
            flat_energy = energy_map[c].cpu().numpy().reshape(-1, 1)
            
            
            kmeans = KMeans(n_clusters=self.num_clusters, n_init=1, random_state=42)
            labels = kmeans.fit_predict(flat_energy)
            centers = kmeans.cluster_centers_.flatten()
            
            
            top_cluster_ids = np.argsort(centers)[-2:] 
            c_mask = np.zeros_like(labels, dtype=np.float32)
            for cid in top_cluster_ids:
                c_mask[labels == cid] = 1.0
            
            masks.append(c_mask.reshape(freq_H, freq_W))
            
        
        initial_mask = torch.tensor(np.stack(masks), dtype=torch.float32, device=x.device)
 
        initial_logits = torch.where(initial_mask > 0.5, 
                                     torch.tensor(2.0, device=x.device), 
                                     torch.tensor(-2.0, device=x.device))
        
        
        self.freq_topology = nn.Parameter(initial_logits.unsqueeze(0))
        self.initialized = True
        
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C, H, W]
        Returns:
            out: [B, C, H, W] (Added residual connection)
        """
        B, C, H, W = x.shape
        
       
        if not self.initialized or self.freq_topology is None:
            self._initialize_topology(x)
        
        
        x_rfft = torch.fft.rfft2(x, dim=(-2, -1))
        
        
        current_freq_shape = x_rfft.shape[-2:] # (H, W//2 + 1)
        param_shape = self.freq_topology.shape[-2:]
        
        if current_freq_shape != param_shape:
            
            topology_resized = F.interpolate(
                self.freq_topology, 
                size=current_freq_shape, 
                mode='bilinear', 
                align_corners=False
            )
        else:
            topology_resized = self.freq_topology

        soft_mask = torch.sigmoid(topology_resized)
        x_filtered_freq = x_rfft * soft_mask
        x_filtered_spatial = torch.fft.irfft2(x_filtered_freq, s=(H, W), dim=(-2, -1))
        
        
        return x + x_filtered_spatial

    def visualize_topology(self):
        
        if self.freq_topology is not None:
            with torch.no_grad():
                return torch.sigmoid(self.freq_topology)
        return None


    

    

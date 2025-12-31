"""
Baseline: pooling semplice (mean/max) di hidden states teacher
"""
import torch
import torch.nn as nn
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class PoolingBaseline:
    """Baseline con pooling di hidden states"""
    
    def __init__(self, config: dict, teacher):
        self.config = config
        self.teacher = teacher
        self.n_latents = config["n_latents"]
        self.d_lat = config["d_lat"]
        self.pooling_type = "mean"  # o "max"
        
        # Proiezione da d_teacher a d_lat
        self.proj = nn.Linear(config.get("d_teacher", 4096), self.d_lat)
        # Sposta su device quando necessario (lazy)
        self._device = None
    
    def pool_chunk(self, teacher_states: torch.Tensor) -> torch.Tensor:
        """
        Pooling di hidden states a N_lat vectors
        
        Args:
            teacher_states: [seq_len, d_teacher]
        
        Returns:
            [N_lat, d_lat]
        """
        # Sposta proj su device se necessario
        device = teacher_states.device
        if self._device != device:
            self.proj = self.proj.to(device)
            self._device = device
        
        seq_len = teacher_states.shape[0]
        
        if self.pooling_type == "mean":
            # Mean pooling
            pooled = teacher_states.mean(dim=0, keepdim=True)  # [1, d_teacher]
        else:
            # Max pooling
            pooled = teacher_states.max(dim=0)[0].unsqueeze(0)  # [1, d_teacher]
        
        # Proietta a d_lat
        pooled = self.proj(pooled)  # [1, d_lat]
        
        # Ripeti per avere N_lat vectors
        latents = pooled.repeat(self.n_latents, 1)  # [N_lat, d_lat]
        
        return latents
    
    def get_latents(self, context_chunks: List[str], device: torch.device) -> torch.Tensor:
        """
        Ottiene latents per tutti i chunk
        
        Returns:
            [num_chunks * N_lat, d_lat]
        """
        all_latents = []
        layer_indices = self.config.get("layer_indices", [-6])
        
        for chunk_text in context_chunks:
            if self.teacher.hidden_states_available:
                states = self.teacher.get_hidden_states(chunk_text, layer_indices)
                if states is not None:
                    states = states.to(device)
                    latents = self.pool_chunk(states)
                    all_latents.append(latents)
                else:
                    # Fallback: latents dummy
                    latents = torch.randn(self.n_latents, self.d_lat, device=device) * 0.01
                    all_latents.append(latents)
            else:
                # Fallback: latents dummy
                latents = torch.randn(self.n_latents, self.d_lat, device=device) * 0.01
                all_latents.append(latents)
        
        return torch.cat(all_latents, dim=0)


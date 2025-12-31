"""
Fase 4: Proiezione latents → reasoner space
"""
import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class LatentToReasonerProjection(nn.Module):
    """
    Proietta latents da d_lat a d_reasoner
    
    CRITICO: Latents sono off-manifold e devono essere proiettati
    nello spazio embedding del reasoner.
    """
    
    def __init__(self, d_lat: int, d_reasoner: int):
        """
        Args:
            d_lat: Dimensione latents (es. 512)
            d_reasoner: Dimensione embedding reasoner (es. 4096 per Qwen)
        """
        super().__init__()
        self.d_lat = d_lat
        self.d_reasoner = d_reasoner
        
        # Proiezione lineare
        self.projection = nn.Linear(d_lat, d_reasoner)
        
        # Layer norm per stabilità
        self.norm = nn.LayerNorm(d_reasoner)
    
    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latents: [N_lat, d_lat] o [batch, N_lat, d_lat]
        
        Returns:
            reasoner_embeds: [N_lat, d_reasoner] o [batch, N_lat, d_reasoner]
        """
        # Proietta
        reasoner_embeds = self.projection(latents)  # [..., d_reasoner]
        
        # Normalizza
        reasoner_embeds = self.norm(reasoner_embeds)
        
        return reasoner_embeds


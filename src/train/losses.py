"""
Loss functions per training compressore (Fase 5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CompressionLoss(nn.Module):
    """Loss combinata: primary + auxiliary"""
    
    def __init__(self, config: dict, auxiliary_type: str = "localization"):
        """
        Args:
            config: Config dict
            auxiliary_type: "localization" o "alignment"
        """
        super().__init__()
        self.auxiliary_type = auxiliary_type
        self.auxiliary_weight = config.get("train", {}).get("auxiliary_weight", 0.1)
        
        if auxiliary_type == "localization":
            # Opzione A: Answer localization
            # Predice quale chunk contiene la risposta
            # Nota: questo richiede di sapere quale chunk contiene la risposta
            # Per ora usiamo un approccio semplificato
            self.auxiliary_loss_fn = self._localization_loss
        elif auxiliary_type == "alignment":
            # Opzione B: Mean alignment
            self.auxiliary_loss_fn = self._alignment_loss
        else:
            raise ValueError(f"auxiliary_type deve essere 'localization' o 'alignment', trovato: {auxiliary_type}")
    
    def _localization_loss(self, latents: torch.Tensor, true_chunk_id: Optional[int] = None) -> torch.Tensor:
        """
        Loss per predire quale chunk contiene la risposta
        
        Args:
            latents: [num_chunks * N_lat, d_lat] o dict per chunk
            true_chunk_id: ID del chunk che contiene la risposta (opzionale)
        
        Returns:
            Loss scalar
        """
        # Per ora: loss semplificata che incoraggia latents diversi per chunk diversi
        # Se abbiamo true_chunk_id, possiamo fare classification
        # Altrimenti: variance loss per incoraggiare diversità
        
        if isinstance(latents, dict):
            # Latents per chunk separati
            chunk_latents = list(latents.values())
            if len(chunk_latents) > 1:
                # Calcola varianza tra chunk
                means = torch.stack([lat.mean() for lat in chunk_latents])
                variance = means.var()
                # Vogliamo alta varianza (chunk diversi)
                loss = -variance  # Negativo perché vogliamo massimizzare
            else:
                loss = torch.tensor(0.0, device=chunk_latents[0].device)
        else:
            # Latents concatenati: assumiamo N_lat per chunk
            # Questo è più complesso, per ora ritorna 0
            loss = torch.tensor(0.0, device=latents.device)
        
        return loss
    
    def _alignment_loss(self, latents: torch.Tensor, teacher_states_mean: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Loss per allineare mean(latents) ≈ mean(teacher_hidden_states)
        
        Args:
            latents: [N_lat, d_lat]
            teacher_states_mean: [d_teacher] mean dei teacher states (opzionale)
        
        Returns:
            Loss scalar
        """
        latent_mean = latents.mean(dim=0)  # [d_lat]
        
        if teacher_states_mean is not None:
            # Proietta teacher_states_mean a d_lat se necessario
            if teacher_states_mean.shape[0] != latents.shape[1]:
                # Dovremmo avere una proiezione, per ora skip
                return torch.tensor(0.0, device=latents.device)
            loss = F.mse_loss(latent_mean, teacher_states_mean)
        else:
            # Senza teacher mean: incoraggia latents a non collassare
            # Penalizza se mean è troppo vicino a zero
            loss = F.mse_loss(latent_mean, torch.zeros_like(latent_mean))
        
        return loss
    
    def forward(self, primary_loss: torch.Tensor, latents: torch.Tensor,
                auxiliary_data: Optional[dict] = None) -> torch.Tensor:
        """
        Calcola loss totale
        
        Args:
            primary_loss: Cross-entropy loss principale
            latents: Latents dal compressore
            auxiliary_data: Dict con dati per auxiliary loss
        
        Returns:
            Loss totale
        """
        # Primary loss
        total_loss = primary_loss
        
        # Auxiliary loss
        if self.auxiliary_type == "localization":
            true_chunk_id = auxiliary_data.get("true_chunk_id") if auxiliary_data else None
            aux_loss = self._localization_loss(latents, true_chunk_id)
        else:  # alignment
            teacher_mean = auxiliary_data.get("teacher_states_mean") if auxiliary_data else None
            aux_loss = self._alignment_loss(latents, teacher_mean)
        
        total_loss = total_loss + self.auxiliary_weight * aux_loss
        
        return total_loss, aux_loss.item() if isinstance(aux_loss, torch.Tensor) else 0.0


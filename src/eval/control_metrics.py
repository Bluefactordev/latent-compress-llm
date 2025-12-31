"""
Metriche OBBLIGATORIE per esperimento controllo (Fase 1)
"""
import torch
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


def compute_needle_accuracy(predictions: List[str], references: List[str], 
                           metadata: List[dict]) -> float:
    """Accuracy su needle facts"""
    correct = 0
    total = 0
    
    for pred, ref, meta in zip(predictions, references, metadata):
        qa_type = meta.get("qa", {}).get("type", "")
        if qa_type == "needle":
            total += 1
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_versioning_accuracy(predictions: List[str], references: List[str],
                               metadata: List[dict]) -> float:
    """Accuracy su versioning (latest vs old)"""
    correct = 0
    total = 0
    
    for pred, ref, meta in zip(predictions, references, metadata):
        qa_type = meta.get("qa", {}).get("type", "")
        if qa_type == "versioning":
            total += 1
            if pred.strip().lower() == ref.strip().lower():
                correct += 1
    
    return correct / total if total > 0 else 0.0


def compute_attention_entropy(attention_weights: torch.Tensor) -> float:
    """
    Calcola entropia dell'attenzione dei latents
    
    Args:
        attention_weights: [n_heads, N_lat, seq_len] o [N_lat, seq_len]
    
    Returns:
        Entropia media
    """
    if attention_weights.dim() == 3:
        # Multi-head: media su heads
        attention_weights = attention_weights.mean(dim=0)  # [N_lat, seq_len]
    
    # Normalizza per riga (ogni latent ha distribuzione su seq_len)
    probs = attention_weights / (attention_weights.sum(dim=-1, keepdim=True) + 1e-10)
    
    # Entropia: -sum(p * log(p))
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # [N_lat]
    
    return entropy.mean().item()


def compute_latent_variance(latents: torch.Tensor) -> float:
    """
    Calcola varianza dei latents (per evitare collasso)
    
    Args:
        latents: [N_lat, d_lat]
    
    Returns:
        Varianza media
    """
    # Varianza per dimensione
    variance = latents.var(dim=0)  # [d_lat]
    return variance.mean().item()


def compute_accuracy_vs_distance(predictions: List[str], references: List[str],
                                 metadata: List[dict]) -> Dict[int, float]:
    """
    Accuracy vs distanza del needle dal punto di inserimento
    
    Returns:
        Dict {distance_bucket: accuracy}
    """
    buckets = {}
    
    for pred, ref, meta in zip(predictions, references, metadata):
        qa_meta = meta.get("qa", {})
        if qa_meta.get("type") == "needle":
            distance = qa_meta.get("distance", -1)
            if distance >= 0:
                # Bucket: 0-1k, 1k-2k, 2k-4k, 4k-8k, 8k+
                bucket = min(distance // 1000, 8) * 1000
                if bucket not in buckets:
                    buckets[bucket] = {"correct": 0, "total": 0}
                
                buckets[bucket]["total"] += 1
                if pred.strip().lower() == ref.strip().lower():
                    buckets[bucket]["correct"] += 1
    
    # Calcola accuracy per bucket
    accuracy_by_distance = {}
    for bucket, counts in buckets.items():
        accuracy_by_distance[bucket] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
    
    return accuracy_by_distance


def log_batch_metrics(step: int, loss: float, latents: torch.Tensor,
                     attention_weights: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Logga metriche per batch (OBBLIGATORIE)
    
    Returns:
        Dict con tutte le metriche
    """
    metrics = {
        "step": step,
        "loss_ce": loss,
        "latent_variance": compute_latent_variance(latents)
    }
    
    if attention_weights is not None:
        metrics["attention_entropy"] = compute_attention_entropy(attention_weights)
    
    return metrics


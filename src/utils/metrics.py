"""
Metriche di valutazione
"""
import re
from typing import List, Dict
import numpy as np


def normalize_answer(text: str) -> str:
    """Normalizza risposta per confronto"""
    text = text.lower().strip()
    # Rimuovi punteggiatura extra
    text = re.sub(r'[^\w\s]', '', text)
    # Rimuovi spazi multipli
    text = re.sub(r'\s+', ' ', text)
    return text


def exact_match(pred: str, gold: str) -> bool:
    """Exact match normalizzato"""
    return normalize_answer(pred) == normalize_answer(gold)


def token_f1(pred: str, gold: str) -> float:
    """F1 a livello di token"""
    pred_tokens = set(normalize_answer(pred).split())
    gold_tokens = set(normalize_answer(gold).split())
    
    if len(gold_tokens) == 0:
        return 1.0 if len(pred_tokens) == 0 else 0.0
    
    if len(pred_tokens) == 0:
        return 0.0
    
    intersection = pred_tokens & gold_tokens
    precision = len(intersection) / len(pred_tokens) if pred_tokens else 0.0
    recall = len(intersection) / len(gold_tokens) if gold_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Calcola metriche aggregate"""
    em_scores = [exact_match(p, r) for p, r in zip(predictions, references)]
    f1_scores = [token_f1(p, r) for p, r in zip(predictions, references)]
    
    return {
        "exact_match": np.mean(em_scores),
        "token_f1": np.mean(f1_scores),
        "num_examples": len(predictions)
    }



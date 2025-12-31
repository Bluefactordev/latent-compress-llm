"""
Test di sanità OBBLIGATORI per esperimento controllo
Questi test devono essere NEGATIVI (distruggere le risposte)
"""
import torch
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


def shuffle_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Shuffle latents (test di sanità)
    
    Se l'accuracy non crolla, stai facendo prompt-tuning mascherato, non compressione.
    
    Args:
        latents: [N_lat, d_lat]
    
    Returns:
        Latents con ordine randomizzato
    """
    N_lat = latents.shape[0]
    indices = torch.randperm(N_lat, device=latents.device)
    return latents[indices]


def zero_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Zero latents (test di sanità)
    
    Se l'accuracy non crolla, i latents sono irrilevanti.
    
    Args:
        latents: [N_lat, d_lat]
    
    Returns:
        Latents azzerati
    """
    return torch.zeros_like(latents)


def only_last_chunk_latents(latents: torch.Tensor, n_latents_per_chunk: int) -> torch.Tensor:
    """
    Usa solo latents dell'ultimo chunk (test di sanità)
    
    Se l'accuracy non cambia, solo l'ultimo chunk conta (no compressione globale).
    
    Args:
        latents: [num_chunks * N_lat, d_lat]
        n_latents_per_chunk: N_lat per chunk
    
    Returns:
        Latents con solo ultimo chunk, resto zero
    """
    num_chunks = latents.shape[0] // n_latents_per_chunk
    if num_chunks == 0:
        return latents
    
    # Prendi solo ultimo chunk
    last_chunk_start = (num_chunks - 1) * n_latents_per_chunk
    last_chunk_end = num_chunks * n_latents_per_chunk
    
    result = torch.zeros_like(latents)
    result[last_chunk_start:last_chunk_end] = latents[last_chunk_start:last_chunk_end]
    
    return result


def random_latents(latents: torch.Tensor) -> torch.Tensor:
    """
    Latents completamente random (test di sanità)
    
    Se l'accuracy non crolla, il modello "indovina" senza informazione.
    
    Args:
        latents: [N_lat, d_lat]
    
    Returns:
        Latents random con stessa shape
    """
    return torch.randn_like(latents) * 0.02  # Stessa scala iniziale


def run_sanity_tests(compressor, reasoner, teacher, test_dataset, device, config,
                    n_latents_per_chunk: int) -> Dict[str, Dict[str, float]]:
    """
    Esegue tutti i test di sanità
    
    Returns:
        Dict {test_name: {metric: value}}
    """
    from src.utils.metrics import compute_metrics
    
    results = {}
    
    # Test normale (baseline)
    logger.info("Test baseline (normale)...")
    baseline_predictions = []
    baseline_references = []
    
    for example in test_dataset[:20]:  # Limita a 20 per velocità
        # Estrai latents normali
        teacher_states_list = []
        token_positions_list = []
        teacher_layers = config.get("teacher_layers", [-6, -18, -30])
        
        for chunk_text in example["context_chunks"]:
            if teacher.hidden_states_available:
                states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
                if states_dict is not None:
                    states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                    teacher_states_list.append(states_dict_device)
                    seq_len = list(states_dict.values())[0].shape[0]
                    token_positions_list.append(torch.arange(seq_len, device=device))
        
        with torch.no_grad():
            latents = compressor(teacher_states_list, token_positions_list=token_positions_list)
            pred = reasoner.generate_with_latents(
                latents,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        baseline_predictions.append(pred)
        baseline_references.append(example["answer"])
    
    baseline_metrics = compute_metrics(baseline_predictions, baseline_references)
    results["baseline"] = baseline_metrics
    
    # Test: Shuffle
    logger.info("Test SANITÀ: Shuffle latents...")
    shuffle_predictions = []
    shuffle_references = []
    
    for example in test_dataset[:20]:
        teacher_states_list = []
        token_positions_list = []
        teacher_layers = config.get("teacher_layers", [-6, -18, -30])
        
        for chunk_text in example["context_chunks"]:
            if teacher.hidden_states_available:
                states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
                if states_dict is not None:
                    states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                    teacher_states_list.append(states_dict_device)
                    seq_len = list(states_dict.values())[0].shape[0]
                    token_positions_list.append(torch.arange(seq_len, device=device))
        
        with torch.no_grad():
            latents = compressor(teacher_states_list, token_positions_list=token_positions_list)
            latents_shuffled = shuffle_latents(latents)
            pred = reasoner.generate_with_latents(
                latents_shuffled,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        shuffle_predictions.append(pred)
        shuffle_references.append(example["answer"])
    
    shuffle_metrics = compute_metrics(shuffle_predictions, shuffle_references)
    results["shuffle"] = shuffle_metrics
    
    # Test: Zero
    logger.info("Test SANITÀ: Zero latents...")
    zero_predictions = []
    zero_references = []
    
    for example in test_dataset[:20]:
        teacher_states_list = []
        token_positions_list = []
        teacher_layers = config.get("teacher_layers", [-6, -18, -30])
        
        for chunk_text in example["context_chunks"]:
            if teacher.hidden_states_available:
                states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
                if states_dict is not None:
                    states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                    teacher_states_list.append(states_dict_device)
                    seq_len = list(states_dict.values())[0].shape[0]
                    token_positions_list.append(torch.arange(seq_len, device=device))
        
        with torch.no_grad():
            latents = compressor(teacher_states_list, token_positions_list=token_positions_list)
            latents_zeroed = zero_latents(latents)
            pred = reasoner.generate_with_latents(
                latents_zeroed,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        zero_predictions.append(pred)
        zero_references.append(example["answer"])
    
    zero_metrics = compute_metrics(zero_predictions, zero_references)
    results["zero"] = zero_metrics
    
    # Test: Only last chunk
    logger.info("Test SANITÀ: Solo ultimo chunk...")
    last_chunk_predictions = []
    last_chunk_references = []
    
    for example in test_dataset[:20]:
        teacher_states_list = []
        token_positions_list = []
        teacher_layers = config.get("teacher_layers", [-6, -18, -30])
        
        for chunk_text in example["context_chunks"]:
            if teacher.hidden_states_available:
                states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
                if states_dict is not None:
                    states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                    teacher_states_list.append(states_dict_device)
                    seq_len = list(states_dict.values())[0].shape[0]
                    token_positions_list.append(torch.arange(seq_len, device=device))
        
        with torch.no_grad():
            latents = compressor(teacher_states_list, token_positions_list=token_positions_list)
            latents_last = only_last_chunk_latents(latents, n_latents_per_chunk)
            pred = reasoner.generate_with_latents(
                latents_last,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        last_chunk_predictions.append(pred)
        last_chunk_references.append(example["answer"])
    
    last_chunk_metrics = compute_metrics(last_chunk_predictions, last_chunk_references)
    results["only_last_chunk"] = last_chunk_metrics
    
    return results


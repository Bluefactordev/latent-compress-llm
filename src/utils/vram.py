"""
Utilities per monitoraggio VRAM
"""
import torch
from typing import Dict


def get_vram_usage() -> Dict[str, float]:
    """Ottiene uso VRAM corrente in GB"""
    if not torch.cuda.is_available():
        return {"allocated": 0.0, "reserved": 0.0, "max_allocated": 0.0}
    
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "max_allocated": torch.cuda.max_memory_allocated() / 1e9
    }


def reset_peak_memory():
    """Reset peak memory stats"""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()



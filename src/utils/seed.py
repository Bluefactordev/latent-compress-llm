"""
Utilities per seeding riproducibile
"""
import random
import numpy as np
import torch


def set_seed(seed=42):
    """
    Imposta seed per riproducibilità
    
    Args:
        seed: Valore seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



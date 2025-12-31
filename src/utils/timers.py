"""
Utilities per timing e profiling
"""
import time
from contextlib import contextmanager
from typing import Dict
import torch


class Timer:
    """Timer per misurare tempi di esecuzione"""
    
    def __init__(self):
        self.times: Dict[str, float] = {}
        self.starts: Dict[str, float] = {}
    
    @contextmanager
    def time(self, name: str):
        """Context manager per timing"""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)
    
    def start(self, name: str):
        """Avvia timer"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.starts[name] = time.time()
    
    def stop(self, name: str):
        """Ferma timer e registra"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name in self.starts:
            elapsed = time.time() - self.starts[name]
            if name not in self.times:
                self.times[name] = []
            self.times[name].append(elapsed)
            del self.starts[name]
    
    def get_mean(self, name: str) -> float:
        """Ottiene tempo medio per nome"""
        if name in self.times:
            return sum(self.times[name]) / len(self.times[name])
        return 0.0
    
    def get_total(self, name: str) -> float:
        """Ottiene tempo totale per nome"""
        if name in self.times:
            return sum(self.times[name])
        return 0.0
    
    def reset(self):
        """Reset tutti i timer"""
        self.times.clear()
        self.starts.clear()



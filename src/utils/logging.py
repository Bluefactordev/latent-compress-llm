"""
Sistema di logging standardizzato
"""
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logging(config=None, log_dir="logs"):
    """
    Configura il sistema di logging
    
    Args:
        config: Config object con LOG_LEVEL e LOG_DOMINI_SEPARATI
        log_dir: Directory per i log
    """
    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    
    # Nome file con timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir_path / f"diario_{timestamp}.log"
    
    # Configurazione logging
    log_level = getattr(config, 'LOG_LEVEL', logging.INFO) if config else logging.INFO
    
    logging.basicConfig(
        level=log_level,
        format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"Logging inizializzato. File: {log_file}")
    return logger


def get_logger(name):
    """Ottiene un logger con nome specifico"""
    return logging.getLogger(name)



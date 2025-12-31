"""
Utilities per tokenizzazione
"""
from transformers import AutoTokenizer
from typing import List, Tuple


def get_tokenizer(model_name: str):
    """Carica tokenizer per modello"""
    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)


def count_tokens(text: str, tokenizer) -> int:
    """Conta token in un testo"""
    return len(tokenizer.encode(text, add_special_tokens=False))


def chunk_text(text: str, tokenizer, chunk_size: int, overlap: int = 0) -> List[Tuple[str, int, int]]:
    """
    Divide testo in chunk con overlap
    
    Returns:
        List of (chunk_text, start_pos, end_pos) in tokens
    """
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append((chunk_text, start, end))
        
        if end >= len(tokens):
            break
        start = end - overlap
    
    return chunks



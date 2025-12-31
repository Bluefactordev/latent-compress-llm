"""
Baseline RAG: retrieval top-k chunks
"""
import torch
from typing import List
from rank_bm25 import BM25Okapi
import logging

logger = logging.getLogger(__name__)


class RAGBaseline:
    """RAG baseline con BM25 retrieval"""
    
    def __init__(self, config: dict):
        self.config = config
        self.rag_k = config["baselines"]["rag_k"]
        self.chunk_size = config["baselines"].get("rag_chunk_size", 2048)
    
    def retrieve_chunks(self, question: str, context_chunks: List[str], k: int) -> List[str]:
        """
        Retrieval top-k chunks usando BM25
        
        Args:
            question: Domanda
            context_chunks: Lista di chunk di contesto
            k: Numero di chunk da recuperare
        
        Returns:
            Lista di chunk recuperati
        """
        # Tokenizza per BM25
        tokenized_chunks = [chunk.lower().split() for chunk in context_chunks]
        tokenized_question = question.lower().split()
        
        # Crea BM25
        bm25 = BM25Okapi(tokenized_chunks)
        
        # Score
        scores = bm25.get_scores(tokenized_question)
        
        # Top-k
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        
        retrieved = [context_chunks[i] for i in top_indices]
        return retrieved
    
    def prepare_context(self, question: str, context_chunks: List[str], k: int) -> str:
        """Prepara contesto per reasoner"""
        retrieved = self.retrieve_chunks(question, context_chunks, k)
        context = "\n\n".join(retrieved)
        return context



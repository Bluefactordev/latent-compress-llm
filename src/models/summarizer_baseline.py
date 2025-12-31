"""
Baseline: summary per chunk usando teacher
"""
import json
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger(__name__)


class SummarizerBaseline:
    """Baseline che riassume ogni chunk con teacher"""
    
    def __init__(self, config: dict, teacher):
        self.config = config
        self.teacher = teacher
        self.summary_tokens = config["baselines"]["summary_tokens_per_chunk"]
        self.cache_dir = Path(config["paths"]["cache_dir"]) / "summaries"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def summarize_chunk(self, chunk_text: str, chunk_id: int) -> str:
        """Riassume un chunk"""
        # Controlla cache
        cache_file = self.cache_dir / f"summary_{hash(chunk_text)}.txt"
        if cache_file.exists():
            return cache_file.read_text().strip()
        
        # Prompt per summary
        prompt = f"Summarize the following text in at most {self.summary_tokens} tokens:\n\n{chunk_text}\n\nSummary:"
        
        # Usa teacher per generare summary
        summary = self.teacher.generate_answer_vllm(
            chunk_text,
            f"Summarize this text in at most {self.summary_tokens} tokens.",
            max_tokens=self.summary_tokens,
            cache_dir=self.cache_dir / "vllm_cache"
        )
        
        # Salva in cache
        cache_file.write_text(summary)
        
        return summary
    
    def prepare_context(self, context_chunks: List[str]) -> str:
        """Prepara contesto con summaries"""
        summaries = []
        for i, chunk in enumerate(context_chunks):
            summary = self.summarize_chunk(chunk, i)
            summaries.append(summary)
        
        return "\n\n".join(summaries)



"""
Teacher/Reader model: vLLM per generazione, Transformers per hidden states
"""
import json
import requests
from pathlib import Path
from typing import List, Optional, Tuple
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)


class TeacherReader:
    """Wrapper per teacher model: vLLM per generazione, Transformers per features"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_name = config["teacher_model"]
        self.vllm_url = config.get("teacher_vllm_url", "http://localhost:8000/v1")
        self.tokenizer = None
        self.transformers_model = None
        self.hidden_states_available = False
        
        # Carica tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        
        # Prova a caricare Transformers per hidden states
        self._load_transformers_model()
    
    def _load_transformers_model(self):
        """Tenta di caricare modello Transformers per hidden states"""
        try:
            logger.info(f"Tentativo di caricare {self.model_name} in Transformers per hidden states...")
            
            # Prova prima con AWQ se supportato
            try:
                from awq import AutoAWQForCausalLM
                logger.info("Tentativo con AutoAWQ...")
                self.transformers_model = AutoAWQForCausalLM.from_quantized(
                    self.model_name,
                    trust_remote_code=True,
                    device_map="auto"
                )
                self.hidden_states_available = True
                logger.info("✓ Modello AWQ caricato con successo")
                return
            except Exception as e:
                logger.warning(f"AutoAWQ fallito: {e}")
            
            # Fallback: prova con quantizzazione 8-bit
            try:
                logger.info("Tentativo con quantizzazione 8-bit...")
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    device_map="auto"
                )
                self.transformers_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
                self.hidden_states_available = True
                logger.info("✓ Modello 8-bit caricato con successo")
                return
            except Exception as e:
                logger.warning(f"Quantizzazione 8-bit fallita: {e}")
            
            # Fallback finale: prova modello più piccolo come proxy
            logger.warning("Tentativo con modello proxy più piccolo...")
            proxy_model_name = "Qwen/Qwen2.5-7B-Instruct"  # Modello più piccolo
            try:
                self.transformers_model = AutoModelForCausalLM.from_pretrained(
                    proxy_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                self.hidden_states_available = True
                logger.warning(f"⚠ Usando modello proxy {proxy_model_name} per hidden states")
                logger.warning("Le hidden states potrebbero non corrispondere esattamente al teacher")
                return
            except Exception as e:
                logger.error(f"Anche modello proxy fallito: {e}")
            
            logger.error("❌ Impossibile caricare modello per hidden states. Useremo solo logits.")
            self.hidden_states_available = False
            
        except Exception as e:
            logger.error(f"Errore nel caricamento Transformers: {e}")
            self.hidden_states_available = False
    
    def generate_answer_vllm(self, context: str, question: str, 
                           max_tokens: int = 64, temperature: float = 0.0,
                           cache_dir: Optional[Path] = None) -> str:
        """
        Genera risposta usando vLLM API
        
        Args:
            context: Contesto completo
            question: Domanda
            max_tokens: Max token da generare
            temperature: Temperature sampling
            cache_dir: Directory per cache (opzionale)
        
        Returns:
            Risposta generata
        """
        # Controlla cache
        if cache_dir:
            cache_file = cache_dir / f"teacher_answer_{hash(context + question)}.json"
            if cache_file.exists():
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                    logger.debug(f"Risposta da cache: {cache_file}")
                    return cached["answer"]
        
        # Prompt
        prompt = f"{context}\n\nQuestion: {question}\nAnswer:"
        
        # Chiama vLLM API
        payload = {
            "model": "default",
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stop": ["\n\n", "Question:"]
        }
        
        try:
            response = requests.post(
                f"{self.vllm_url}/completions",
                json=payload,
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
            answer = result["choices"][0]["text"].strip()
            
            # Salva in cache
            if cache_dir:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                with open(cache_file, 'w') as f:
                    json.dump({"answer": answer, "prompt": prompt}, f)
            
            return answer
        
        except Exception as e:
            logger.error(f"Errore nella chiamata vLLM: {e}")
            raise
    
    def get_hidden_states(self, text_chunk: str, layer_indices: List[int] = [-6]) -> Optional[torch.Tensor]:
        """
        Estrae hidden states da un chunk di testo (single-layer, backward compat)
        
        Args:
            text_chunk: Testo da processare
            layer_indices: Indici layer da estrarre (negativi = dall'ultimo)
        
        Returns:
            Tensor [seq_len, hidden_dim] o None se non disponibile
        """
        multi_layer = self.get_multi_layer_hidden_states(text_chunk, layer_indices)
        if multi_layer is None:
            return None
        # Per backward compat, ritorna solo primo layer
        if isinstance(multi_layer, dict):
            return list(multi_layer.values())[0]
        return multi_layer
    
    def get_multi_layer_hidden_states(self, text_chunk: str, layer_indices: List[int] = [-6, -18, -30]) -> Optional[dict]:
        """
        Estrae hidden states multi-layer da un chunk di testo (Fase 2)
        
        Args:
            text_chunk: Testo da processare
            layer_indices: Lista indici layer da estrarre (negativi = dall'ultimo)
        
        Returns:
            Dict {layer_idx: tensor [seq_len, hidden_dim]} o None se non disponibile
        """
        if not self.hidden_states_available or self.transformers_model is None:
            return None
        
        try:
            self.transformers_model.eval()
            
            # Tokenizza
            inputs = self.tokenizer(
                text_chunk,
                return_tensors="pt",
                truncation=True,
                max_length=self.config.get("teacher_ctx", 262000)
            )
            
            # Sposta su device
            device = next(self.transformers_model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Forward pass
            with torch.no_grad():
                outputs = self.transformers_model(**inputs, output_hidden_states=True)
            
            # Estrai hidden states dai layer specificati
            hidden_states = outputs.hidden_states
            num_layers = len(hidden_states)
            
            # Converti indici negativi e estrai
            result = {}
            for layer_idx in layer_indices:
                actual_idx = layer_idx if layer_idx >= 0 else num_layers + layer_idx
                if 0 <= actual_idx < num_layers:
                    result[layer_idx] = hidden_states[actual_idx][0]  # [seq_len, hidden_dim]
            
            return result if result else None
        
        except Exception as e:
            logger.error(f"Errore nell'estrazione multi-layer hidden states: {e}")
            return None
    
    def batch_generate_answers(self, examples: List[dict], cache_dir: Optional[Path] = None) -> List[str]:
        """Genera risposte in batch (sequenziale per ora)"""
        answers = []
        for example in examples:
            answer = self.generate_answer_vllm(
                example["context"],
                example["question"],
                cache_dir=cache_dir
            )
            answers.append(answer)
        return answers



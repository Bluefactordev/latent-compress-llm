"""
Wrapper per reasoner model con supporto inputs_embeds
"""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Optional
import logging

from src.models.latent_projection import LatentToReasonerProjection

logger = logging.getLogger(__name__)


class ReasonerWrapper:
    """Wrapper per reasoner che accetta latents come embeddings"""
    
    def __init__(self, config: dict, teacher_model_instance=None):
        self.config = config
        self.reasoner_mode = config.get("reasoner_mode", "transfer")
        self.max_ctx = config.get("reasoner_ctx", 64000)
        
        # Se teacher_control, usa stesso modello del teacher
        if self.reasoner_mode == "teacher_control":
            logger.info("Modalità teacher_control: reasoner = teacher (30B)")
            if teacher_model_instance is not None and hasattr(teacher_model_instance, 'transformers_model'):
                # Usa modello Transformers del teacher se disponibile
                if teacher_model_instance.transformers_model is not None:
                    self.model = teacher_model_instance.transformers_model
                    self.tokenizer = teacher_model_instance.tokenizer
                    logger.info("Usando modello Transformers del teacher come reasoner")
                else:
                    # Fallback: carica reasoner_model normale ma logga warning
                    logger.warning("Teacher Transformers model non disponibile, usando reasoner_model")
                    self.model_name = config["reasoner_model"]
                    self._load_model()
            else:
                # Carica reasoner_model ma è un fallback
                logger.warning("Teacher instance non fornita, usando reasoner_model")
                self.model_name = config["reasoner_model"]
                self._load_model()
        else:
            # Modalità transfer: usa reasoner_model normale
            self.model_name = config["reasoner_model"]
            self._load_model()
        
        self.model.eval()
        self.device = next(self.model.parameters()).device
        
        # Fase 4: Proiezione latents → reasoner space
        # Ottieni d_reasoner dall'embedding layer
        try:
            d_reasoner = self.model.get_input_embeddings().embedding_dim
        except:
            # Fallback: usa dimensione standard Qwen
            d_reasoner = 4096
        
        d_lat = config.get("d_lat", 512)
        self.latent_projection = LatentToReasonerProjection(d_lat, d_reasoner).to(self.device)
        logger.info(f"Proiezione latents: {d_lat} → {d_reasoner}")
    
    def _load_model(self):
        """Carica modello reasoner"""
        # Carica tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
        except Exception as e:
            logger.warning(f"Errore caricamento tokenizer {self.model_name}: {e}")
            # Fallback a Qwen2.5
            logger.info("Fallback a Qwen2.5 tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen2.5-7B-Instruct", trust_remote_code=True
            )
        
        # Carica modello con quantizzazione se necessario
        try:
            # Prova prima senza quantizzazione
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                trust_remote_code=True,
                torch_dtype=torch.float16,
                device_map="auto"
            )
            logger.info(f"Modello reasoner caricato senza quantizzazione")
        except Exception as e:
            logger.warning(f"Caricamento senza quantizzazione fallito: {e}")
            logger.info("Tentativo con quantizzazione 8-bit...")
            try:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    trust_remote_code=True,
                    device_map="auto"
                )
                logger.info("Modello reasoner caricato con quantizzazione 8-bit")
            except Exception as e2:
                logger.error(f"Anche quantizzazione fallita: {e2}")
                raise
    
    def forward_with_latents(self, latent_embeds: torch.Tensor, question_text: str,
                            target_answer_text: Optional[str] = None) -> dict:
        """
        Forward pass con latents come embeddings
        
        Args:
            latent_embeds: [N_lat, d_emb] embeddings latenti
            question_text: Testo domanda
            target_answer_text: Risposta target (opzionale, per loss)
        
        Returns:
            dict con 'loss' (se target fornito) e 'logits'
        """
        # Fase 4: Proietta latents → reasoner space
        reasoner_embeds = self.latent_projection(latent_embeds)  # [N_lat, d_reasoner]
        
        # Tokenizza domanda
        question_inputs = self.tokenizer(
            question_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        question_ids = question_inputs["input_ids"].to(self.device)  # [1, q_len]
        
        # Ottieni embeddings della domanda
        question_embeds = self.model.get_input_embeddings()(question_ids)  # [1, q_len, d_reasoner]
        
        # Prepara input: [latents] + [question]
        # Reasoner embeds: [N_lat, d_reasoner] -> [1, N_lat, d_reasoner]
        reasoner_embeds = reasoner_embeds.unsqueeze(0)  # [1, N_lat, d_reasoner]
        
        # Concatena
        inputs_embeds = torch.cat([reasoner_embeds, question_embeds], dim=1)  # [1, N_lat + q_len, d_reasoner]
        
        # Attention mask
        n_lat = latent_embeds.shape[1]
        q_len = question_embeds.shape[1]
        attention_mask = torch.ones(1, n_lat + q_len, device=self.device, dtype=torch.long)
        
        # Se abbiamo target, tokenizza anche quello
        if target_answer_text is not None:
            # Tokenizza risposta target
            answer_inputs = self.tokenizer(
                target_answer_text,
                return_tensors="pt",
                add_special_tokens=False
            )
            answer_ids = answer_inputs["input_ids"].to(self.device)  # [1, a_len]
            
            # Concatena input_ids per calcolare loss
            # Per loss, dobbiamo allineare: input = [latents + question], labels = [question + answer]
            # Ma in realtà vogliamo loss solo sulla parte answer
            # Usiamo un approccio più semplice: prependiamo un token speciale per i latents
            
            # Forward con inputs_embeds
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=None  # Calcoliamo loss manualmente
            )
            
            # Estrai logits
            logits = outputs.logits  # [1, seq_len, vocab_size]
            
            # Per loss: vogliamo predire answer tokens
            # Logits shape: [1, n_lat + q_len, vocab_size]
            # Prendi logits dalla fine della question (ultimo token di question predice primo di answer)
            # Shift: logits[i] predice token[i+1]
            seq_len = logits.shape[1]
            answer_len = answer_ids.shape[1]
            
            # Prendi logits per predire answer (dall'ultimo token di question in poi)
            if n_lat + q_len - 1 + answer_len <= seq_len:
                answer_logits = logits[0, n_lat + q_len - 1: n_lat + q_len - 1 + answer_len, :]
            else:
                # Se non ci sono abbastanza logits, prendi quelli disponibili
                available = seq_len - (n_lat + q_len - 1)
                answer_logits = logits[0, n_lat + q_len - 1:, :]
                answer_ids = answer_ids[:, :available]
            
            # Calcola loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                answer_logits.view(-1, answer_logits.shape[-1]),
                answer_ids.view(-1)
            )
            
            return {"loss": loss, "logits": logits}
        else:
            # Solo forward, no loss
            outputs = self.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask
            )
            return {"logits": outputs.logits}
    
    def generate_with_latents(self, latent_embeds: torch.Tensor, question_text: str,
                              max_new_tokens: int = 64, temperature: float = 0.0) -> str:
        """
        Genera risposta usando latents
        
        Args:
            latent_embeds: [N_lat, d_emb]
            question_text: Domanda
            max_new_tokens: Max token da generare
            temperature: Temperature sampling
        
        Returns:
            Risposta generata
        """
        # Fase 4: Proietta latents → reasoner space
        reasoner_embeds = self.latent_projection(latent_embeds)  # [N_lat, d_reasoner]
        
        # Tokenizza domanda
        question_inputs = self.tokenizer(
            question_text,
            return_tensors="pt",
            add_special_tokens=False
        )
        question_ids = question_inputs["input_ids"].to(self.device)
        question_embeds = self.model.get_input_embeddings()(question_ids)  # [1, q_len, d_reasoner]
        
        # Prepara inputs_embeds
        reasoner_embeds = reasoner_embeds.unsqueeze(0)  # [1, N_lat, d_reasoner]
        inputs_embeds = torch.cat([reasoner_embeds, question_embeds], dim=1)  # [1, N_lat + q_len, d_reasoner]
        
        n_lat = latent_embeds.shape[1]
        q_len = question_embeds.shape[1]
        attention_mask = torch.ones(1, n_lat + q_len, device=self.device, dtype=torch.long)
        
        # Genera
        with torch.no_grad():
            # Usa generate con inputs_embeds
            # Nota: transformers supporta inputs_embeds in generate
            try:
                generated_ids = self.model.generate(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                )
            except Exception as e:
                logger.warning(f"Generate con inputs_embeds fallito: {e}, provo approccio alternativo")
                # Fallback: usa solo question senza latents
                generated_ids = self.model.generate(
                    question_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    pad_token_id=self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else self.tokenizer.pad_token_id
                )
                # Decodifica solo la parte generata
                answer = self.tokenizer.decode(generated_ids[0][question_ids.shape[1]:], skip_special_tokens=True)
                return answer.strip()
        
        # Decodifica solo la parte generata (dopo latents + question)
        generated_tokens = generated_ids[0, n_lat + q_len:]
        answer = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return answer.strip()


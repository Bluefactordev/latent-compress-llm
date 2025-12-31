"""
Training loop per compressore con task distillation
"""
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.tokenization import get_tokenizer
from src.utils.timers import Timer
from src.utils.vram import get_vram_usage, reset_peak_memory
from src.data.dataset import LatentCompressionDataset
from src.models.teacher_reader import TeacherReader
from src.models.compressor import LatentCompressor
from src.models.reasoner_wrapper import ReasonerWrapper
from src.models.latent_projection import LatentToReasonerProjection
from src.train.losses import CompressionLoss


def load_config(config_path: str, exp_name: str = None) -> dict:
    """Carica configurazione base e eventuale override esperimento"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Se exp_name specificato, carica override da exp_grid.yaml
    if exp_name:
        exp_grid_path = Path(config_path).parent / "exp_grid.yaml"
        if exp_grid_path.exists():
            with open(exp_grid_path, 'r') as f:
                exp_grid = yaml.safe_load(f)
            if exp_name in exp_grid.get("experiments", {}):
                exp_config = exp_grid["experiments"][exp_name]
                # Merge: exp_config override su config base
                config.update(exp_config)
                # Merge nested dicts
                if "train" in exp_config:
                    config["train"].update(exp_config["train"])
    
    return config


def cache_teacher_answers(dataset, teacher, cache_dir: Path):
    """Pre-calcola risposte teacher per training"""
    cache_file = cache_dir / "teacher_answers.jsonl"
    
    if cache_file.exists():
        logger.info(f"Caricamento risposte teacher da cache: {cache_file}")
        answers = []
        with open(cache_file, 'r') as f:
            for line in f:
                answers.append(json.loads(line)["answer"])
        return answers
    
    logger.info("Generazione risposte teacher (questo può richiedere tempo)...")
    answers = []
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    with open(cache_file, 'w') as f:
        for i, example in enumerate(tqdm(dataset.examples)):
            answer = teacher.generate_answer_vllm(
                example["context"],
                example["question"],
                cache_dir=cache_dir / "vllm_cache"
            )
            answers.append(answer)
            f.write(json.dumps({"answer": answer}) + '\n')
    
    logger.info(f"Salvate {len(answers)} risposte teacher")
    return answers


def train_step(compressor, reasoner, teacher, batch, teacher_answers, 
               device, config, optimizer, loss_fn):
    """Un passo di training"""
    compressor.train()
    auxiliary_type = config.get("train", {}).get("auxiliary_loss_type", "alignment")
    
    # Estrai hidden states teacher per ogni chunk (Fase 2: multi-layer)
    teacher_states_list = []
    token_positions_list = []
    teacher_layers = config.get("teacher_layers", [-6, -18, -30])
    
    # Gestisci batch: context_chunks può essere lista di liste o lista singola
    context_chunks = batch["context_chunks"]
    if isinstance(context_chunks[0], list):
        # Batch size > 1: prendi primo esempio
        context_chunks = context_chunks[0]
    
    for chunk_text in context_chunks:
        if teacher.hidden_states_available:
            # Fase 2: Multi-layer extraction
            states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
            if states_dict is not None:
                # Sposta su device
                states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                teacher_states_list.append(states_dict_device)
                
                # Token positions per time-bucket (Fase 3)
                seq_len = list(states_dict.values())[0].shape[0]
                token_positions = torch.arange(seq_len, device=device)
                token_positions_list.append(token_positions)
            else:
                # Fallback: dummy multi-layer
                d_teacher = config.get("d_teacher", 4096)
                tokens = teacher.tokenizer.encode(chunk_text, return_tensors="pt")
                seq_len = len(tokens[0])
                dummy_dict = {layer: torch.randn(seq_len, d_teacher, device=device) * 0.01 
                             for layer in teacher_layers}
                teacher_states_list.append(dummy_dict)
                token_positions_list.append(torch.arange(seq_len, device=device))
        else:
            # Fallback: dummy multi-layer
            d_teacher = config.get("d_teacher", 4096)
            tokens = teacher.tokenizer.encode(chunk_text, return_tensors="pt")
            seq_len = len(tokens[0])
            dummy_dict = {layer: torch.randn(seq_len, d_teacher, device=device) * 0.01 
                         for layer in teacher_layers}
            teacher_states_list.append(dummy_dict)
            token_positions_list.append(torch.arange(seq_len, device=device))
    
    # Compressore produce latents (con token positions per time-bucket)
    latents = compressor(teacher_states_list, token_positions_list=token_positions_list)  # [total_latents, d_lat]
    
    # Reasoner forward con latents
    question = batch["question"][0] if isinstance(batch["question"], list) else batch["question"]
    # Ottieni indice esempio dal batch
    if isinstance(batch.get("_idx"), list):
        answer_idx = batch["_idx"][0]
    elif isinstance(batch.get("_idx"), torch.Tensor):
        answer_idx = batch["_idx"].item()
    else:
        answer_idx = 0
    
    # Usa teacher answer se disponibile, altrimenti ground truth
    if answer_idx < len(teacher_answers):
        target_answer = teacher_answers[answer_idx]
    else:
        target_answer = batch["answer"][0] if isinstance(batch["answer"], list) else batch["answer"]
    
    result = reasoner.forward_with_latents(
        latents,
        question,
        target_answer
    )
    
    primary_loss = result["loss"]
    
    # Fase 5: Auxiliary loss
    # Prepara dati per auxiliary loss
    auxiliary_data = {}
    if auxiliary_type == "alignment":
        # Calcola mean dei teacher states
        if teacher_states_list:
            # Prendi mean del primo chunk come esempio
            first_chunk_states = teacher_states_list[0]
            if isinstance(first_chunk_states, dict):
                # Multi-layer: prendi mean del primo layer
                first_layer = list(first_chunk_states.values())[0]
                teacher_mean = first_layer.mean(dim=0)  # [d_teacher]
            else:
                teacher_mean = first_chunk_states.mean(dim=0)
            auxiliary_data["teacher_states_mean"] = teacher_mean
    
    # Loss totale
    total_loss, aux_loss_value = loss_fn(primary_loss, latents, auxiliary_data)
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    
    return {
        "total_loss": total_loss.item(),
        "primary_loss": primary_loss.item(),
        "auxiliary_loss": aux_loss_value
    }


def validate(compressor, reasoner, teacher, val_dataset, device, config):
    """Valutazione su validation set"""
    compressor.eval()
    
    losses = []
    predictions = []
    references = []
    
    with torch.no_grad():
        for i in range(min(50, len(val_dataset))):  # Limita a 50 per velocità
            example = val_dataset[i]
            
            # Estrai hidden states
            teacher_states_list = []
            for chunk_text in example["context_chunks"]:
                if teacher.hidden_states_available:
                    states = teacher.get_hidden_states(chunk_text, config.get("layer_indices", [-6]))
                    if states is not None:
                        teacher_states_list.append(states.to(device))
                    else:
                        d_teacher = config.get("d_teacher", 4096)
                        dummy = torch.randn(100, d_teacher, device=device) * 0.01
                        teacher_states_list.append(dummy)
                else:
                    d_teacher = config.get("d_teacher", 4096)
                    dummy = torch.randn(100, d_teacher, device=device) * 0.01
                    teacher_states_list.append(dummy)
            
            # Compressore
            latents = compressor(teacher_states_list)
            
            # Genera risposta
            question = example["question"]
            pred_answer = reasoner.generate_with_latents(
                latents,
                question,
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
            
            predictions.append(pred_answer)
            references.append(example["answer"])
    
    # Calcola metriche
    from src.utils.metrics import compute_metrics
    metrics = compute_metrics(predictions, references)
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--exp", type=str, default="S0")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config, args.exp)
    set_seed(42)
    logger = setup_logging(config)
    logger.info(f"Esperimento: {args.exp}")
    logger.info(f"Config: {config}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Paths
    exp_dir = Path(config["train"]["checkpoint_dir"]) / args.exp
    exp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Carica modelli
    logger.info("Caricamento modelli...")
    teacher = TeacherReader(config)
    reasoner = ReasonerWrapper(config, teacher_model_instance=teacher)  # Passa teacher per teacher_control
    compressor = LatentCompressor(config).to(device)
    
    # Fase 4: Proiezione latents (già in reasoner, ma anche come modulo separato se necessario)
    logger.info(f"Reasoner mode: {config.get('reasoner_mode', 'transfer')}")
    
    logger.info(f"Hidden states disponibili: {teacher.hidden_states_available}")
    
    # Carica dataset
    data_dir = Path(config["paths"]["data_dir"])
    train_file = data_dir / "raw" / "synthetic_train.jsonl"
    val_file = data_dir / "raw" / "synthetic_val.jsonl"
    
    teacher_tokenizer = get_tokenizer(config["teacher_model"])
    train_dataset = LatentCompressionDataset(
        str(train_file),
        teacher_tokenizer,
        config["chunk_size_tokens"],
        config["chunk_overlap_tokens"]
    )
    val_dataset = LatentCompressionDataset(
        str(val_file),
        teacher_tokenizer,
        config["chunk_size_tokens"],
        config["chunk_overlap_tokens"],
        max_examples=200
    )
    
    # Cache teacher answers
    teacher_answers = cache_teacher_answers(train_dataset, teacher, cache_dir)
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["train"]["batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    # Optimizer: solo compressor + latent_projection (Fase 5: freeze policy)
    # Teacher e Reasoner sono frozen
    trainable_params = list(compressor.parameters()) + list(reasoner.latent_projection.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["train"]["lr"]
    )
    logger.info(f"Parametri trainabili: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    # Fase 5: Loss combinata
    auxiliary_type = config.get("train", {}).get("auxiliary_loss_type", "alignment")
    loss_fn = CompressionLoss(config, auxiliary_type=auxiliary_type)
    logger.info(f"Auxiliary loss type: {auxiliary_type}")
    
    # Training loop
    train_config = config["train"]
    max_steps = train_config["max_steps"]
    eval_steps = train_config.get("eval_steps", 200)
    save_steps = train_config.get("save_steps", 500)
    grad_accum = train_config.get("gradient_accumulation_steps", 8)
    
    logger.info(f"Inizio training: {max_steps} steps")
    timer = Timer()
    reset_peak_memory()
    
    step = 0
    best_val_f1 = 0.0
    
    for epoch in range(100):  # Max epochs
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Training step
            with timer.time("train_step"):
                loss = train_step(
                    compressor, reasoner, teacher, batch, teacher_answers,
                    device, config, optimizer, loss_fn
                )
            
            step += 1
            
            if step % 10 == 0:
                if isinstance(loss, dict):
                    logger.info(f"Step {step}/{max_steps}, Total Loss: {loss['total_loss']:.4f}, "
                              f"Primary: {loss['primary_loss']:.4f}, Aux: {loss['auxiliary_loss']:.4f}")
                else:
                    logger.info(f"Step {step}/{max_steps}, Loss: {loss:.4f}")
                vram = get_vram_usage()
                logger.info(f"VRAM: {vram['allocated']:.2f}GB / {vram['max_allocated']:.2f}GB")
            
            # Validation
            if step % eval_steps == 0:
                logger.info("Validazione...")
                val_metrics = validate(compressor, reasoner, teacher, val_dataset, device, config)
                logger.info(f"Val metrics: {val_metrics}")
                
                if val_metrics["token_f1"] > best_val_f1:
                    best_val_f1 = val_metrics["token_f1"]
                    # Salva best model
                    checkpoint = {
                        "step": step,
                        "compressor_state": compressor.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_metrics": val_metrics
                    }
                    torch.save(checkpoint, exp_dir / "best_model.pt")
                    logger.info(f"Salvato best model (F1: {best_val_f1:.4f})")
            
            # Save checkpoint
            if step % save_steps == 0:
                checkpoint = {
                    "step": step,
                    "compressor_state": compressor.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                }
                torch.save(checkpoint, exp_dir / f"checkpoint_step_{step}.pt")
                logger.info(f"Checkpoint salvato: step {step}")
        
        if step >= max_steps:
            break
    
    logger.info("Training completato!")
    logger.info(f"Best val F1: {best_val_f1:.4f}")
    
    # Salva modello finale
    final_checkpoint = {
        "step": step,
        "compressor_state": compressor.state_dict(),
        "optimizer_state": optimizer.state_dict()
    }
    torch.save(final_checkpoint, exp_dir / "final_model.pt")


if __name__ == "__main__":
    main()


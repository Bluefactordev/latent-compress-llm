"""
Training loop per esperimento controllo (Fase 1)
Con logging metriche OBBLIGATORIE
"""
import argparse
import yaml
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import csv

from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.tokenization import get_tokenizer
from src.utils.timers import Timer
from src.utils.vram import get_vram_usage, reset_peak_memory
from src.utils.metrics import compute_metrics
from src.data.dataset import LatentCompressionDataset
from src.models.teacher_reader import TeacherReader
from src.models.compressor import LatentCompressor
from src.models.reasoner_wrapper import ReasonerWrapper
from src.train.losses import CompressionLoss
from src.eval.control_metrics import (
    compute_needle_accuracy, compute_versioning_accuracy,
    compute_accuracy_vs_distance, log_batch_metrics
)
from src.eval.sanity_tests import run_sanity_tests


def load_config(config_path: str) -> dict:
    """Carica configurazione"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def cache_teacher_answers(dataset, teacher, cache_dir: Path):
    """Pre-calcola risposte teacher per training"""
    cache_file = cache_dir / "teacher_answers.jsonl"
    
    if cache_file.exists():
        logger = get_logger(__name__)
        logger.info(f"Caricamento risposte teacher da cache: {cache_file}")
        answers = []
        with open(cache_file, 'r') as f:
            for line in f:
                answers.append(json.loads(line)["answer"])
        return answers
    
    logger = get_logger(__name__)
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
    """Un passo di training con logging metriche"""
    compressor.train()
    auxiliary_type = config.get("train", {}).get("auxiliary_loss_type", "alignment")
    
    # Estrai hidden states teacher per ogni chunk (multi-layer)
    teacher_states_list = []
    token_positions_list = []
    teacher_layers = config.get("teacher_layers", [-6, -18, -30])
    
    context_chunks = batch["context_chunks"]
    if isinstance(context_chunks[0], list):
        context_chunks = context_chunks[0]
    
    for chunk_text in context_chunks:
        if teacher.hidden_states_available:
            states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
            if states_dict is not None:
                states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                teacher_states_list.append(states_dict_device)
                seq_len = list(states_dict.values())[0].shape[0]
                token_positions = torch.arange(seq_len, device=device)
                token_positions_list.append(token_positions)
            else:
                d_teacher = config.get("d_teacher", 4096)
                tokens = teacher.tokenizer.encode(chunk_text, return_tensors="pt")
                seq_len = len(tokens[0])
                dummy_dict = {layer: torch.randn(seq_len, d_teacher, device=device) * 0.01 
                             for layer in teacher_layers}
                teacher_states_list.append(dummy_dict)
                token_positions_list.append(torch.arange(seq_len, device=device))
        else:
            d_teacher = config.get("d_teacher", 4096)
            tokens = teacher.tokenizer.encode(chunk_text, return_tensors="pt")
            seq_len = len(tokens[0])
            dummy_dict = {layer: torch.randn(seq_len, d_teacher, device=device) * 0.01 
                         for layer in teacher_layers}
            teacher_states_list.append(dummy_dict)
            token_positions_list.append(torch.arange(seq_len, device=device))
    
    # Compressore produce latents
    latents = compressor(teacher_states_list, token_positions_list=token_positions_list)
    
    # Reasoner forward con latents
    question = batch["question"][0] if isinstance(batch["question"], list) else batch["question"]
    if isinstance(batch.get("_idx"), list):
        answer_idx = batch["_idx"][0]
    elif isinstance(batch.get("_idx"), torch.Tensor):
        answer_idx = batch["_idx"].item()
    else:
        answer_idx = 0
    
    target_answer = teacher_answers[answer_idx] if answer_idx < len(teacher_answers) else batch["answer"]
    
    result = reasoner.forward_with_latents(
        latents,
        question,
        target_answer
    )
    
    primary_loss = result["loss"]
    
    # Auxiliary loss
    auxiliary_data = {}
    if auxiliary_type == "alignment":
        if teacher_states_list:
            first_chunk_states = teacher_states_list[0]
            if isinstance(first_chunk_states, dict):
                first_layer = list(first_chunk_states.values())[0]
                teacher_mean = first_layer.mean(dim=0)
            else:
                teacher_mean = first_chunk_states.mean(dim=0)
            auxiliary_data["teacher_states_mean"] = teacher_mean
    
    total_loss, aux_loss_value = loss_fn(primary_loss, latents, auxiliary_data)
    
    # Backward
    trainable_params = list(compressor.parameters()) + list(reasoner.latent_projection.parameters())
    optimizer.zero_grad()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
    optimizer.step()
    
    return {
        "total_loss": total_loss.item(),
        "primary_loss": primary_loss.item(),
        "auxiliary_loss": aux_loss_value,
        "latents": latents.detach()
    }


def evaluate_control(compressor, reasoner, teacher, dataset, device, config):
    """Valutazione con metriche OBBLIGATORIE"""
    compressor.eval()
    
    predictions = []
    references = []
    metadata_list = []
    teacher_full_predictions = []
    
    logger = get_logger(__name__)
    logger.info("Valutazione controllo...")
    
    with torch.no_grad():
        for example in tqdm(dataset):
            # Teacher full context (baseline)
            teacher_answer = teacher.generate_answer_vllm(
                example["context"],
                example["question"],
                max_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
            teacher_full_predictions.append(teacher_answer)
            
            # Student (latents)
            teacher_states_list = []
            token_positions_list = []
            teacher_layers = config.get("teacher_layers", [-6, -18, -30])
            
            for chunk_text in example["context_chunks"]:
                if teacher.hidden_states_available:
                    states_dict = teacher.get_multi_layer_hidden_states(chunk_text, teacher_layers)
                    if states_dict is not None:
                        states_dict_device = {k: v.to(device) for k, v in states_dict.items()}
                        teacher_states_list.append(states_dict_device)
                        seq_len = list(states_dict.values())[0].shape[0]
                        token_positions_list.append(torch.arange(seq_len, device=device))
            
            latents = compressor(teacher_states_list, token_positions_list=token_positions_list)
            
            pred = reasoner.generate_with_latents(
                latents,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
            
            predictions.append(pred)
            references.append(example["answer"])
            metadata_list.append(example.get("metadata", {}))
    
    # Metriche OBBLIGATORIE
    metrics = compute_metrics(predictions, references)
    metrics["needle_accuracy"] = compute_needle_accuracy(predictions, references, metadata_list)
    metrics["versioning_accuracy"] = compute_versioning_accuracy(predictions, references, metadata_list)
    metrics["accuracy_vs_distance"] = compute_accuracy_vs_distance(predictions, references, metadata_list)
    
    # Confronto 30B(full) vs 30B(latents)
    teacher_metrics = compute_metrics(teacher_full_predictions, references)
    metrics["teacher_full_accuracy"] = teacher_metrics["exact_match"]
    metrics["latents_accuracy"] = metrics["exact_match"]
    metrics["relative_accuracy"] = metrics["latents_accuracy"] / teacher_metrics["exact_match"] if teacher_metrics["exact_match"] > 0 else 0.0
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/control_experiment.yaml")
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    set_seed(42)
    logger = setup_logging(config)
    logger.info("=" * 80)
    logger.info("ESPERIMENTO CONTROLLO (Fase 1)")
    logger.info("=" * 80)
    logger.info(f"Config: {args.config}")
    logger.info(f"Reasoner mode: {config.get('reasoner_mode')}")
    logger.info(f"N_latents: {config['n_latents']}")
    logger.info(f"Chunk size: {config['chunk_size_tokens']}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Paths
    exp_dir = Path(config["train"]["checkpoint_dir"])
    exp_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(config["paths"]["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carica modelli
    logger.info("Caricamento modelli...")
    teacher = TeacherReader(config)
    reasoner = ReasonerWrapper(config, teacher_model_instance=teacher)
    compressor = LatentCompressor(config).to(device)
    
    logger.info(f"Hidden states disponibili: {teacher.hidden_states_available}")
    logger.info(f"Reasoner mode: {config.get('reasoner_mode')}")
    
    # Carica dataset
    data_dir = Path(config["paths"]["data_dir"])
    train_file = data_dir / "raw" / "synthetic_train.jsonl"
    val_file = data_dir / "raw" / "synthetic_val.jsonl"
    test_file = data_dir / "raw" / "synthetic_test.jsonl"
    
    teacher_tokenizer = get_tokenizer("Qwen/Qwen2.5-7B-Instruct")  # Compatible tokenizer
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
        max_examples=50
    )
    test_dataset = LatentCompressionDataset(
        str(test_file),
        teacher_tokenizer,
        config["chunk_size_tokens"],
        config["chunk_overlap_tokens"],
        max_examples=100
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
    
    # Optimizer
    trainable_params = list(compressor.parameters()) + list(reasoner.latent_projection.parameters())
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=config["train"]["lr"]
    )
    logger.info(f"Parametri trainabili: {sum(p.numel() for p in trainable_params) / 1e6:.2f}M")
    
    # Loss
    auxiliary_type = config.get("train", {}).get("auxiliary_loss_type", "alignment")
    loss_fn = CompressionLoss(config, auxiliary_type=auxiliary_type)
    
    # Training loop
    train_config = config["train"]
    max_steps = train_config["max_steps"]
    eval_steps = train_config.get("eval_steps", 100)
    save_steps = train_config.get("save_steps", 200)
    
    logger.info(f"Inizio training: {max_steps} steps")
    timer = Timer()
    reset_peak_memory()
    
    # CSV per logging metriche batch
    metrics_file = output_dir / "batch_metrics.csv"
    with open(metrics_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["step", "loss_ce", "latent_variance", "attention_entropy"])
    
    step = 0
    best_val_accuracy = 0.0
    
    for epoch in range(100):
        for batch in train_loader:
            if step >= max_steps:
                break
            
            # Training step
            with timer.time("train_step"):
                result = train_step(
                    compressor, reasoner, teacher, batch, teacher_answers,
                    device, config, optimizer, loss_fn
                )
            
            step += 1
            
            # Log metriche batch OBBLIGATORIE
            if step % 10 == 0:
                batch_metrics = log_batch_metrics(
                    step,
                    result["primary_loss"],
                    result["latents"]
                )
                
                with open(metrics_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        batch_metrics["step"],
                        batch_metrics["loss_ce"],
                        batch_metrics["latent_variance"],
                        batch_metrics.get("attention_entropy", 0.0)
                    ])
                
                logger.info(f"Step {step}/{max_steps}, Total Loss: {result['total_loss']:.4f}, "
                          f"Primary: {result['primary_loss']:.4f}, Aux: {result['auxiliary_loss']:.4f}, "
                          f"Variance: {batch_metrics['latent_variance']:.6f}")
            
            # Validation
            if step % eval_steps == 0:
                logger.info("Validazione...")
                val_metrics = evaluate_control(compressor, reasoner, teacher, val_dataset, device, config)
                logger.info(f"Val metrics: {val_metrics}")
                
                val_accuracy = val_metrics.get("exact_match", 0.0)
                if val_accuracy > best_val_accuracy:
                    best_val_accuracy = val_accuracy
                    checkpoint = {
                        "step": step,
                        "compressor_state": compressor.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "val_metrics": val_metrics
                    }
                    torch.save(checkpoint, exp_dir / "best_model.pt")
                    logger.info(f"Salvato best model (Accuracy: {best_val_accuracy:.4f})")
            
            # Save checkpoint
            if step % save_steps == 0:
                checkpoint = {
                    "step": step,
                    "compressor_state": compressor.state_dict(),
                    "optimizer_state": optimizer.state_dict()
                }
                torch.save(checkpoint, exp_dir / f"checkpoint_step_{step}.pt")
        
        if step >= max_steps:
            break
    
    logger.info("Training completato!")
    
    # Valutazione finale su test set
    logger.info("=" * 80)
    logger.info("VALUTAZIONE FINALE SU TEST SET")
    logger.info("=" * 80)
    test_metrics = evaluate_control(compressor, reasoner, teacher, test_dataset, device, config)
    
    # Salva risultati
    results_file = output_dir / "control_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            "config": config,
            "test_metrics": test_metrics,
            "best_val_accuracy": best_val_accuracy,
            "vram": get_vram_usage()
        }, f, indent=2)
    
    logger.info(f"Risultati salvati in {results_file}")
    logger.info(f"Test metrics: {test_metrics}")
    
    # Test di sanità OBBLIGATORI
    logger.info("=" * 80)
    logger.info("TEST DI SANITÀ (devono essere NEGATIVI)")
    logger.info("=" * 80)
    sanity_results = run_sanity_tests(
        compressor, reasoner, teacher, test_dataset, device, config,
        n_latents_per_chunk=config["n_latents"]
    )
    
    sanity_file = output_dir / "sanity_tests.json"
    with open(sanity_file, 'w') as f:
        json.dump(sanity_results, f, indent=2)
    
    logger.info(f"Test di sanità salvati in {sanity_file}")
    logger.info(f"Sanity results: {sanity_results}")
    
    # Criteri PASS/FAIL
    logger.info("=" * 80)
    logger.info("CRITERI PASS/FAIL")
    logger.info("=" * 80)
    
    relative_acc = test_metrics.get("relative_accuracy", 0.0)
    needle_acc = test_metrics.get("needle_accuracy", 0.0)
    shuffle_acc = sanity_results.get("shuffle", {}).get("exact_match", 1.0)
    zero_acc = sanity_results.get("zero", {}).get("exact_match", 1.0)
    
    passed = (
        relative_acc >= 0.85 and
        needle_acc >= 0.70 and
        shuffle_acc < 0.5 and  # Deve crollare
        zero_acc < 0.5  # Deve crollare
    )
    
    if passed:
        logger.info("✅ PASS: Esperimento controllo superato!")
        logger.info(f"  - Relative accuracy: {relative_acc:.2%} (>= 85%)")
        logger.info(f"  - Needle accuracy: {needle_acc:.2%} (>= 70%)")
        logger.info(f"  - Shuffle distrugge: {shuffle_acc:.2%} (< 50%)")
        logger.info(f"  - Zero distrugge: {zero_acc:.2%} (< 50%)")
    else:
        logger.warning("❌ FAIL: Esperimento controllo fallito")
        logger.warning(f"  - Relative accuracy: {relative_acc:.2%} (richiesto >= 85%)")
        logger.warning(f"  - Needle accuracy: {needle_acc:.2%} (richiesto >= 70%)")
        logger.warning(f"  - Shuffle accuracy: {shuffle_acc:.2%} (richiesto < 50%)")
        logger.warning(f"  - Zero accuracy: {zero_acc:.2%} (richiesto < 50%)")
    
    # Salva verdict
    verdict_file = output_dir / "verdict.txt"
    with open(verdict_file, 'w') as f:
        f.write(f"PASS: {passed}\n")
        f.write(f"Relative accuracy: {relative_acc:.4f}\n")
        f.write(f"Needle accuracy: {needle_acc:.4f}\n")
        f.write(f"Shuffle accuracy: {shuffle_acc:.4f}\n")
        f.write(f"Zero accuracy: {zero_acc:.4f}\n")
    
    logger.info(f"Verdict salvato in {verdict_file}")


if __name__ == "__main__":
    main()


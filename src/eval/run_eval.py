"""
Sistema di valutazione completo
"""
import argparse
import yaml
import json
from pathlib import Path
import torch
from tqdm import tqdm

from src.utils.seed import set_seed
from src.utils.logging import setup_logging, get_logger
from src.utils.tokenization import get_tokenizer
from src.utils.timers import Timer
from src.utils.vram import get_vram_usage, reset_peak_memory
from src.utils.metrics import compute_metrics, exact_match, token_f1
from src.data.dataset import LatentCompressionDataset
from src.models.teacher_reader import TeacherReader
from src.models.compressor import LatentCompressor
from src.models.reasoner_wrapper import ReasonerWrapper
from src.models.rag_baseline import RAGBaseline
from src.models.summarizer_baseline import SummarizerBaseline
from src.models.pooling_baseline import PoolingBaseline


def load_config(config_path: str, exp_name: str = None) -> dict:
    """Carica configurazione"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if exp_name:
        exp_grid_path = Path(config_path).parent / "exp_grid.yaml"
        if exp_grid_path.exists():
            with open(exp_grid_path, 'r') as f:
                exp_grid = yaml.safe_load(f)
            if exp_name in exp_grid.get("experiments", {}):
                exp_config = exp_grid["experiments"][exp_name]
                config.update(exp_config)
                if "train" in exp_config:
                    config["train"].update(exp_config["train"])
    
    return config


def evaluate_teacher(teacher, dataset, config):
    """Valuta teacher full context"""
    logger = get_logger(__name__)
    logger.info("Valutazione Teacher Full Context...")
    
    predictions = []
    references = []
    latencies = []
    timer = Timer()
    
    for i, example in enumerate(tqdm(dataset)):
        with timer.time("teacher_generate"):
            pred = teacher.generate_answer_vllm(
                example["context"],
                example["question"],
                max_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        predictions.append(pred)
        references.append(example["answer"])
        latencies.append(timer.get_mean("teacher_generate"))
    
    metrics = compute_metrics(predictions, references)
    metrics["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0.0
    metrics["method"] = "teacher_full"
    
    return metrics


def evaluate_student(compressor, reasoner, teacher, dataset, config, device):
    """Valuta student (compressor + reasoner)"""
    logger = get_logger(__name__)
    logger.info("Valutazione Student (Compressor + Reasoner)...")
    
    predictions = []
    references = []
    latencies = []
    timer = Timer()
    
    compressor.eval()
    
    for example in tqdm(dataset):
        with timer.time("student_generate"):
            # Estrai hidden states
            teacher_states_list = []
            layer_indices = config.get("layer_indices", [-6])
            
            for chunk_text in example["context_chunks"]:
                if teacher.hidden_states_available:
                    states = teacher.get_hidden_states(chunk_text, layer_indices)
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
            with torch.no_grad():
                latents = compressor(teacher_states_list)
            
            # Reasoner
            pred = reasoner.generate_with_latents(
                latents,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        predictions.append(pred)
        references.append(example["answer"])
        latencies.append(timer.get_mean("student_generate"))
    
    metrics = compute_metrics(predictions, references)
    metrics["avg_latency"] = sum(latencies) / len(latencies) if latencies else 0.0
    metrics["method"] = "student"
    
    return metrics


def evaluate_rag(rag_baseline, reasoner, dataset, config, device):
    """Valuta baseline RAG"""
    logger = get_logger(__name__)
    
    all_metrics = []
    
    for k in config["baselines"]["rag_k"]:
        logger.info(f"Valutazione RAG k={k}...")
        predictions = []
        references = []
        timer = Timer()
        
        for example in tqdm(dataset):
            with timer.time("rag_generate"):
                # Retrieval
                context = rag_baseline.prepare_context(
                    example["question"],
                    example["context_chunks"],
                    k
                )
                
                # Reasoner con contesto recuperato
                # Usa reasoner normalmente (non con latents)
                inputs = reasoner.tokenizer(
                    f"{context}\n\nQuestion: {example['question']}\nAnswer:",
                    return_tensors="pt",
                    truncation=True,
                    max_length=reasoner.max_ctx
                ).to(device)
                
                with torch.no_grad():
                    outputs = reasoner.model.generate(
                        **inputs,
                        max_new_tokens=config["eval"]["max_new_tokens"],
                        temperature=config["eval"]["temperature"] if config["eval"]["temperature"] > 0 else None,
                        do_sample=config["eval"]["temperature"] > 0,
                        pad_token_id=reasoner.tokenizer.eos_token_id
                    )
                
                pred = reasoner.tokenizer.decode(
                    outputs[0][inputs["input_ids"].shape[1]:],
                    skip_special_tokens=True
                )
            
            predictions.append(pred)
            references.append(example["answer"])
        
        metrics = compute_metrics(predictions, references)
        metrics["avg_latency"] = timer.get_mean("rag_generate")
        metrics["method"] = f"rag_k{k}"
        all_metrics.append(metrics)
    
    return all_metrics


def evaluate_summary(summarizer, reasoner, dataset, config, device):
    """Valuta baseline summary"""
    logger = get_logger(__name__)
    logger.info("Valutazione Summary Baseline...")
    
    predictions = []
    references = []
    timer = Timer()
    
    for example in tqdm(dataset):
        with timer.time("summary_generate"):
            # Summary
            context = summarizer.prepare_context(example["context_chunks"])
            
            # Reasoner
            inputs = reasoner.tokenizer(
                f"{context}\n\nQuestion: {example['question']}\nAnswer:",
                return_tensors="pt",
                truncation=True,
                max_length=reasoner.max_ctx
            ).to(device)
            
            with torch.no_grad():
                outputs = reasoner.model.generate(
                    **inputs,
                    max_new_tokens=config["eval"]["max_new_tokens"],
                    temperature=config["eval"]["temperature"] if config["eval"]["temperature"] > 0 else None,
                    do_sample=config["eval"]["temperature"] > 0,
                    pad_token_id=reasoner.tokenizer.eos_token_id
                )
            
            pred = reasoner.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
        
        predictions.append(pred)
        references.append(example["answer"])
    
    metrics = compute_metrics(predictions, references)
    metrics["avg_latency"] = timer.get_mean("summary_generate")
    metrics["method"] = "summary"
    
    return metrics


def evaluate_pooling(pooling, reasoner, teacher, dataset, config, device):
    """Valuta baseline pooling"""
    logger = get_logger(__name__)
    logger.info("Valutazione Pooling Baseline...")
    
    predictions = []
    references = []
    timer = Timer()
    
    for example in tqdm(dataset):
        with timer.time("pooling_generate"):
            # Pooling latents
            latents = pooling.get_latents(example["context_chunks"], device)
            
            # Reasoner con latents
            pred = reasoner.generate_with_latents(
                latents,
                example["question"],
                max_new_tokens=config["eval"]["max_new_tokens"],
                temperature=config["eval"]["temperature"]
            )
        
        predictions.append(pred)
        references.append(example["answer"])
    
    metrics = compute_metrics(predictions, references)
    metrics["avg_latency"] = timer.get_mean("pooling_generate")
    metrics["method"] = "pooling"
    
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--exp", type=str, default="S0")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config, args.exp)
    set_seed(42)
    logger = setup_logging(config)
    logger.info(f"Esperimento: {args.exp}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    
    # Paths
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Carica modelli
    logger.info("Caricamento modelli...")
    teacher = TeacherReader(config)
    reasoner = ReasonerWrapper(config)
    
    # Carica compressore se checkpoint specificato
    compressor = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            compressor = LatentCompressor(config).to(device)
            checkpoint = torch.load(checkpoint_path, map_location=device)
            compressor.load_state_dict(checkpoint["compressor_state"])
            logger.info(f"Compressore caricato da {checkpoint_path}")
        else:
            logger.warning(f"Checkpoint non trovato: {checkpoint_path}")
    
    # Carica dataset test
    data_dir = Path(config["paths"]["data_dir"])
    test_file = data_dir / "raw" / "synthetic_test.jsonl"
    
    teacher_tokenizer = get_tokenizer(config["teacher_model"])
    test_dataset = LatentCompressionDataset(
        str(test_file),
        teacher_tokenizer,
        config["chunk_size_tokens"],
        config["chunk_overlap_tokens"]
    )
    
    logger.info(f"Dataset test: {len(test_dataset)} esempi")
    
    # Valutazioni
    all_results = []
    
    reset_peak_memory()
    
    # 1. Teacher full
    teacher_metrics = evaluate_teacher(teacher, test_dataset, config)
    all_results.append(teacher_metrics)
    logger.info(f"Teacher: {teacher_metrics}")
    
    # 2. Student (se disponibile)
    if compressor is not None:
        student_metrics = evaluate_student(
            compressor, reasoner, teacher, test_dataset, config, device
        )
        all_results.append(student_metrics)
        logger.info(f"Student: {student_metrics}")
    
    # 3. RAG
    rag_baseline = RAGBaseline(config)
    rag_metrics_list = evaluate_rag(rag_baseline, reasoner, test_dataset, config, device)
    all_results.extend(rag_metrics_list)
    for m in rag_metrics_list:
        logger.info(f"RAG {m['method']}: {m}")
    
    # 4. Summary
    summarizer = SummarizerBaseline(config, teacher)
    summary_metrics = evaluate_summary(summarizer, reasoner, test_dataset, config, device)
    all_results.append(summary_metrics)
    logger.info(f"Summary: {summary_metrics}")
    
    # 5. Pooling
    pooling = PoolingBaseline(config, teacher)
    pooling_metrics = evaluate_pooling(
        pooling, reasoner, teacher, test_dataset, config, device
    )
    all_results.append(pooling_metrics)
    logger.info(f"Pooling: {pooling_metrics}")
    
    # VRAM stats
    vram = get_vram_usage()
    logger.info(f"VRAM peak: {vram['max_allocated']:.2f}GB")
    
    # Salva risultati
    results_file = output_dir / "report.json"
    with open(results_file, 'w') as f:
        json.dump({
            "experiment": args.exp,
            "results": all_results,
            "vram": vram
        }, f, indent=2)
    
    logger.info(f"Risultati salvati in {results_file}")
    
    return results_file


if __name__ == "__main__":
    main()



# Latent Compression for Long-Context LLM

Latent compression system for very long contexts (up to ~262k tokens) using a Reader/Teacher model (Qwen3 30B MoE AWQ) and a smaller Reasoner (Qwen3 4B).

## Architecture

1. **Reader/Teacher**: Qwen3 30B MoE AWQ 4bit reads the full context (up to 262k tokens)
2. **Compressor**: Perceiver-style resampler that compresses the context into much shorter latent token sequences
3. **Reasoner**: Qwen3 4B consumes only (latents + question) to generate the answer

## Structure

```
latent-compress-llm/
  configs/          # YAML configurations
  data/             # Datasets and cache
  src/              # Source code
  runs/             # Training checkpoints
  outputs/          # Reports and plots
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate synthetic dataset

```bash
python -m src.data.generate_synth --config configs/base.yaml
```

### 2. Train compressor

```bash
python -m src.train.train_compressor --config configs/base.yaml --exp S0
```

### 3. Evaluation

```bash
python -m src.eval.run_eval --config configs/base.yaml --exp S0
python -m src.eval.report --input outputs/report.json
```

## Experiments

- **S0**: Sanity check - 32k context, N_lat 1024
- **S1**: Main - 256k context, N_lat {8192, 4096, 1024}

## Baselines

1. Teacher full context (vLLM)
2. RAG top-k chunks
3. Summary per chunk
4. Pooling latents (mean/max)

## Assumptions

- Teacher available on vLLM port 8000
- If hidden states extraction fails with AWQ, fallback to logits-only or proxy model
- Reasoner uses Transformers to support inputs_embeds



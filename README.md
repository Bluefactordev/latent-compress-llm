# Latent Compression for Long-Context LLM

Sistema di compressione latente per contesti molto lunghi (fino a ~262k token) utilizzando un modello Reader/Teacher (Qwen3 30B MoE AWQ) e un Reasoner più piccolo (Qwen3 4B).

## Architettura

1. **Reader/Teacher**: Qwen3 30B MoE AWQ 4bit legge il contesto completo (fino a 262k token)
2. **Compressore**: Resampler stile Perceiver che comprime il contesto in sequenze di token latenti molto più corte
3. **Reasoner**: Qwen3 4B consuma solo (latenti + domanda) per generare la risposta

## Struttura

```
latent-compress-llm/
  configs/          # Configurazioni YAML
  data/             # Dataset e cache
  src/              # Codice sorgente
  runs/             # Checkpoint training
  outputs/          # Report e plot
```

## Setup

```bash
pip install -r requirements.txt
```

## Utilizzo

### 1. Generare dataset sintetico

```bash
python -m src.data.generate_synth --config configs/base.yaml
```

### 2. Training compressore

```bash
python -m src.train.train_compressor --config configs/base.yaml --exp S0
```

### 3. Valutazione

```bash
python -m src.eval.run_eval --config configs/base.yaml --exp S0
python -m src.eval.report --input outputs/report.json
```

## Esperimenti

- **S0**: Sanity check - contesto 32k, N_lat 1024
- **S1**: Main - contesto 256k, N_lat {8192, 4096, 1024}

## Baseline

1. Teacher full context (vLLM)
2. RAG top-k chunks
3. Summary per chunk
4. Pooling latents (mean/max)

## Assunzioni

- Teacher disponibile su vLLM porta 8000
- Se estrazione hidden states fallisce con AWQ, si usa fallback logits-only o proxy model
- Reasoner usa Transformers per supportare inputs_embeds



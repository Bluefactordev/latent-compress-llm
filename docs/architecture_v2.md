# Architecture V2 - Latent Compression LLM

## Overview

Pipeline per compressione semantica latente di contesti molto lunghi senza summarization testuale.

## Componenti Principali

### 1. Reader/Teacher (Qwen3-30B)
- **Ruolo**: Legge contesto completo (fino a 256k+ token)
- **Output**: Hidden states multi-layer + risposte ground truth
- **Modalità**: Frozen durante training

### 2. Latent Compressor (Perceiver-style)

#### 2.1 Input Processing
```
Teacher Hidden States (multi-layer):
  H_l6  [seq_len, d_teacher]  ← layer -6
  H_l18 [seq_len, d_teacher]   ← layer -18  
  H_l30 [seq_len, d_teacher]   ← layer -30

→ Proiezione per layer:
  H_l6_proj  = Linear(H_l6 → d_lat)
  H_l18_proj = Linear(H_l18 → d_lat)
  H_l30_proj = Linear(H_l30 → d_lat)

→ Concatena:
  H = concat(H_l6_proj, H_l18_proj, H_l30_proj)
  H = Linear(H → d_lat)  # Final projection
```

#### 2.2 Temporal Inductive Bias
```
Per ogni token in teacher states:
  time_bucket = floor(token_pos / chunk_len * NUM_BUCKETS)
  H_token += time_bucket_embedding[time_bucket]

NUM_BUCKETS = 8
```

#### 2.3 Compressor Blocks
```
Latent Tokens: [N_lat, d_lat] (learnable)

Per ogni chunk:
  For each block:
    1. Cross-Attention:
       Q = latents
       K, V = teacher_states (con time_bucket embeddings)
    2. Self-Attention:
       Q, K, V = latents
    3. Feed-Forward:
       latents = SwiGLU(latents)
    4. Residual + RMSNorm

Output per chunk: [N_lat, d_lat]
Global: concat([latents_chunk_0, ..., latents_chunk_N])
```

#### 2.4 Global Resampler (opzionale)
```
Se total_latents > reasoner_ctx:
  global_latents = GlobalResampler(chunk_latents)
  Output: [N_global, d_lat]
```

### 3. Latent → Reasoner Projection

**CRITICO**: Latents devono essere proiettati nello spazio del reasoner.

```
latents [N_lat, d_lat]
  ↓
latent_to_reasoner = Linear(d_lat → d_reasoner)
  ↓
reasoner_embeds [N_lat, d_reasoner]
```

### 4. Reasoner

#### 4.1 Modalità Controllo (Fase 1)
```
reasoner_mode: "teacher_control"
reasoner_model: Qwen3-30B (stesso del teacher)

Input: [reasoner_embeds] + [question_token_ids]
Output: answer
```

#### 4.2 Modalità Transfer (Fase 8)
```
reasoner_mode: "transfer"
reasoner_model: Qwen3-4B

Input: [reasoner_embeds] + [question_token_ids]
Output: answer
```

## Training Strategy

### Freeze Policy
- ❄️ Teacher: **FROZEN**
- ❄️ Reasoner: **FROZEN**
- 🔥 Train: **ONLY** compressor + latent_to_reasoner

### Loss Function

#### Primary Loss
```
L_primary = CrossEntropy(
  reasoner_output,
  teacher_answer
)
```

#### Auxiliary Loss (scegli UNA)

**Opzione A - Answer Localization**
```
L_aux = CrossEntropy(
  predict_chunk_id(latents),
  true_chunk_id
)
L_total = L_primary + 0.1 * L_aux
```

**Opzione B - Mean Alignment**
```
L_aux = MSE(
  mean(latents),
  mean(teacher_hidden_states)
)
L_total = L_primary + 0.1 * L_aux
```

## Flusso Dati

```
Long Context (256k tokens)
  ↓
Chunking (64k per chunk, overlap 1k)
  ↓
Teacher Processing (per chunk):
  - Extract H_l6, H_l18, H_l30
  - Add time_bucket embeddings
  ↓
Compressor (per chunk):
  - Cross-attn: latents ← teacher_states
  - Self-attn: latents
  - FF + residual
  ↓
Global Resampler (se necessario)
  ↓
Latent Projection:
  latents → reasoner_embeds
  ↓
Reasoner:
  [reasoner_embeds] + [question] → answer
```

## Metriche

### Quality
- Exact Match (EM)
- Token-level F1
- Needle accuracy (per distanza)
- Multi-hop accuracy
- Versioning accuracy (latest vs old)

### System
- Latency (teacher, compressor, reasoner)
- VRAM usage
- Compression ratio

## Ablation Tests (Fase 7)

1. **Latents Shuffled**: Randomizza ordine latents
2. **Latents Zeroed**: Sostituisci con zeri
3. **Only Last Chunk**: Usa solo latents ultimo chunk
4. **Random Latents**: Latents completamente random

Se accuracy non cambia → non c'è compressione, solo prompt tuning.


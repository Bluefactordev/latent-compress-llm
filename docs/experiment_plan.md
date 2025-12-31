# Experiment Plan - Latent Compression LLM

## Principio Guida

**Prima dimostrare che la compressione NON distrugge informazione. Solo dopo dimostrare che un modello più piccolo può usarla.**

## Fase 0 - Pulizia e Documentazione ✅

- [x] README_DEV.md
- [x] docs/architecture_v2.md
- [x] docs/experiment_plan.md

## Fase 1 - Esperimento di Controllo (PRIORITÀ MASSIMA)

### 1.1 Reasoner = Teacher (30B)

**Implementazione**:
- Aggiungere `reasoner_mode: "teacher_control"` in config
- Quando attivo: reasoner usa stesso modello del teacher (30B)
- Reasoner NON legge testo, solo `[LATENTS] + [QUESTION]`

**Scopo**:
- Se fallisce → compressione è sbagliata
- Se funziona → problema è capacity transfer

**Esperimenti**:
1. Contesto corto: 16k-32k (1-2 chunk)
2. Contesto medio: 64k (4 chunk)
3. Contesto lungo: 128k-256k (8-16 chunk)

**Metriche Minime**:
- teacher(full context) answer
- reasoner(latents) answer
- Exact match
- Token-level F1
- Distanza risposta da punto inserimento (needle)

**Criterio STOP**:
- Se accuracy < 70% su needle semplici → **STOP**, non andare avanti

## Fase 2 - Multi-Layer Teacher Features (OBBLIGATORIA)

### 2.1 Estrazione Multi-Layer

**Implementazione**:
```python
TEACHER_LAYERS = [-6, -18, -30]

Per ogni chunk:
  H_l6 = get_hidden_states(chunk, layer=-6)
  H_l18 = get_hidden_states(chunk, layer=-18)
  H_l30 = get_hidden_states(chunk, layer=-30)
  
  # Proiezione separata
  H_l6_proj = Linear(H_l6 → d_lat)
  H_l18_proj = Linear(H_l18 → d_lat)
  H_l30_proj = Linear(H_l30 → d_lat)
  
  # Concatena o somma pesata
  H = concat(H_l6_proj, H_l18_proj, H_l30_proj)
  H = Linear(H → d_lat)
```

**Motivo**: Facts, relations, structure vivono in layer diversi. Single-layer = perdita strutturale.

## Fase 3 - Temporal/Positional Inductive Bias (CRITICO)

### 3.1 Time-Bucket Embedding

**Implementazione**:
```python
NUM_BUCKETS = 8

Per ogni token in teacher states:
  token_pos = posizione token nel chunk
  time_bucket = floor(token_pos / chunk_len * NUM_BUCKETS)
  
  Durante cross-attention:
    K_token += time_bucket_embedding[time_bucket]
```

**Motivo**: Dà ai latents nozione di early/mid/late, "recency" debole ma essenziale.

### 3.2 Latent Ordering

**Implementazione**:
- Ogni latent riceve:
  - `chunk_id_embedding[chunk_id]`
  - `latent_idx_embedding[latent_index]`

**Obbligatorio**, non opzionale.

## Fase 4 - Proiezione Latents → Reasoner Space (OBBLIGATORIA)

**Implementazione**:
```python
latent_to_reasoner = nn.Linear(d_lat, d_reasoner)

Pipeline:
  latents [N_lat, d_lat]
    ↓
  latent_to_reasoner
    ↓
  reasoner_embeds [N_lat, d_reasoner]
    ↓
  inputs_embeds per reasoner
```

**Motivo**: Latents sono off-manifold. Devono essere proiettati nello spazio del reasoner.

## Fase 5 - Training Strategy (RISTRUTTURATA)

### 5.1 Freeze Policy

- ❄️ Teacher: **FROZEN**
- ❄️ Reasoner: **FROZEN**
- 🔥 Train: **ONLY** compressor + latent_to_reasoner

### 5.2 Loss

**Primary Loss** (obbligatoria):
- Cross-entropy sulla risposta (teacher answer o ground truth)

**Auxiliary Loss** (scegli UNA):

**Opzione A - Answer Localization**:
```python
predicted_chunk = predict_chunk_id(latents)
L_aux = CrossEntropy(predicted_chunk, true_chunk_id)
L_total = L_primary + 0.1 * L_aux
```

**Opzione B - Mean Alignment**:
```python
L_aux = MSE(mean(latents), mean(teacher_hidden_states))
L_total = L_primary + 0.1 * L_aux
```

## Fase 6 - Dataset (VINCOLI CHIARI)

### 6.1 Ordine Esperimenti

**NON partire da 256k**. Sequenza obbligatoria:

1. **16k** (1 chunk)
   - Needle semplice
   - Needle con override
   - Multi-hop 2 step
   - Rumore 95%

2. **32k** (2 chunk)
   - Stessi test

3. **64k** (4 chunk)
   - Stessi test

4. **128k** (8 chunk)
   - Stessi test

5. **256k** (16 chunk)
   - Stessi test

### 6.2 Test Obbligatori per Ogni Lunghezza

- Needle semplice
- Needle con override (stesso key, valore diverso)
- Multi-hop 2 step
- Rumore 95%

## Fase 7 - Ablation "Killer" (OBBLIGATORIA)

**Implementa test automatici**:

1. **Latents Shuffled**: Randomizza ordine latents
2. **Latents Zeroed**: Sostituisci con zeri
3. **Only Last Chunk**: Usa solo latents ultimo chunk
4. **Random Latents**: Latents completamente random

**Criterio**: Se accuracy non cambia → stai facendo prompt tuning, non compressione.

## Fase 8 - Solo Dopo: Transfer a 4B

**Quando**:
- 30B(reasoner) + latents ≈ 30B(full context)

**Allora**:
- Sostituisci reasoner con Qwen3-4B
- Non cambiare nient'altro
- Misura degrado

**Questo è il risultato scientifico**.

## Fase 9 - Criteri di Stop

**FERMA tutto se**:
- Controllo 30B fallisce (< 70% su needle semplici)
- Auxiliary loss non scende
- Latents non usati (ablation invarianti)

**CONTINUA solo se**:
- Clear signal che latents contengono informazione
- Accuracy controllo 30B > 70%

## Output Finali Attesi

1. `docs/architecture_v2.md` con schema ✅
2. `docs/experiment_plan.md` ✅
3. Metriche per:
   - Full context
   - Latents + 30B
   - Latents + 4B
4. Almeno 1 grafico:
   - Accuracy vs compression ratio

## Timeline Stimata

- Fase 1: 2-3 giorni (critica)
- Fase 2-4: 2-3 giorni (implementazione)
- Fase 5: 1 giorno (training)
- Fase 6: 1-2 giorni (dataset)
- Fase 7: 1 giorno (ablation)
- Fase 8: 1 giorno (transfer)

**Totale**: ~10-12 giorni


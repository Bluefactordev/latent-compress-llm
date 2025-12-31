# Implementation Status - V2 Architecture

## ✅ Fase 0 - Documentazione
- [x] README_DEV.md creato
- [x] docs/architecture_v2.md creato
- [x] docs/experiment_plan.md creato
- [x] Problemi noti documentati

## ✅ Fase 1 - Reasoner = Teacher (30B)
- [x] `reasoner_mode: "teacher_control"` implementato in config
- [x] ReasonerWrapper supporta modalità teacher_control
- [x] Quando attivo, reasoner usa stesso modello del teacher (30B)
- [x] Reasoner NON legge testo, solo `[LATENTS] + [QUESTION]`

**File modificati**:
- `configs/base.yaml`: aggiunto `reasoner_mode`
- `src/models/reasoner_wrapper.py`: supporto teacher_control

## ✅ Fase 2 - Multi-Layer Teacher Features
- [x] Estrazione multi-layer: `teacher_layers: [-6, -18, -30]`
- [x] Proiezione separata per ogni layer
- [x] Concatena layer: `H = concat(H_l6, H_l18, H_l30)`
- [x] Proiezione finale: `Linear(H → d_lat)`

**File modificati**:
- `src/models/teacher_reader.py`: `get_multi_layer_hidden_states()`
- `src/models/compressor.py`: supporto multi-layer in forward
- `configs/base.yaml`: `teacher_layers` config

## ✅ Fase 3 - Temporal/Positional Inductive Bias
- [x] Time-bucket embeddings: `num_time_buckets: 8`
- [x] Time-bucket calcolato: `floor(token_pos / chunk_len * NUM_BUCKETS)`
- [x] Time-bucket embeddings aggiunti alle keys durante cross-attention
- [x] Chunk ID embedding per latents
- [x] Latent index embedding per latents

**File modificati**:
- `src/models/compressor.py`: `CrossAttentionBlock` con time-bucket
- `configs/base.yaml`: `num_time_buckets` config

## ✅ Fase 4 - Proiezione Latents → Reasoner Space
- [x] `LatentToReasonerProjection` module creato
- [x] Proiezione: `Linear(d_lat → d_reasoner)`
- [x] LayerNorm per stabilità
- [x] Integrato in ReasonerWrapper

**File creati**:
- `src/models/latent_projection.py`

**File modificati**:
- `src/models/reasoner_wrapper.py`: usa proiezione prima di inputs_embeds

## ✅ Fase 5 - Training Strategy
- [x] Freeze policy: Teacher frozen, Reasoner frozen
- [x] Train ONLY: compressor + latent_projection
- [x] Primary loss: Cross-entropy su risposta
- [x] Auxiliary loss: Alignment (mean alignment) implementato
- [x] Loss combinata: `L_total = L_primary + 0.1 * L_aux`

**File creati**:
- `src/train/losses.py`: `CompressionLoss` con auxiliary

**File modificati**:
- `src/train/train_compressor.py`: freeze policy, loss combinata
- `configs/base.yaml`: `auxiliary_loss_type`, `auxiliary_weight`

## ⏳ Fase 6 - Dataset Sequenziale
**Status**: Da implementare
- [ ] Generatore dati con sequenza: 16k → 32k → 64k → 128k → 256k
- [ ] Test obbligatori per ogni lunghezza:
  - Needle semplice
  - Needle con override
  - Multi-hop 2 step
  - Rumore 95%

## ⏳ Fase 7 - Ablation Tests
**Status**: Da implementare
- [ ] Latents shuffled
- [ ] Latents zeroed
- [ ] Only last chunk latents
- [ ] Random latents

## ⏳ Fase 8 - Transfer a 4B
**Status**: Da implementare (solo se Fase 1 passa)
- [ ] Quando 30B(reasoner) + latents ≈ 30B(full context)
- [ ] Sostituisci reasoner con Qwen3-4B
- [ ] Misura degrado

## Note Implementative

### Backward Compatibility
- `get_hidden_states()` mantiene compatibilità single-layer
- `get_multi_layer_hidden_states()` nuovo metodo per multi-layer
- Compressore supporta sia dict (multi-layer) che tensor (single-layer)

### Config Updates
```yaml
reasoner_mode: "teacher_control"  # o "transfer"
teacher_layers: [-6, -18, -30]
num_time_buckets: 8
train:
  auxiliary_loss_type: "alignment"  # o "localization"
  auxiliary_weight: 0.1
```

### Prossimi Passi
1. Testare Fase 1 (teacher_control) su contesto corto (16k-32k)
2. Verificare che multi-layer extraction funzioni
3. Validare temporal bias su dataset con versioning
4. Implementare Fase 6 (dataset sequenziale)
5. Eseguire esperimenti controllo

## Bug Noti / Da Sistemare
1. `train_step` richiede `loss_fn` come parametro (già corretto)
2. Eval loop deve essere aggiornato per multi-layer
3. Ablation tests da implementare


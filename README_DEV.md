# README DEV - Latent Compression LLM

## Stato Attuale

### Implementazione Base
- ✅ Struttura repository completa
- ✅ Generatore dati sintetici
- ✅ Teacher vLLM (porta 8000) funzionante
- ✅ Compressore Perceiver-style base
- ✅ Reasoner wrapper con inputs_embeds
- ✅ Training loop base
- ✅ Baseline (RAG, summary, pooling)
- ✅ Sistema valutazione

### Problemi Noti (CRITICI)

#### 1. Mismatch Teacher/Reasoner
**Problema**: Reasoner è 4B mentre Teacher è 30B. Impossibile distinguere se il fallimento è dovuto a:
- Compressione che distrugge informazione
- Capacity transfer insufficiente

**Soluzione**: Fase 1 - Reasoner = Teacher (30B) per controllo

#### 2. Single-Layer Hidden States
**Problema**: Estrazione solo da layer -6. Perdita di informazioni strutturali:
- Facts in layer early
- Relations in layer middle  
- Structure in layer late

**Soluzione**: Fase 2 - Multi-layer extraction [-6, -18, -30]

#### 3. No Temporal Inductive Bias
**Problema**: Latents non hanno nozione di:
- Ordine temporale (early/mid/late)
- Versioning (latest vs old)
- Override detection

**Soluzione**: Fase 3 - Time-bucket embeddings + chunk ordering

#### 4. Loss Troppo Indiretta
**Problema**: Solo cross-entropy su risposta finale. Gradient signal debole.

**Soluzione**: Fase 5 - Auxiliary loss (localization o alignment)

#### 5. Latents Off-Manifold
**Problema**: Latents in d_lat (512) iniettati direttamente in reasoner che si aspetta d_reasoner (diverso).

**Soluzione**: Fase 4 - Proiezione latents → reasoner space

## Run Precedente

Il run precedente è **diagnostico, non conclusivo**. Serve per:
- Verificare che l'infrastruttura funzioni
- Identificare i problemi sopra elencati
- Non rappresenta un risultato scientifico valido

## Prossimi Passi

Vedi `docs/experiment_plan.md` per il piano dettagliato.

**ORDINE OBBLIGATORIO**:
1. Fase 0: Documentazione (questo file)
2. Fase 1: Esperimento controllo (30B → 30B)
3. Fase 2: Multi-layer features
4. Fase 3: Temporal bias
5. Fase 4: Proiezione latents
6. Fase 5: Training strategy
7. Fase 6: Dataset sequenziale
8. Fase 7: Ablation tests
9. Fase 8: Transfer a 4B (SOLO se Fase 1 passa)

## Assunzioni

1. Teacher Qwen3-30B disponibile su vLLM porta 8000
2. Tokenizer Qwen2.5 compatibile con Qwen3
3. GPU RTX Pro 6000 96GB sufficiente per training
4. Hidden states extraction possibile (con fallback se necessario)

## Filosofia

**Questo progetto NON è su velocità o productizzazione.**

**È sulla validazione se informazione semantica in un Transformer può essere proiettata in uno spazio latente molto più piccolo senza perdita catastrofica.**


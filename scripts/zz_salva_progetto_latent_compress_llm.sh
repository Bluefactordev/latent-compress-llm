#!/bin/bash
# Esegui questo script dalla root del progetto latent-compress-llm
# Creerà un file chiamato sorgenti/latent_compress_llm_completo_NNN.txt
# Contiene tutto il progetto: Core, Models, Training, Evaluation, Data, Utils, Configs

# Crea la directory sorgenti se non esiste
mkdir -p sorgenti

# Trova il prossimo numero di versione disponibile
BASE_NAME="sorgenti/latent_compress_llm_completo"
COUNTER=1
while [ -f "${BASE_NAME}_$(printf "%03d" $COUNTER).txt" ]; do
    COUNTER=$((COUNTER + 1))
done

OUTPUT_FILE="${BASE_NAME}_$(printf "%03d" $COUNTER).txt"
echo "📝 Creando versione $(printf "%03d" $COUNTER): $OUTPUT_FILE"

echo "--- START OF FILE latent_compress_llm_completo_$(printf "%03d" $COUNTER).txt ---" > "$OUTPUT_FILE"
echo "Generato il: $(date)" >> "$OUTPUT_FILE"
echo "Versione: $(printf "%03d" $COUNTER)" >> "$OUTPUT_FILE"
echo "Sistema di Compressione Latente per Long-Context LLM" >> "$OUTPUT_FILE"
echo "⚠️ IMPORTANTE: Sistema completo per compressione contesti molto lunghi (fino a ~262k token)" >> "$OUTPUT_FILE"
echo "📦 ARCHITETTURA:" >> "$OUTPUT_FILE"
echo "   ✅ Reader/Teacher: Qwen3 30B MoE AWQ 4bit legge contesto completo (fino a 262k token)" >> "$OUTPUT_FILE"
echo "   ✅ Compressore: Resampler stile Perceiver che comprime contesto in sequenze latenti corte" >> "$OUTPUT_FILE"
echo "   ✅ Reasoner: Qwen3 4B consuma solo (latenti + domanda) per generare risposta" >> "$OUTPUT_FILE"
echo "   ✅ Baseline: Teacher full context, RAG top-k chunks, Summary per chunk, Pooling latents" >> "$OUTPUT_FILE"

FILES_TO_CAT=(
    # === DOCUMENTAZIONE PRINCIPALE ===
    "README.md"                                  # Documentazione principale progetto

    # === CONFIGURAZIONE ===
    "requirements.txt"                           # Dipendenze Python
    "configs/base.yaml"                          # Configurazione base
    "configs/exp_grid.yaml"                      # Griglia esperimenti

    # === CORE MODULES ===
    "src/__init__.py"                            # Package initialization

    # === DATA MODULE ===
    "src/data/__init__.py"                       # Data package initialization
    "src/data/dataset.py"                        # Dataset loader e utilities
    "src/data/generate_synth.py"                 # Generazione dataset sintetico

    # === MODELS MODULE ===
    "src/models/__init__.py"                     # Models package initialization
    "src/models/compressor.py"                   # Compressore latente (Resampler Perceiver-style)
    "src/models/teacher_reader.py"               # Wrapper per Teacher/Reader (Qwen3 30B)
    "src/models/reasoner_wrapper.py"             # Wrapper per Reasoner (Qwen3 4B)
    "src/models/rag_baseline.py"                 # Baseline RAG top-k chunks
    "src/models/summarizer_baseline.py"          # Baseline summary per chunk
    "src/models/pooling_baseline.py"             # Baseline pooling latents (mean/max)

    # === TRAINING MODULE ===
    "src/train/__init__.py"                      # Training package initialization
    "src/train/train_compressor.py"              # Training loop compressore
    "src/train/distill.py"                       # Distillation utilities

    # === EVALUATION MODULE ===
    "src/eval/__init__.py"                       # Evaluation package initialization
    "src/eval/run_eval.py"                       # Esecuzione valutazione
    "src/eval/report.py"                         # Generazione report e plot

    # === UTILS MODULE ===
    "src/utils/__init__.py"                      # Utils package initialization
    "src/utils/logging.py"                       # Sistema logging
    "src/utils/metrics.py"                       # Metriche valutazione
    "src/utils/tokenization.py"                  # Utilities tokenizzazione
    "src/utils/timers.py"                        # Timing utilities
    "src/utils/seed.py"                          # Seed management
    "src/utils/vram.py"                          # VRAM monitoring
)

FOUND_FILES=0
MISSING_FILES=0
TOTAL_SIZE=0

for file in "${FILES_TO_CAT[@]}"; do
    if [ -f "$file" ]; then
        echo "" >> "$OUTPUT_FILE"
        echo "=== START OF FILE $file ===" >> "$OUTPUT_FILE"
        echo "Percorso: $file" >> "$OUTPUT_FILE"
        echo "Dimensione: $(wc -c < "$file") bytes" >> "$OUTPUT_FILE"
        echo "Ultima modifica: $(stat -c %y "$file" 2>/dev/null || stat -f %Sm "$file")" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        cat "$file" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        echo "=== END OF FILE $file ===" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        
        FOUND_FILES=$((FOUND_FILES + 1))
        TOTAL_SIZE=$((TOTAL_SIZE + $(wc -c < "$file")))
    else
        echo "" >> "$OUTPUT_FILE"
        echo "=== FILE NOT FOUND: $file ===" >> "$OUTPUT_FILE"
        echo "" >> "$OUTPUT_FILE"
        MISSING_FILES=$((MISSING_FILES + 1))
    fi
done

# Aggiungi sezione con architettura e design
echo "" >> "$OUTPUT_FILE"
echo "=== ARCHITETTURA SISTEMA ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "🎯 OBIETTIVI:" >> "$OUTPUT_FILE"
echo "  - Compressione contesti molto lunghi: Fino a ~262k token in sequenze latenti corte" >> "$OUTPUT_FILE"
echo "  - Efficienza computazionale: Teacher grande legge tutto, Reasoner piccolo genera risposta" >> "$OUTPUT_FILE"
echo "  - Training end-to-end: Compressore addestrato con distillation da Teacher" >> "$OUTPUT_FILE"
echo "  - Baseline comparison: Confronto con RAG, Summary, Pooling" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "✨ COMPONENTI PRINCIPALI:" >> "$OUTPUT_FILE"
echo "  1. Teacher/Reader (Qwen3 30B MoE AWQ):" >> "$OUTPUT_FILE"
echo "     - Legge contesto completo fino a 262k token" >> "$OUTPUT_FILE"
echo "     - Disponibile su vLLM porta 8000" >> "$OUTPUT_FILE"
echo "     - Estrae hidden states da layer specifici (default: -6)" >> "$OUTPUT_FILE"
echo "     - Fallback: logits-only o proxy model se AWQ non supporta hidden states" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "  2. Compressore (Resampler Perceiver-style):" >> "$OUTPUT_FILE"
echo "     - Comprime hidden states in sequenze latenti (N_lat tokens)" >> "$OUTPUT_FILE"
echo "     - Supporta resampler globale opzionale (n_latents_global)" >> "$OUTPUT_FILE"
echo "     - Architettura: n_compressor_blocks, n_heads, ff_dim configurabili" >> "$OUTPUT_FILE"
echo "     - Output: token latenti per Reasoner" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "  3. Reasoner (Qwen3 4B):" >> "$OUTPUT_FILE"
echo "     - Consuma solo (latenti + domanda) invece di contesto completo" >> "$OUTPUT_FILE"
echo "     - Usa Transformers per supportare inputs_embeds" >> "$OUTPUT_FILE"
echo "     - Genera risposta finale" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "  4. Baseline Systems:" >> "$OUTPUT_FILE"
echo "     - Teacher full context: Baseline upper bound" >> "$OUTPUT_FILE"
echo "     - RAG top-k chunks: Retrieval top-k chunks rilevanti" >> "$OUTPUT_FILE"
echo "     - Summary per chunk: Summary di ogni chunk" >> "$OUTPUT_FILE"
echo "     - Pooling latents: Mean/max pooling su hidden states" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== FLUSSO DI ESECUZIONE ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "1. GENERAZIONE DATASET:" >> "$OUTPUT_FILE"
echo "   python -m src.data.generate_synth --config configs/base.yaml" >> "$OUTPUT_FILE"
echo "   - Genera dataset sintetico con contesti lunghi" >> "$OUTPUT_FILE"
echo "   - Configurabile: train_size, val_size, test_size, num_keys, noise_ratio" >> "$OUTPUT_FILE"
echo "   - Range contesto: min_context_tokens - max_context_tokens" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "2. TRAINING COMPRESSORE:" >> "$OUTPUT_FILE"
echo "   python -m src.train.train_compressor --config configs/base.yaml --exp S0" >> "$OUTPUT_FILE"
echo "   - Training loop con distillation da Teacher" >> "$OUTPUT_FILE"
echo "   - Configurabile: lr, batch_size, gradient_accumulation_steps, max_steps" >> "$OUTPUT_FILE"
echo "   - Checkpoint salvati in runs/" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "3. VALUTAZIONE:" >> "$OUTPUT_FILE"
echo "   python -m src.eval.run_eval --config configs/base.yaml --exp S0" >> "$OUTPUT_FILE"
echo "   - Valutazione su test set" >> "$OUTPUT_FILE"
echo "   - Confronto con baseline" >> "$OUTPUT_FILE"
echo "   - Genera report.json" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "4. REPORT:" >> "$OUTPUT_FILE"
echo "   python -m src.eval.report --input outputs/report.json" >> "$OUTPUT_FILE"
echo "   - Genera plot e statistiche" >> "$OUTPUT_FILE"
echo "   - Output in outputs/" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== ESPERIMENTI CONFIGURATI ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "S0 - Sanity Check:" >> "$OUTPUT_FILE"
echo "  - Contesto: 32k token" >> "$OUTPUT_FILE"
echo "  - N_lat: 1024" >> "$OUTPUT_FILE"
echo "  - Global resampler: false" >> "$OUTPUT_FILE"
echo "  - Max steps: 2000" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "S1_4096 - Main 4096:" >> "$OUTPUT_FILE"
echo "  - Contesto: 256k token" >> "$OUTPUT_FILE"
echo "  - N_lat: 4096" >> "$OUTPUT_FILE"
echo "  - Global resampler: true" >> "$OUTPUT_FILE"
echo "  - Max steps: 5000" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "S1_8192 - Main 8192:" >> "$OUTPUT_FILE"
echo "  - Contesto: 256k token" >> "$OUTPUT_FILE"
echo "  - N_lat: 8192" >> "$OUTPUT_FILE"
echo "  - Global resampler: true" >> "$OUTPUT_FILE"
echo "  - Max steps: 5000" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "S1_1024 - Main 1024:" >> "$OUTPUT_FILE"
echo "  - Contesto: 256k token" >> "$OUTPUT_FILE"
echo "  - N_lat: 1024" >> "$OUTPUT_FILE"
echo "  - Global resampler: true" >> "$OUTPUT_FILE"
echo "  - Max steps: 5000" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== STRUTTURA PROGETTO ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "latent-compress-llm/" >> "$OUTPUT_FILE"
echo "├── configs/          # Configurazioni YAML" >> "$OUTPUT_FILE"
echo "│   ├── base.yaml     # Config base" >> "$OUTPUT_FILE"
echo "│   └── exp_grid.yaml # Griglia esperimenti" >> "$OUTPUT_FILE"
echo "├── data/             # Dataset e cache" >> "$OUTPUT_FILE"
echo "├── src/              # Codice sorgente" >> "$OUTPUT_FILE"
echo "│   ├── data/         # Dataset e generazione" >> "$OUTPUT_FILE"
echo "│   ├── models/       # Modelli: compressor, teacher, reasoner, baseline" >> "$OUTPUT_FILE"
echo "│   ├── train/        # Training loop e distillation" >> "$OUTPUT_FILE"
echo "│   ├── eval/         # Valutazione e report" >> "$OUTPUT_FILE"
echo "│   └── utils/        # Utilities: logging, metrics, tokenization, etc." >> "$OUTPUT_FILE"
echo "├── runs/             # Checkpoint training" >> "$OUTPUT_FILE"
echo "├── outputs/          # Report e plot" >> "$OUTPUT_FILE"
echo "├── README.md         # Documentazione principale" >> "$OUTPUT_FILE"
echo "└── requirements.txt  # Dipendenze Python" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== ASSUNZIONI E LIMITAZIONI ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "✅ ASSUNZIONI:" >> "$OUTPUT_FILE"
echo "  - Teacher disponibile su vLLM porta 8000" >> "$OUTPUT_FILE"
echo "  - Se estrazione hidden states fallisce con AWQ, si usa fallback logits-only o proxy model" >> "$OUTPUT_FILE"
echo "  - Reasoner usa Transformers per supportare inputs_embeds" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "⚠️  LIMITAZIONI:" >> "$OUTPUT_FILE"
echo "  - Teacher AWQ potrebbe non supportare estrazione hidden states" >> "$OUTPUT_FILE"
echo "  - Compressione introduce perdita di informazione" >> "$OUTPUT_FILE"
echo "  - Training richiede GPU con VRAM sufficiente" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== CONFIGURAZIONE COMPRESSORE ===" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Parametri principali:" >> "$OUTPUT_FILE"
echo "  - n_latents: Numero token latenti (default: 4096)" >> "$OUTPUT_FILE"
echo "  - d_lat: Dimensione embedding latenti (default: 512)" >> "$OUTPUT_FILE"
echo "  - n_latents_global: Token latenti globali opzionali (default: 8192)" >> "$OUTPUT_FILE"
echo "  - use_global_resampler: Abilita resampler globale (default: true)" >> "$OUTPUT_FILE"
echo "  - n_compressor_blocks: Numero blocchi transformer (default: 2)" >> "$OUTPUT_FILE"
echo "  - n_heads: Numero attention heads (default: 8)" >> "$OUTPUT_FILE"
echo "  - ff_dim: Dimensione feed-forward (default: 2048)" >> "$OUTPUT_FILE"
echo "" >> "$OUTPUT_FILE"
echo "Layer hidden states:" >> "$OUTPUT_FILE"
echo "  - layer_indices: Indici layer da cui estrarre hidden states (default: [-6])" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== STATISTICHE PROGETTO ===" >> "$OUTPUT_FILE"
echo "Data generazione: $(date)" >> "$OUTPUT_FILE"
echo "Totale file processati: ${#FILES_TO_CAT[@]}" >> "$OUTPUT_FILE"
echo "File trovati: $FOUND_FILES" >> "$OUTPUT_FILE"
echo "File mancanti: $MISSING_FILES" >> "$OUTPUT_FILE"
echo "Dimensione totale codice: $((TOTAL_SIZE / 1024)) KB" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== COMPONENTI INCLUSI ===" >> "$OUTPUT_FILE"
echo "✅ Data Module - Dataset loader e generazione sintetica" >> "$OUTPUT_FILE"
echo "✅ Models Module - Compressore, Teacher, Reasoner, Baseline" >> "$OUTPUT_FILE"
echo "✅ Training Module - Training loop e distillation" >> "$OUTPUT_FILE"
echo "✅ Evaluation Module - Valutazione e report generation" >> "$OUTPUT_FILE"
echo "✅ Utils Module - Logging, metrics, tokenization, timers, seed, vram" >> "$OUTPUT_FILE"
echo "✅ Configuration - Config YAML per base e esperimenti" >> "$OUTPUT_FILE"
echo "✅ Documentation - README principale" >> "$OUTPUT_FILE"
echo "✅ Dependencies - requirements.txt completo" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== DESIGN PRINCIPLES ===" >> "$OUTPUT_FILE"
echo "1. Modularità: Componenti separati e riutilizzabili" >> "$OUTPUT_FILE"
echo "2. Configurabilità: Tutto configurabile via YAML" >> "$OUTPUT_FILE"
echo "3. Baseline comparison: Confronto sistematico con metodi alternativi" >> "$OUTPUT_FILE"
echo "4. End-to-end training: Training completo con distillation" >> "$OUTPUT_FILE"
echo "5. Scalabilità: Supporto contesti molto lunghi (fino a 262k token)" >> "$OUTPUT_FILE"
echo "6. Efficienza: Reasoner piccolo consuma solo latenti compressi" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "=== DIPENDENZE PRINCIPALI ===" >> "$OUTPUT_FILE"
echo "  - torch>=2.0.0: PyTorch per training e inference" >> "$OUTPUT_FILE"
echo "  - transformers>=4.40.0: Hugging Face Transformers" >> "$OUTPUT_FILE"
echo "  - vllm>=0.4.0: vLLM per Teacher inference" >> "$OUTPUT_FILE"
echo "  - accelerate>=0.30.0: Accelerate per training distribuito" >> "$OUTPUT_FILE"
echo "  - datasets>=2.16.0: Hugging Face Datasets" >> "$OUTPUT_FILE"
echo "  - einops>=0.7.0: Operazioni tensor eleganti" >> "$OUTPUT_FILE"
echo "  - autoawq>=0.2.0: AutoAWQ per quantizzazione" >> "$OUTPUT_FILE"
echo "  - rank-bm25>=0.2.2: BM25 per baseline RAG" >> "$OUTPUT_FILE"

echo "" >> "$OUTPUT_FILE"
echo "--- END OF FILE latent_compress_llm_completo_$(printf "%03d" $COUNTER).txt ---" >> "$OUTPUT_FILE"

echo "✅ File $OUTPUT_FILE creato con successo."
echo "📊 Processati ${#FILES_TO_CAT[@]} file"
echo "📁 File trovati: $FOUND_FILES"
echo "❌ File mancanti: $MISSING_FILES"
echo "💾 Dimensione totale: $((TOTAL_SIZE / 1024)) KB"
echo ""
echo "🎯 Il file contiene tutto il progetto Latent Compression LLM completo:"
echo "   - Data Module: Dataset loader e generazione sintetica"
echo "   - Models Module: Compressore, Teacher, Reasoner, Baseline"
echo "   - Training Module: Training loop e distillation"
echo "   - Evaluation Module: Valutazione e report"
echo "   - Utils Module: Logging, metrics, tokenization, timers, seed, vram"
echo "   - Configuration: Config YAML per base e esperimenti"
echo "   - Documentation: README principale"
echo "   - Dependencies: requirements.txt"
echo ""
echo "📖 Per eseguire: chmod +x scripts/zz_salva_progetto_latent_compress_llm.sh && ./scripts/zz_salva_progetto_latent_compress_llm.sh"
echo "📂 Output salvato in: $OUTPUT_FILE"



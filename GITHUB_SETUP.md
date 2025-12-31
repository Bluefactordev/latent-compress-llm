# Setup Repository GitHub - latent-compress-llm

## 📝 Repository Description

**Title**: Latent Compression for Long-Context LLM

**Description** (for GitHub):
```
Latent compression system for very long contexts (up to ~262k tokens) using a Reader/Teacher/Reasoner architecture. The system compresses extremely long contexts into short latent sequences through a Perceiver-style compressor, enabling smaller models to reason effectively over very long documents.
```

**Key Features**:
- **Reader/Teacher**: Qwen3 30B MoE AWQ 4bit reads full context (up to 262k tokens)
- **Compressor**: Perceiver-style resampler for latent compression
- **Reasoner**: Qwen3 4B operates only on compressed latents + question
- **Baselines**: Comparison with Teacher full context, RAG, Summary, Pooling

## 🏷️ Tags Consigliati

```
latent-compression
long-context
llm
perceiver
qwen
compression
transformer
deep-learning
pytorch
nlp
knowledge-distillation
teacher-student
context-compression
```

## 📋 Istruzioni per il Primo Commit

### 1. Verifica stato repository

```bash
cd /root/deep-steering/latent-compress-llm
git status
```

### 2. Inizializza repository (se non già fatto)

```bash
git init
```

### 3. Aggiungi file al staging

```bash
# Aggiungi tutti i file (escludendo quelli in .gitignore)
git add .

# Oppure aggiungi selettivamente:
git add README.md
git add requirements.txt
git add configs/
git add src/
git add scripts/
git add docs/
git add IMPLEMENTATION_STATUS.md
git add README_DEV.md
git add VERSION_TAG.txt
git add .gitignore
```

### 4. Crea il primo commit

```bash
git commit -m "Initial commit: Latent Compression for Long-Context LLM

- Reader/Teacher/Reasoner architecture for compressing contexts up to 262k tokens
- Perceiver-style compressor with multi-layer extraction
- Support for Qwen3 30B MoE AWQ (Teacher) and Qwen3 4B (Reasoner)
- Training system with freeze policy and auxiliary loss
- Baseline implementations (RAG, Summary, Pooling)
- YAML configurations for experiments S0 and S1
- Complete documentation (README, architecture docs, experiment plan)
- Version: v2_control_ready"
```

### 5. Aggiungi remote GitHub (dopo aver creato il repo su GitHub)

```bash
# Sostituisci USERNAME con il tuo username GitHub
git remote add origin https://github.com/USERNAME/latent-compress-llm.git

# Oppure con SSH:
git remote add origin git@github.com:USERNAME/latent-compress-llm.git
```

### 6. Push del primo commit

```bash
# Push al branch main (o master se il tuo repo usa master)
git branch -M main
git push -u origin main
```

## 🔄 Comandi Utili Successivi

### Aggiungere tag di versione

```bash
git tag -a v2.0.0 -m "Version v2_control_ready - Phases 1-5 implemented"
git push origin v2.0.0
```

### Verificare cosa verrà committato

```bash
git status
git diff --cached
```

### Escludere file specifici (se necessario)

```bash
# Aggiungi al .gitignore prima di git add
echo "nome_file" >> .gitignore
```

## ⚠️ Note Importanti

1. **File esclusi automaticamente** (via .gitignore):
   - `venv/` e altri ambienti virtuali
   - `data/`, `runs/`, `outputs/`, `logs/`
   - File di checkpoint e modelli grandi (`.pth`, `.pt`, `.ckpt`)
   - Cache e file temporanei

2. **File da includere**:
   - Tutto il codice sorgente in `src/`
   - Configurazioni in `configs/`
   - Script in `scripts/`
   - Documentazione in `docs/`
   - README e file di stato

3. **Dimensione repository**: Assicurati che i file di modello grandi non vengano committati. Usa Git LFS se necessario per file binari.

## 📦 Struttura Repository

```
latent-compress-llm/
├── README.md                    # Documentazione principale
├── README_DEV.md               # Documentazione sviluppatori
├── IMPLEMENTATION_STATUS.md    # Stato implementazione
├── VERSION_TAG.txt             # Versione corrente
├── requirements.txt            # Dipendenze Python
├── .gitignore                  # File da ignorare
├── configs/                    # Configurazioni YAML
├── src/                        # Codice sorgente
├── scripts/                    # Script utility
├── docs/                       # Documentazione tecnica
└── sorgenti/                   # File sorgente originali
```


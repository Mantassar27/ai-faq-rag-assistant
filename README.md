# 🤖 AI FAQ RAG — Assistant e-commerce intelligent

Système de question-réponse basé sur le pattern **RAG (Retrieval-Augmented Generation)** pour répondre automatiquement aux questions FAQ d'un site e-commerce, en s'appuyant uniquement sur la documentation officielle.

---

## 🏗️ Architecture

```
User Question
     │
     ▼
┌─────────────┐
│  FastAPI    │  POST /ask
│   /upload   │  POST /upload
│   /health   │  GET  /health
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Retrieval  │  Embedding de la question
│             │  Recherche top-k chunks (ChromaDB)
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Generation │  Injection contexte + few-shot
│   (Mistral) │  Génération réponse structurée
└──────┬──────┘
       │
       ▼
  JSON Response
  { answer, sources, confidence }
```

**Pipeline complet :**
```
Documents (PDF/TXT/JSON)
  → Ingestion (chargement + nettoyage + chunking)
  → Embedding (paraphrase-multilingual-MiniLM-L12-v2)
  → ChromaDB (stockage vectoriel persistant)
  → Retrieval (similarité cosinus, top-k)
  → LLM Mistral (génération avec prompt engineering)
  → Réponse structurée (answer + sources + confidence)
```

---

## ⚙️ Setup

### Prérequis
- Python 3.10+
- [Ollama](https://ollama.com/download) installé

### Installation en 3 commandes

```bash
pip install -r requirements.txt
ollama pull mistral
uvicorn src.api.main:app --reload
```

### Configuration `.env`
```env
LLM_BACKEND=ollama
EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
CHROMA_DIR=./data/chroma_db
COLLECTION_NAME=faq_chunks
```

### 🐳 Docker

```bash
docker build -t rag-app .
docker run -p 8000:8000 --env-file .env rag-app
```

---

## 🔌 Endpoints API

| Méthode | Endpoint  | Description                        |
|---------|-----------|------------------------------------|
| `POST`  | `/upload` | Uploader un document (PDF/TXT/JSON)|
| `POST`  | `/ask`    | Poser une question                 |
| `GET`   | `/health` | Vérifier l'état de l'API           |
| `GET`   | `/docs`   | Documentation Swagger interactive  |

### Exemple `/upload`
```bash
curl -X POST http://localhost:8000/upload \
  -F "file=@data/raw/livraison.txt"
```
```json
{
  "filename": "livraison.txt",
  "chunks_created": 12,
  "doc_count": 12
}
```

### Exemple `/ask`
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Quels sont les délais de livraison ?"}'
```
```json
{
  "answer": "Réponse : La livraison standard nationale prend entre 2 et 5 jours ouvrables. Pour les zones éloignées, le délai peut aller jusqu'à 7 jours ouvrables.\nSources : [livraison.txt]",
  "sources": ["livraison.txt"],
  "confidence": "high",
  "latency_ms": 3420
}
```

---

## 🛠️ Choix techniques

### FastAPI
- **Performance** : basé sur Starlette/asyncio, l'un des frameworks Python les plus rapides
- **Documentation automatique** : Swagger UI généré sans configuration (`/docs`)
- **Validation** : Pydantic intégré pour valider les requêtes/réponses
- **Moderne** : support natif async, type hints, lifespan events

### ChromaDB
- **Open-source** et embarqué : aucun serveur externe à configurer
- **Persistance sur disque** : les vecteurs survivent aux redémarrages
- **Distance cosinus** : idéale pour la similarité sémantique en NLP
- **API simple** : `upsert`, `query` en quelques lignes

### Modèle d'embedding — `paraphrase-multilingual-MiniLM-L12-v2`
- **Multilingue** : supporte le français nativement (crucial pour ce projet)
- **Léger** : 12 couches, 384 dimensions, tourne sur CPU sans GPU
- **Performant** : bon équilibre vitesse / qualité pour la recherche sémantique

### LLM — Mistral 7B (via Ollama)
- **Gratuit et local** : aucun coût d'API, aucune donnée envoyée à l'extérieur
- **Français** : Mistral est entraîné sur du contenu français, meilleur que LLaMA pour notre cas
- **Qualité** : meilleure instruction-following que des modèles plus anciens
- **Flexible** : remplaçable par GPT-4 via variable d'environnement `LLM_BACKEND=openai`

---

## 📁 Structure du projet

```
ai-faq-rag/
├── README.md
├── reflection.md
├── requirements.txt
├── .env.example
├── Dockerfile
├── ingest_data.py
│
├── data/
│   ├── raw/          ← documents source (PDF, TXT, JSON)
│   ├── processed/
│   └── chroma_db/    ← base vectorielle persistante
│
└── src/
    ├── ingestion/    ← loader, cleaner, chunker
    ├── embedding/    ← embedder, vector_store
    ├── retrieval/    ← retriever (+ MMR reranking)
    ├── generation/   ← generator, prompt engineering
    └── api/          ← FastAPI (main.py)
```

---

## 📊 Résultats d'évaluation

Évaluation sur 10 questions (7 in-scope + 3 hors-scope) basée sur `livraison.txt` :

| Métrique | Score |
|---|---|
| Score global | **0.90** (9/10) |
| Questions in-scope | **1.00** (6/6) ✅ |
| Questions hors-scope | **0.75** (3/4) ✅ |
| Latence moyenne | ~16s (CPU, sans GPU) |
| Évaluation | 🔥 Très bon |

> Évaluation complète disponible dans `eval/results.json`

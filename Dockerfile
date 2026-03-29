# ── Build stage ──────────────────────────────────────────────
FROM python:3.11-slim

# Métadonnées
LABEL maintainer="ShopVite AI Team"
LABEL description="AI FAQ RAG — Assistant e-commerce"

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CHROMA_DIR=/app/data/chroma_db \
    COLLECTION_NAME=faq_chunks \
    LLM_BACKEND=ollama \
    EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2

WORKDIR /app

# Dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Dépendances Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code source
COPY src/ ./src/
COPY data/raw/ ./data/raw/

# Pré-téléchargement du modèle d'embedding au build
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')"

# Port exposé
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s \
  CMD curl -f http://localhost:8000/health || exit 1

# Lancement
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
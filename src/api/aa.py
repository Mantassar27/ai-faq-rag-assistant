"""
main.py — API FastAPI pour le pipeline RAG
"""

from __future__ import annotations
import os
import time
from contextlib import asynccontextmanager
from dotenv import load_dotenv
load_dotenv()  # ← charge le .env AVANT tout import

import shutil
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.embedding import (
    load_model,
    get_chroma_client,
    get_or_create_collection,
)
from src.retrieval import retrieve
from src.generation import generate


# ─────────────────────────────────────────────────────────────
# Schémas Pydantic
# ─────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, example="Quelle est la politique de retour ?")
    k: int        = Field(default=5,  ge=1, le=20, description="Nombre de chunks à récupérer")


class AskResponse(BaseModel):
    answer:     str
    sources:    list[str]
    confidence: str          # "high" | "medium" | "low"
    latency_ms: int


class UploadResponse(BaseModel):
    filename:    str
    chunks_created: int
    doc_count:   int


class HealthResponse(BaseModel):
    status:     str
    model:      str
    collection: str
    doc_count:  int


# ─────────────────────────────────────────────────────────────
# État global (chargé une seule fois au démarrage)
# ─────────────────────────────────────────────────────────────

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle et ChromaDB au démarrage de l'API."""
    print("[api] 🚀 Démarrage — chargement des ressources...")

    model_name       = os.getenv("EMBEDDING_MODEL",   "all-MiniLM-L6-v2")
    chroma_dir       = os.getenv("CHROMA_DIR",        "./data/chroma_db")
    collection_name  = os.getenv("COLLECTION_NAME",   "faq_chunks")
    llm_backend      = os.getenv("LLM_BACKEND",       "ollama")  # défaut = ollama

    print(f"[api] LLM_BACKEND = '{llm_backend}'")
    print(f"[api] EMBEDDING_MODEL = '{model_name}'")

    state["model"]           = load_model(model_name)
    state["model_name"]      = model_name
    state["client"]          = get_chroma_client(chroma_dir)
    state["collection"]      = get_or_create_collection(state["client"], collection_name)
    state["collection_name"] = collection_name
    state["llm_backend"]     = llm_backend

    print("[api] ✅ Prêt !")
    yield
    print("[api] 🛑 Arrêt de l'API")


# ─────────────────────────────────────────────────────────────
# Application FastAPI
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI FAQ RAG API",
    description="Pipeline RAG pour répondre aux questions e-commerce",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health():
    """Vérifie que l'API est opérationnelle."""
    collection = state.get("collection")
    return {
        "status":     "ok",
        "model":      state.get("model_name", "?"),
        "collection": state.get("collection_name", "?"),
        "doc_count":  collection.count() if collection else 0,
    }


@app.post("/upload", response_model=UploadResponse, tags=["Ingestion"])
async def upload(
    file: UploadFile = File(...),
    strategy: str = "paragraphs",
    chunk_size: int = 500,
    overlap: int = 50,
):
    """
    Upload un fichier (PDF / TXT / JSON), l'ingère et le stocke dans ChromaDB.
    Formats acceptés : .txt, .md, .json, .pdf
    """
    ALLOWED = {".txt", ".md", ".json", ".pdf"}
    suffix = Path(file.filename).suffix.lower()

    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Format '{suffix}' non supporté. Acceptés : {ALLOWED}",
        )

    # Sauvegarde temporaire du fichier uploadé
    upload_dir = Path("./data/raw")
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / file.filename

    with dest.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        from src.ingestion import ingest
        from src.embedding import embed_chunks, store_chunks

        # Ingestion : chargement + nettoyage + chunking
        chunks = ingest(
            str(dest),
            strategy=strategy,
            chunk_size=chunk_size,
            overlap=overlap,
        )

        # Embedding + stockage ChromaDB
        embeddings = embed_chunks(chunks, model=state["model"])
        store_chunks(state["collection"], chunks, embeddings)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {
        "filename":       file.filename,
        "chunks_created": len(chunks),
        "doc_count":      state["collection"].count(),
    }


@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask(body: AskRequest):
    """
    Répond à une question en utilisant le pipeline RAG complet :
      1. Retrieval des chunks pertinents
      2. Génération de la réponse via LLM
    """
    if not state.get("collection"):
        raise HTTPException(status_code=503, detail="Vector store non initialisé")

    if state["collection"].count() == 0:
        raise HTTPException(status_code=503, detail="La base de données est vide — ingérez des documents d'abord")

    t0 = time.perf_counter()

    try:
        # 1. Retrieval
        results = retrieve(
            query=body.question,
            collection=state["collection"],
            model=state["model"],
            k=body.k,
            score_threshold=0.30,
        )

        # 2. Génération
        output = generate(
            question=body.question,
            results=results,
            backend=state["llm_backend"],
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = int((time.perf_counter() - t0) * 1000)

    return {
        "answer":     output["answer"],
        "sources":    output["sources"],
        "confidence": output["confidence"],
        "latency_ms": latency_ms,
    }


@app.get("/", tags=["Info"])
def root():
    return {"message": "AI FAQ RAG API — voir /docs pour la documentation interactive"}

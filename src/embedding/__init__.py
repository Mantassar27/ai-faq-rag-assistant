"""
src/embedding/__init__.py
Pipeline complet : chunks → embeddings → ChromaDB
"""

from .embedder import load_model, embed_chunks, embed_query
from .vector_store import (
    get_chroma_client,
    get_or_create_collection,
    store_chunks,
    query_collection,
    delete_collection,
)


def build_vector_store(
    chunks: list[dict],
    model_name: str = "all-MiniLM-L6-v2",
    persist_directory: str = "./data/chroma_db",
    collection_name: str = "faq_chunks",
    batch_size: int = 32,
) -> tuple:
    """
    Pipeline complet en une fonction :
      1. Charge le modèle d'embedding
      2. Encode tous les chunks
      3. Stocke dans ChromaDB

    Args:
        chunks             : sortie de src.ingestion.ingest()
        model_name         : modèle SentenceTransformer
        persist_directory  : dossier de persistence ChromaDB
        collection_name    : nom de la collection
        batch_size         : taille des batchs d'encodage

    Returns:
        (collection, model) — utilisables directement pour la recherche
    """
    model      = load_model(model_name)
    embeddings = embed_chunks(chunks, model=model, batch_size=batch_size)

    client     = get_chroma_client(persist_directory)
    collection = get_or_create_collection(client, collection_name)

    store_chunks(collection, chunks, embeddings)

    return collection, model


__all__ = [
    "load_model",
    "embed_chunks",
    "embed_query",
    "get_chroma_client",
    "get_or_create_collection",
    "store_chunks",
    "query_collection",
    "delete_collection",
    "build_vector_store",
]

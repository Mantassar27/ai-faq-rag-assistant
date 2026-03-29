"""
vector_store.py — Stockage et recherche dans ChromaDB
"""

from __future__ import annotations


def get_chroma_client(persist_directory: str = "./data/chroma_db"):
    """
    Crée ou ouvre un client ChromaDB persistant sur disque.

    Args:
        persist_directory : dossier où ChromaDB sauvegarde les données

    Returns:
        chromadb.Client
    """
    try:
        import chromadb
    except ImportError:
        raise ImportError("Installe ChromaDB : pip install chromadb")

    client = chromadb.PersistentClient(path=persist_directory)
    print(f"[vector_store] ChromaDB ouvert → '{persist_directory}'")
    return client


def get_or_create_collection(client, collection_name: str = "faq_chunks"):
    """
    Récupère ou crée une collection ChromaDB.

    Args:
        client          : client ChromaDB
        collection_name : nom de la collection

    Returns:
        chromadb.Collection
    """
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},  # distance cosinus (meilleure pour NLP)
    )
    print(f"[vector_store] Collection '{collection_name}' — {collection.count()} doc(s) existants")
    return collection


def store_chunks(
    collection,
    chunks: list[dict],
    embeddings: list[list[float]],
) -> None:
    """
    Insère les chunks et leurs vecteurs dans ChromaDB.

    Args:
        collection : collection ChromaDB cible
        chunks     : liste de dicts [{"chunk_id": 0, "source": "...", "text": "..."}, ...]
        embeddings : liste de vecteurs correspondants

    Raises:
        ValueError : si chunks et embeddings n'ont pas la même longueur
    """
    if len(chunks) != len(embeddings):
        raise ValueError(
            f"Mismatch : {len(chunks)} chunks mais {len(embeddings)} embeddings"
        )

    ids        = [f"{c['source']}__chunk_{c['chunk_id']}" for c in chunks]
    documents  = [c["text"] for c in chunks]
    metadatas  = [{"source": c["source"], "chunk_id": c["chunk_id"], "char_count": c.get("char_count", 0)} for c in chunks]

    # ChromaDB gère automatiquement les doublons via les IDs
    collection.upsert(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    print(f"[vector_store] ✅ {len(chunks)} chunks stockés — total collection : {collection.count()}")


def query_collection(
    collection,
    query_embedding: list[float],
    n_results: int = 5,
) -> list[dict]:
    """
    Recherche les chunks les plus proches d'un vecteur requête.

    Args:
        collection      : collection ChromaDB
        query_embedding : vecteur de la requête utilisateur
        n_results       : nombre de résultats à retourner

    Returns:
        Liste de dicts triés par pertinence :
        [{"text": "...", "source": "...", "score": 0.92}, ...]
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "source":   meta.get("source", "?"),
            "chunk_id": meta.get("chunk_id", -1),
            "score":    round(1 - dist, 4),  # distance cosinus → similarité
        })

    return hits


def delete_collection(client, collection_name: str) -> None:
    """Supprime une collection ChromaDB."""
    client.delete_collection(collection_name)
    print(f"[vector_store] 🗑️  Collection '{collection_name}' supprimée")

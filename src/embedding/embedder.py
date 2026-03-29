"""
embedder.py — Conversion des chunks en vecteurs (embeddings)
"""

from __future__ import annotations
from typing import Union


def load_model(model_name: str = "all-MiniLM-L6-v2"):
    """
    Charge un modèle SentenceTransformer.

    Modèles recommandés :
        - "all-MiniLM-L6-v2"        → rapide, léger, bon pour l'anglais
        - "paraphrase-multilingual-MiniLM-L12-v2" → multilingue (FR inclus) ✅
        - "all-mpnet-base-v2"        → plus précis, plus lent

    Args:
        model_name : nom du modèle HuggingFace

    Returns:
        Instance SentenceTransformer
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Installe sentence-transformers : pip install sentence-transformers")

    print(f"[embedder] Chargement du modèle '{model_name}'...")
    model = SentenceTransformer(model_name)
    print(f"[embedder] ✅ Modèle chargé — dimension des vecteurs : {model.get_sentence_embedding_dimension()}")
    return model


def embed_chunks(
    chunks: Union[list[str], list[dict]],
    model=None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    show_progress: bool = True,
) -> list[list[float]]:
    """
    Convertit une liste de chunks en vecteurs.

    Args:
        chunks       : liste de strings OU liste de dicts avec clé "text"
        model        : modèle déjà chargé (optionnel, évite de recharger)
        model_name   : nom du modèle à charger si model=None
        batch_size   : taille des batchs pour l'encodage
        show_progress: affiche une barre de progression

    Returns:
        Liste de vecteurs (list[float])
    """
    if model is None:
        model = load_model(model_name)

    # Accepte soit des strings, soit des dicts {"text": "..."}
    texts = [
        c["text"] if isinstance(c, dict) else c
        for c in chunks
    ]

    print(f"[embedder] Encodage de {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=False,    # retourne des listes Python
    )

    # Convertit en list[list[float]] (compatible JSON / ChromaDB)
    vectors = [e.tolist() for e in embeddings]

    print(f"[embedder] ✅ {len(vectors)} vecteurs générés — dim={len(vectors[0])}")
    return vectors


def embed_query(query: str, model=None, model_name: str = "all-MiniLM-L6-v2") -> list[float]:
    """
    Encode une seule requête utilisateur en vecteur.

    Args:
        query      : question de l'utilisateur
        model      : modèle déjà chargé (optionnel)
        model_name : nom du modèle si model=None

    Returns:
        Vecteur (list[float])
    """
    if model is None:
        model = load_model(model_name)

    vector = model.encode(query, convert_to_numpy=False)
    return vector.tolist()

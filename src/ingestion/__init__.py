"""
src/ingestion/__init__.py
Expose l'API publique du module d'ingestion.
"""

from .loader import load_document, load_folder
from .cleaner import clean_text
from .chunker import chunk_text, chunk_by_sentences, chunk_by_paragraphs, build_chunks_with_metadata


def ingest(
    file_path: str,
    strategy: str = "characters",
    chunk_size: int = 500,
    overlap: int = 50,
    clean: bool = True,
    **loader_kwargs,
) -> list[dict]:
    """
    Pipeline complet d'ingestion en une seule fonction :
      1. Charge le document
      2. Nettoie le texte
      3. Découpe en chunks avec métadonnées

    Args:
        file_path  : chemin vers le fichier (PDF / TXT / JSON)
        strategy   : "characters" | "sentences" | "paragraphs"
        chunk_size : taille des chunks (si strategy="characters")
        overlap    : chevauchement (si strategy="characters")
        clean      : appliquer le nettoyage ?
        **loader_kwargs : options pour le loader (ex. text_field pour JSON)

    Returns:
        Liste de chunks avec métadonnées
    """
    from pathlib import Path

    raw_text = load_document(file_path, **loader_kwargs)

    if clean:
        raw_text = clean_text(raw_text)

    source = Path(file_path).name

    # chunk_size et overlap ne s'appliquent qu'à la stratégie "characters"
    extra = {}
    if strategy == "characters":
        extra = {"chunk_size": chunk_size, "overlap": overlap}

    return build_chunks_with_metadata(
        text=raw_text,
        source=source,
        strategy=strategy,
        **extra,
    )


__all__ = [
    "load_document",
    "load_folder",
    "clean_text",
    "chunk_text",
    "chunk_by_sentences",
    "chunk_by_paragraphs",
    "build_chunks_with_metadata",
    "ingest",
]
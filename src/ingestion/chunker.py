"""
chunker.py — Découpage du texte en chunks pour le pipeline RAG
"""

from __future__ import annotations


# ─────────────────────────────────────────────
# 1. Chunking par caractères (méthode de base)
# ─────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """
    Découpe un texte en chunks de taille fixe avec chevauchement.

    Args:
        text       : texte à découper
        chunk_size : nombre de caractères par chunk
        overlap    : chevauchement entre deux chunks consécutifs

    Returns:
        Liste de strings (chunks)
    """
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        start += chunk_size - overlap  # recul de `overlap` caractères

    # Supprime les chunks vides ou trop courts (< 10 chars)
    return [c for c in chunks if len(c) >= 10]


# ─────────────────────────────────────────────
# 2. Chunking par phrases (plus naturel)
# ─────────────────────────────────────────────

def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[str]:
    """
    Découpe le texte en groupes de phrases.

    Args:
        text                : texte à découper
        sentences_per_chunk : nombre de phrases par chunk
        overlap_sentences   : nombre de phrases partagées entre chunks

    Returns:
        Liste de chunks (chaque chunk = plusieurs phrases)
    """
    import re

    # Séparation sur . ! ? suivis d'un espace ou fin de ligne
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in sentences if s.strip()]

    if not sentences:
        return []

    chunks = []
    step = sentences_per_chunk - overlap_sentences
    step = max(step, 1)  # évite step=0

    for i in range(0, len(sentences), step):
        chunk = " ".join(sentences[i : i + sentences_per_chunk])
        if chunk:
            chunks.append(chunk)

    return chunks


# ─────────────────────────────────────────────
# 3. Chunking par paragraphes
# ─────────────────────────────────────────────

def chunk_by_paragraphs(text: str, max_chars: int = 1000) -> list[str]:
    """
    Découpe le texte en respectant les paragraphes.
    Si un paragraphe dépasse max_chars, il est re-découpé par chunk_text.

    Args:
        text      : texte à découper
        max_chars : taille max d'un chunk en caractères

    Returns:
        Liste de chunks
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks = []

    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
        else:
            # Paragraphe trop long → re-découpage
            sub_chunks = chunk_text(para, chunk_size=max_chars, overlap=50)
            chunks.extend(sub_chunks)

    return chunks


# ─────────────────────────────────────────────
# 4. Ajout des métadonnées sur les chunks
# ─────────────────────────────────────────────

def build_chunks_with_metadata(
    text: str,
    source: str = "unknown",
    strategy: str = "characters",
    **kwargs,
) -> list[dict]:
    """
    Découpe un texte et retourne des chunks enrichis de métadonnées.

    Args:
        text     : texte nettoyé à découper
        source   : nom du fichier source
        strategy : "characters" | "sentences" | "paragraphs"
        **kwargs : paramètres passés à la fonction de découpage

    Returns:
        Liste de dicts :
        [
          {
            "chunk_id": 0,
            "source": "doc.pdf",
            "text": "...",
            "char_count": 487,
          },
          ...
        ]
    """
    strategies = {
        "characters":  chunk_text,
        "sentences":   chunk_by_sentences,
        "paragraphs":  chunk_by_paragraphs,
    }

    if strategy not in strategies:
        raise ValueError(f"Stratégie inconnue : '{strategy}'. Choisir parmi {list(strategies)}")

    raw_chunks = strategies[strategy](text, **kwargs)

    return [
        {
            "chunk_id":   i,
            "source":     source,
            "text":       chunk,
            "char_count": len(chunk),
        }
        for i, chunk in enumerate(raw_chunks)
    ]

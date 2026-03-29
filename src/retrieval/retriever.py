"""
retriever.py — Recherche des chunks les plus pertinents à une question
"""

from __future__ import annotations


def retrieve(
    query: str,
    collection,
    model,
    k: int = 5,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Pipeline de retrieval complet :
      1. Encode la question en vecteur
      2. Recherche les k chunks les plus proches dans ChromaDB
      3. Filtre par score minimal (optionnel)

    Args:
        query           : question de l'utilisateur
        collection      : collection ChromaDB
        model           : modèle SentenceTransformer déjà chargé
        k               : nombre de chunks à retourner
        score_threshold : score de similarité minimum (0.0 = pas de filtre)

    Returns:
        Liste de dicts triés par score décroissant :
        [{"text": "...", "source": "...", "chunk_id": 0, "score": 0.92}, ...]
    """
    from src.embedding import embed_query, query_collection

    # 1. Encoder la question
    query_vector = embed_query(query, model=model)

    # 2. Recherche vectorielle
    results = query_collection(collection, query_vector, n_results=k)

    # 3. Filtrage par score
    if score_threshold > 0.0:
        results = [r for r in results if r["score"] >= score_threshold]

    print(f"[retriever] '{query[:60]}...' → {len(results)} chunk(s) trouvé(s)")
    return results


def retrieve_with_rerank(
    query: str,
    collection,
    model,
    k: int = 5,
    fetch_k: int = 20,
    score_threshold: float = 0.0,
) -> list[dict]:
    """
    Retrieval avec re-ranking par Maximum Marginal Relevance (MMR).
    Récupère fetch_k candidats puis sélectionne k résultats
    diversifiés pour éviter les chunks redondants.

    Args:
        fetch_k : nombre de candidats récupérés avant re-ranking
        k       : nombre final de chunks retenus

    Returns:
        Liste de k chunks diversifiés et pertinents
    """
    from src.embedding import embed_query, query_collection
    import numpy as np

    query_vector = embed_query(query, model=model)
    candidates   = query_collection(collection, query_vector, n_results=fetch_k)

    if score_threshold > 0.0:
        candidates = [c for c in candidates if c["score"] >= score_threshold]

    if len(candidates) <= k:
        return candidates

    # ── MMR : sélection itérative maximisant pertinence - redondance ──
    selected   = []
    remaining  = candidates.copy()

    # Vecteurs des candidats (recalculés pour MMR)
    texts = [c["text"] for c in candidates]
    vecs  = model.encode(texts, convert_to_numpy=True)
    q_vec = model.encode(query, convert_to_numpy=True)

    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    lambda_mmr = 0.5  # équilibre pertinence / diversité

    for _ in range(k):
        if not remaining:
            break

        best_score = -float("inf")
        best_idx   = 0

        for i, cand in enumerate(remaining):
            idx        = candidates.index(cand)
            relevance  = cosine(vecs[idx], q_vec)
            redundancy = max(
                (cosine(vecs[idx], vecs[candidates.index(s)]) for s in selected),
                default=0.0,
            )
            mmr_score  = lambda_mmr * relevance - (1 - lambda_mmr) * redundancy

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx   = i

        selected.append(remaining.pop(best_idx))

    print(f"[retriever] MMR → {len(selected)} chunk(s) diversifiés sélectionnés")
    return selected


def format_context(results: list[dict], max_chars: int = 3000) -> str:
    """
    Formate les chunks récupérés en un bloc de contexte pour le LLM.

    Args:
        results   : sortie de retrieve()
        max_chars : limite de caractères totaux du contexte

    Returns:
        String formaté prêt à être injecté dans le prompt
    """
    context_parts = []
    total = 0

    for r in results:
        block = f"[Source: {r['source']}]\n{r['text']}"
        if total + len(block) > max_chars:
            break
        context_parts.append(block)
        total += len(block)

    return "\n\n---\n\n".join(context_parts)

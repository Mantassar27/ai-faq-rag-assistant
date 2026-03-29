"""
test_embedding.py — Démo du pipeline embedding + ChromaDB
Lancer depuis la racine : python -m src.embedding.test_embedding
"""

from src.embedding import build_vector_store, embed_query, query_collection

# ── Chunks simulés (normalement issus de src.ingestion.ingest) ───────────────
sample_chunks = [
    {"chunk_id": 0, "source": "faq.txt", "text": "Le RAG combine recherche et génération de texte.", "char_count": 50},
    {"chunk_id": 1, "source": "faq.txt", "text": "ChromaDB est une base de données vectorielle open-source.", "char_count": 55},
    {"chunk_id": 2, "source": "faq.txt", "text": "Les embeddings transforment le texte en vecteurs numériques.", "char_count": 58},
    {"chunk_id": 3, "source": "faq.txt", "text": "SentenceTransformers encode des phrases en vecteurs denses.", "char_count": 57},
    {"chunk_id": 4, "source": "faq.txt", "text": "La distance cosinus mesure la similarité entre deux vecteurs.", "char_count": 60},
]

print("=" * 60)
print("TEST 1 — build_vector_store()")
print("=" * 60)

collection, model = build_vector_store(
    chunks=sample_chunks,
    model_name="all-MiniLM-L6-v2",
    persist_directory="./data/chroma_db",
    collection_name="faq_test",
)

print("\n" + "=" * 60)
print("TEST 2 — Recherche sémantique")
print("=" * 60)

query = "Comment stocker des vecteurs ?"
print(f"  Requête : '{query}'")

q_vector = embed_query(query, model=model)
results  = query_collection(collection, q_vector, n_results=3)

for r in results:
    print(f"\n  📄 [{r['source']}] score={r['score']}")
    print(f"     {r['text']}")

print("\n✅ Tests embedding terminés avec succès.")

"""
test_ingestion.py — Démo rapide du module d'ingestion
Lancer depuis la racine du projet : python -m src.ingestion.test_ingestion
"""

from src.ingestion import ingest, clean_text, chunk_text

# ── Test 1 : chunk_text basique ──────────────────────────────────────────────
sample = (
    "Le RAG (Retrieval-Augmented Generation) combine la recherche d'information "
    "et la génération de texte. Il permet à un LLM de s'appuyer sur des documents "
    "externes pour répondre à des questions précises sans avoir à tout mémoriser "
    "pendant l'entraînement. C'est une approche très utilisée pour les chatbots FAQ."
)

print("=" * 60)
print("TEST 1 — chunk_text(chunk_size=100, overlap=20)")
print("=" * 60)
chunks = chunk_text(sample, chunk_size=100, overlap=20)
for i, c in enumerate(chunks):
    print(f"  Chunk {i} ({len(c)} chars) : {c[:80]}...")

# ── Test 2 : clean_text ──────────────────────────────────────────────────────
dirty = "Visitez   notre site :  https://example.com   !!!\n\n\n\nContactez  info@mail.com"
print("\n" + "=" * 60)
print("TEST 2 — clean_text")
print("=" * 60)
print("  Avant :", dirty)
print("  Après :", clean_text(dirty, remove_urls_flag=True, remove_emails_flag=True))

# ── Test 3 : pipeline ingest() sur un fichier TXT ───────────────────────────
import tempfile, os

print("\n" + "=" * 60)
print("TEST 3 — ingest() sur un fichier TXT temporaire")
print("=" * 60)

content = "\n\n".join([
    "La première étape d'un pipeline RAG est l'ingestion des documents.",
    "Ensuite, les chunks sont transformés en vecteurs (embeddings).",
    "Le retrieval récupère les chunks les plus pertinents à la question.",
    "Enfin, le LLM génère une réponse à partir de ces chunks.",
])

with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False, encoding="utf-8") as f:
    f.write(content)
    tmp_path = f.name

chunks = ingest(tmp_path, strategy="paragraphs")
for chunk in chunks:
    print(f"  [{chunk['chunk_id']}] source={chunk['source']} | {chunk['char_count']} chars")
    print(f"       {chunk['text'][:100]}")

os.unlink(tmp_path)
print("\n✅ Tests terminés avec succès.")

from dotenv import load_dotenv
load_dotenv()

from src.ingestion import ingest
from src.embedding import build_vector_store

# 1. Charger et découper le document
chunks = ingest("data/raw/faq.txt", strategy="paragraphs")
print(f"✅ {len(chunks)} chunks créés")

# 2. Embedder et stocker dans ChromaDB
collection, model = build_vector_store(
    chunks=chunks,
    persist_directory="./data/chroma_db",
    collection_name="faq_chunks",
)
print(f"✅ Base ChromaDB prête — {collection.count()} documents")
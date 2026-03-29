"""
ragas_ollama_eval.py — Évaluation RAGAS v0.4.3 avec Ollama
Lancer : python eval/ragas_ollama_eval.py
"""

import requests
import json
import datetime
from datasets import Dataset
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_ollama import ChatOllama, OllamaEmbeddings

API_URL = "http://localhost:8000/ask"

# ─────────────────────────────────────────────────────────────
# 1. Questions de test
# ─────────────────────────────────────────────────────────────

questions = [
    {
        "question":     "Quels sont les délais de livraison standard ?",
        "ground_truth": "La livraison standard nationale prend 2 à 5 jours ouvrables.",
    },
    {
        "question":     "La livraison internationale est-elle disponible ?",
        "ground_truth": "Oui, ShopVite propose la livraison internationale avec des délais de 5 à 15 jours ouvrables.",
    },
    {
        "question":     "Comment suivre ma commande ?",
        "ground_truth": "Un email de confirmation avec un numéro de suivi est envoyé après expédition.",
    },
]

# ─────────────────────────────────────────────────────────────
# 2. Appel API
# ─────────────────────────────────────────────────────────────

print("=" * 65)
print("📡 Collecte des réponses via l'API...")
print("=" * 65)

data = []

for q in questions:
    print(f"\n  ❓ {q['question']}")
    try:
        resp = requests.post(API_URL, json={"question": q["question"]}, timeout=120).json()

        answer   = resp.get("answer", "")
        contexts = resp.get("contexts", [])

        if not contexts:
            contexts = [answer] if answer else ["Aucun contexte disponible"]
            print(f"  ⚠️  Fallback contexts utilisé")

        print(f"  ✅ OK — {len(contexts)} contexts")

        data.append({
            "question":     q["question"],
            "answer":       answer,
            "contexts":     contexts,
            "ground_truth": q["ground_truth"],
        })

    except Exception as e:
        print(f"  ❌ Erreur : {e}")

if not data:
    print("❌ Aucune donnée — vérifiez que l'API tourne sur localhost:8000")
    exit(1)

# ─────────────────────────────────────────────────────────────
# 3. Configuration LLM + Embeddings Ollama
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("⚙️  Configuration RAGAS v0.4.3 + Ollama...")
print("=" * 65)

llm = LangchainLLMWrapper(
    ChatOllama(model="mistral", temperature=0, base_url="http://localhost:11434", timeout=120)
)

emb = LangchainEmbeddingsWrapper(
    OllamaEmbeddings(model="mistral", base_url="http://localhost:11434")
)

faithfulness_metric = Faithfulness(llm=llm)
relevancy_metric    = AnswerRelevancy(llm=llm, embeddings=emb)

print("  ✅ LLM        : Mistral via Ollama")
print("  ✅ Embeddings : Mistral via Ollama")
print("  ✅ Métriques  : Faithfulness, AnswerRelevancy")

# ─────────────────────────────────────────────────────────────
# 4. Évaluation
# ─────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("📊 Évaluation RAGAS (2-5 min)...")
print("=" * 65)

dataset = Dataset.from_list(data)

result = evaluate(
    dataset=dataset,
    metrics=[faithfulness_metric, relevancy_metric],
    raise_exceptions=False,
)

# ─────────────────────────────────────────────────────────────
# 5. Résultats
# ─────────────────────────────────────────────────────────────

def safe_score(val):
    if isinstance(val, list):
        valid = [v for v in val if v is not None and str(v) != "nan"]
        return round(sum(valid) / len(valid), 4) if valid else 0.0
    try:
        return round(float(val), 4)
    except:
        return 0.0

faith = safe_score(result["faithfulness"])
relev = safe_score(result["answer_relevancy"])
score = round((faith + relev) / 2, 4)

print("\n" + "=" * 65)
print("🏆 RÉSULTATS RAGAS v0.4.3")
print("=" * 65)
print(f"  🔒 faithfulness     (anti-hallucination) : {faith:.2f}  {'🟢' if faith >= 0.8 else '🟡' if faith >= 0.6 else '🔴'}")
print(f"  🎯 answer_relevancy (pertinence réponse) : {relev:.2f}  {'🟢' if relev >= 0.8 else '🟡' if relev >= 0.6 else '🔴'}")
print(f"  {'─' * 50}")
print(f"  ⭐ Score global                          : {score:.2f}  {'🟢 Excellent' if score >= 0.8 else '🟡 Correct' if score >= 0.6 else '🔴 À améliorer'}")

output = {
    "date":             datetime.datetime.now().isoformat(),
    "ragas_version":    "0.4.3",
    "llm":              "mistral (ollama)",
    "nb_questions":     len(data),
    "faithfulness":     faith,
    "answer_relevancy": relev,
    "global_score":     score,
}

with open("eval/results_ragas.json", "w", encoding="utf-8") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

print(f"\n💾 Résultats sauvegardés dans eval/results_ragas.json")
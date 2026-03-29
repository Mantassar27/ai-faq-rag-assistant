"""
eval.py — Évaluation du pipeline RAG
Lance depuis le dossier eval/ : python eval.py
"""

import json
import time
import requests
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8000/ask"


def ask_api(question: str) -> dict:
    """Envoie une question à l'API et retourne la réponse."""
    try:
        response = requests.post(
            API_URL,
            json={"question": question},
            timeout=60,
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        print(f"  ❌ Impossible de contacter l'API sur {API_URL}")
        print("     → Lance d'abord : uvicorn src.api.main:app --reload")
        exit(1)
    except Exception as e:
        return {"answer": f"ERREUR: {e}", "sources": [], "confidence": "low"}


def compute_score(ground_truth: str, answer: str) -> int:
    """
    Score simple : 1 si ground_truth trouvé dans la réponse, 0 sinon.
    Case-insensitive.
    """
    return 1 if ground_truth.lower() in answer.lower() else 0


def evaluate():
    """Pipeline d'évaluation complet."""

    questions_path = Path(__file__).parent / "questions.json"
    with open(questions_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 60)
    print("🧪 ÉVALUATION DU PIPELINE RAG")
    print(f"   API        : {API_URL}")
    print(f"   Questions  : {len(data)}")
    print(f"   Date       : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    results      = []
    total_score  = 0
    total_time   = 0

    for i, item in enumerate(data, 1):
        question     = item["question"]
        ground_truth = item["ground_truth"]

        print(f"\n[{i}/{len(data)}] {question}")

        t0       = time.time()
        response = ask_api(question)
        elapsed  = round((time.time() - t0) * 1000)

        answer     = response.get("answer", "")
        sources    = response.get("sources", [])
        confidence = response.get("confidence", "?")
        score      = compute_score(ground_truth, answer)

        total_score += score
        total_time  += elapsed

        # Affichage
        status = "✅" if score == 1 else "❌"
        print(f"  {status} Score      : {score}")
        print(f"  💬 Réponse    : {answer[:120]}{'...' if len(answer) > 120 else ''}")
        print(f"  📄 Sources    : {sources}")
        print(f"  🎯 Confidence : {confidence}")
        print(f"  ⏱️  Latence    : {elapsed}ms")

        results.append({
            "question":     question,
            "ground_truth": ground_truth,
            "answer":       answer,
            "sources":      sources,
            "confidence":   confidence,
            "score":        score,
            "latency_ms":   elapsed,
        })

    # ── Rapport final ──────────────────────────────────────────
    avg_score   = total_score / len(data)
    avg_latency = total_time  / len(data)

    # Scores par catégorie
    in_scope_items  = [r for r in results if r["ground_truth"] != "Je n'ai pas d'information"]
    out_scope_items = [r for r in results if r["ground_truth"] == "Je n'ai pas d'information"]

    in_scope_score  = sum(r["score"] for r in in_scope_items)  / len(in_scope_items)  if in_scope_items  else 0
    out_scope_score = sum(r["score"] for r in out_scope_items) / len(out_scope_items) if out_scope_items else 0

    print("\n" + "=" * 60)
    print("📊 RAPPORT FINAL")
    print("=" * 60)
    print(f"  Score global      : {avg_score:.2f} ({total_score}/{len(data)})")
    print(f"  Questions in-scope  : {in_scope_score:.2f} ({sum(r['score'] for r in in_scope_items)}/{len(in_scope_items)})")
    print(f"  Questions hors-scope: {out_scope_score:.2f} ({sum(r['score'] for r in out_scope_items)}/{len(out_scope_items)})")
    print(f"  Latence moyenne   : {avg_latency:.0f}ms")

    # Évaluation qualitative
    if avg_score >= 0.8:
        grade = "🔥 Très bon"
    elif avg_score >= 0.7:
        grade = "✅ Bon"
    elif avg_score >= 0.5:
        grade = "⚠️  Moyen"
    else:
        grade = "❌ À améliorer"

    print(f"  Évaluation        : {grade}")
    print("=" * 60)

    # Sauvegarde résultats
    output = {
        "meta": {
            "date":          datetime.now().isoformat(),
            "api_url":       API_URL,
            "total_questions": len(data),
        },
        "scores": {
            "global":      round(avg_score, 2),
            "in_scope":    round(in_scope_score, 2),
            "out_scope":   round(out_scope_score, 2),
            "avg_latency_ms": round(avg_latency),
        },
        "results": results,
    }

    output_path = Path(__file__).parent / "results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Résultats sauvegardés → eval/results.json")


if __name__ == "__main__":
    evaluate()
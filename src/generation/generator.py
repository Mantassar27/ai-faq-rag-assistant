"""
generator.py — Génération de réponse via LLM (OpenAI ou Ollama local)
"""

from __future__ import annotations
from .prompt import SYSTEM_PROMPT, build_prompt, build_confidence


# ─────────────────────────────────────────────────────────────
# Backend OpenAI (GPT-3.5 / GPT-4)
# ─────────────────────────────────────────────────────────────

def generate_openai(
    question: str,
    context: str,
    model: str = "gpt-3.5-turbo",
    temperature: float = 0.2,
    max_tokens: int = 512,
) -> str:
    """
    Génère une réponse via l'API OpenAI.
    Nécessite : pip install openai + variable OPENAI_API_KEY dans .env

    Args:
        question    : question de l'utilisateur
        context     : contexte formaté (chunks récupérés)
        model       : modèle OpenAI à utiliser
        temperature : créativité (0 = déterministe, 1 = créatif)
        max_tokens  : longueur max de la réponse

    Returns:
        Réponse générée sous forme de string
    """
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("Installe openai : pip install openai")

    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_prompt(question, context)},
        ],
    )

    return response.choices[0].message.content.strip()


# ─────────────────────────────────────────────────────────────
# Backend Ollama (LLM local, gratuit)
# ─────────────────────────────────────────────────────────────

def generate_ollama(
    question: str,
    context: str,
    model: str = "mistral",
    temperature: float = 0.2,
) -> str:
    """
    Génère une réponse via Ollama (LLM local).
    Nécessite : ollama installé + `ollama pull mistral`

    Args:
        model : ex. "mistral", "llama3", "phi3"

    Returns:
        Réponse générée
    """
    try:
        import ollama
    except ImportError:
        raise ImportError("Installe ollama : pip install ollama")

    prompt = build_prompt(question, context)

    response = ollama.chat(
        model=model,
        options={"temperature": temperature},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt},
        ],
    )

    return response["message"]["content"].strip()


# ─────────────────────────────────────────────────────────────
# Point d'entrée unifié
# ─────────────────────────────────────────────────────────────

def generate(
    question: str,
    results: list[dict],
    backend: str = "openai",
    **kwargs,
) -> dict:
    """
    Pipeline de génération complet.

    Args:
        question : question de l'utilisateur
        results  : chunks récupérés (sortie de retrieve())
        backend  : "openai" | "ollama"
        **kwargs : options passées au backend (model, temperature...)

    Returns:
        Dict :
        {
          "answer"    : "La politique de retour est...",
          "sources"   : ["faq.pdf", "conditions.txt"],
          "confidence": "high"
        }
    """
    from src.retrieval import format_context

    # Construire le contexte
    context = format_context(results)

    # Appel LLM
    backends = {
        "openai": generate_openai,
        "ollama": generate_ollama,
    }

    if backend not in backends:
        raise ValueError(f"Backend inconnu : '{backend}'. Choisir parmi {list(backends)}")

    answer = backends[backend](question, context, **kwargs)

    # Dédupliquer les sources
    sources = list(dict.fromkeys(r["source"] for r in results))

    return {
        "answer":     answer,
        "sources":    sources,
        "confidence": build_confidence(results),
    }
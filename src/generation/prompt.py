"""
prompt.py — Templates de prompts pour le pipeline RAG
"""

# ─────────────────────────────────────────────────────────────
# Prompt principal (e-commerce FAQ)
# ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """Tu es un assistant e-commerce professionnel, précis et fiable pour ShopVite.

RÈGLES STRICTES — à respecter absolument :

1. Réponds UNIQUEMENT avec les informations présentes dans le contexte fourni.
2. Tu n'as PAS le droit d'utiliser tes connaissances personnelles ou générales.
3. Tu ne dois JAMAIS inventer, supposer ou extrapoler une réponse.
4. Si l'information est absente du contexte, réponds exactement :
   "Je n'ai pas d'information sur ce sujet dans notre documentation."
5. Cite toujours les sources sous forme de liste.
6. Réponds en français, de manière claire, concise et professionnelle.
7. Ne répète pas la question dans ta réponse.

FORMAT DE RÉPONSE OBLIGATOIRE (toujours respecter cette structure) :

Réponse : <ta réponse basée uniquement sur le contexte>
Sources : <liste des fichiers sources utilisés>

Toute réponse ne respectant pas ce format est incorrecte."""


def build_prompt(question: str, context: str) -> str:
    """
    Construit le prompt utilisateur avec few-shot examples,
    contexte injecté et question.

    Args:
        question : question de l'utilisateur
        context  : chunks formatés issus du retrieval

    Returns:
        Prompt complet prêt à envoyer au LLM
    """
    return f"""Voici des extraits de notre documentation officielle :

{context}

---

EXEMPLES (few-shot) :

Exemple 1 — question dans le contexte :
Question : Quels sont les délais de livraison standard ?
Contexte : [Source: livraison.txt] Livraison standard nationale : 2 à 5 jours ouvrables.
Réponse : La livraison standard nationale prend entre 2 et 5 jours ouvrables.
Sources : [livraison.txt]

Exemple 2 — question hors contexte :
Question : Quels sont vos horaires d'ouverture ?
Contexte : (aucune information disponible sur ce sujet)
Réponse : Je n'ai pas d'information sur ce sujet dans notre documentation.
Sources : []

Exemple 3 — question partielle :
Question : La livraison est-elle gratuite ?
Contexte : [Source: livraison.txt] Livraison gratuite pour toute commande dépassant un certain montant. Le montant exact est indiqué lors du passage en caisse.
Réponse : La livraison est gratuite à partir d'un certain montant de commande. Ce montant est précisé lors du passage en caisse.
Sources : [livraison.txt]

---

Question du client : {question}

Réponds en respectant STRICTEMENT les règles et le format imposé.
"""


def build_confidence(results: list[dict]) -> str:
    """
    Estime le niveau de confiance selon le score du meilleur chunk.

    Args:
        results : liste de chunks avec scores

    Returns:
        "high" | "medium" | "low"
    """
    if not results:
        return "low"

    best_score = results[0]["score"]

    if best_score >= 0.80:
        return "high"
    elif best_score >= 0.55:
        return "medium"
    else:
        return "low"
# 🧠 Réflexion technique — AI FAQ RAG

## Pourquoi ce prompt ?

Le prompt est conçu en trois couches complémentaires :

1. **SYSTEM_PROMPT avec règles numérotées** : Les LLMs suivent mieux des règles explicitement numérotées que des bullet points. L'interdiction explicite d'utiliser les "connaissances personnelles" réduit drastiquement les hallucinations.

2. **Format imposé (`Réponse : / Sources :`)** : Forcer une structure de sortie garantit la cohérence des réponses et facilite le parsing côté API. Sans cette contrainte, Mistral reformate librement ses réponses.

3. **Few-shot examples (3 cas)** : Couvrir les cas "information présente", "information absente" et "information partielle" permet au modèle de comprendre exactement le comportement attendu dans chaque situation, sans ambiguïté.

## Limites actuelles

- **Latence élevée** (~3–8s) : Mistral 7B en local sur CPU est lent — un GPU ou un modèle quantisé (Q4) réduirait la latence de 5×
- **Chunking fixe** : Le découpage par paragraphes peut couper des informations liées (ex: un titre séparé de son contenu)
- **Pas de mémoire conversationnelle** : Chaque question est indépendante — le contexte des échanges précédents est perdu
- **Modèle d'embedding anglophone de base** : Avant le passage au modèle multilingue, la recherche sémantique en français était peu fiable
- **Pas de reranking par défaut** : Le MMR est implémenté mais non activé par défaut dans l'API

## Améliorations possibles

- Activer le **reranking MMR** pour diversifier les chunks récupérés
- Ajouter une **mémoire de conversation** (historique des échanges)
- Implémenter un **score de confiance plus précis** (ex: basé sur la distance moyenne des k chunks)
- Ajouter un **cache Redis** pour les questions fréquentes
- Passer à **Mistral quantisé (GGUF Q4)** pour réduire la latence sur CPU
- Supporter les **images dans les PDF** via OCR (Tesseract)
"""
loader.py — Chargement de documents PDF, TXT et JSON
"""

import json
import re
from pathlib import Path


def load_txt(file_path: str) -> str:
    """Charge un fichier texte brut."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def load_json(file_path: str, text_field: str = "text") -> str:
    """
    Charge un fichier JSON.
    Supporte :
      - une liste d'objets  [{"text": "..."}, ...]
      - un objet unique     {"text": "..."}
      - une liste de strings ["...", "..."]
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        parts = []
        for item in data:
            if isinstance(item, dict):
                parts.append(item.get(text_field, ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n\n".join(parts)

    if isinstance(data, dict):
        return data.get(text_field, "")

    return str(data)


def load_pdf(file_path: str) -> str:
    """
    Charge un fichier PDF page par page.
    Nécessite : pip install pypdf2
    """
    try:
        import PyPDF2
    except ImportError:
        raise ImportError("Installe PyPDF2 : pip install PyPDF2")

    text_pages = []
    with open(file_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text_pages.append(page.extract_text() or "")

    return "\n\n".join(text_pages)


def load_document(file_path: str, **kwargs) -> str:
    """
    Point d'entrée unique — détecte l'extension et délègue au bon loader.

    Args:
        file_path : chemin vers le fichier
        **kwargs  : options supplémentaires (ex. text_field pour JSON)

    Returns:
        Contenu brut du document sous forme de chaîne
    """
    path = Path(file_path)
    ext = path.suffix.lower()

    loaders = {
        ".txt": load_txt,
        ".md":  load_txt,
        ".json": lambda p: load_json(p, **kwargs),
        ".pdf": load_pdf,
    }

    if ext not in loaders:
        raise ValueError(f"Format non supporté : '{ext}'. Formats acceptés : {list(loaders)}")

    print(f"[loader] Chargement de {path.name} ({ext})")
    return loaders[ext](file_path)


def load_folder(folder_path: str, extensions: list[str] = None, **kwargs) -> list[dict]:
    """
    Charge tous les documents d'un dossier.

    Returns:
        Liste de dicts : [{"source": "fichier.pdf", "content": "..."}, ...]
    """
    extensions = extensions or [".txt", ".md", ".json", ".pdf"]
    folder = Path(folder_path)
    documents = []

    for file in sorted(folder.iterdir()):
        if file.suffix.lower() in extensions:
            try:
                content = load_document(str(file), **kwargs)
                documents.append({"source": file.name, "content": content})
            except Exception as e:
                print(f"[loader] ⚠️  Erreur sur {file.name} : {e}")

    print(f"[loader] {len(documents)} document(s) chargé(s) depuis '{folder_path}'")
    return documents

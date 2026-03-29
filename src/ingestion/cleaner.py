"""
cleaner.py — Nettoyage du texte brut avant découpage
"""

import re
import unicodedata


def remove_extra_whitespace(text: str) -> str:
    """Supprime les espaces multiples et les lignes vides en excès."""
    text = re.sub(r"[ \t]+", " ", text)           # espaces/tabs multiples → un espace
    text = re.sub(r"\n{3,}", "\n\n", text)         # > 2 sauts de ligne → 2
    return text.strip()


def remove_special_characters(text: str, keep_punctuation: bool = True) -> str:
    """
    Supprime les caractères spéciaux non utiles.
    Si keep_punctuation=False, retire aussi la ponctuation.
    """
    if keep_punctuation:
        # Garde lettres, chiffres, ponctuation courante et espaces
        text = re.sub(r"[^\w\s.,;:!?\"'()\-\n]", " ", text)
    else:
        text = re.sub(r"[^\w\s\n]", " ", text)
    return text


def normalize_unicode(text: str) -> str:
    """Normalise les caractères unicode (ex: accents composés → précomposés)."""
    return unicodedata.normalize("NFC", text)


def remove_urls(text: str) -> str:
    """Supprime les URLs."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_emails(text: str) -> str:
    """Supprime les adresses e-mail."""
    return re.sub(r"\S+@\S+\.\S+", "", text)


def fix_encoding_artifacts(text: str) -> str:
    """Corrige les artefacts courants d'encodage PDF (ex: ligatures manquantes)."""
    replacements = {
        "\ufb01": "fi",   # ﬁ ligature
        "\ufb02": "fl",   # ﬂ ligature
        "\u2019": "'",    # apostrophe typographique
        "\u2018": "'",
        "\u201c": '"',    # guillemets typographiques
        "\u201d": '"',
        "\u2013": "-",    # tiret demi-cadratin
        "\u2014": "-",    # tiret cadratin
        "\u00a0": " ",    # espace insécable
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    return text


def clean_text(
    text: str,
    remove_urls_flag: bool = True,
    remove_emails_flag: bool = False,
    keep_punctuation: bool = True,
) -> str:
    """
    Pipeline de nettoyage complet.

    Args:
        text              : texte brut à nettoyer
        remove_urls_flag  : supprimer les URLs ?
        remove_emails_flag: supprimer les e-mails ?
        keep_punctuation  : conserver la ponctuation ?

    Returns:
        Texte nettoyé
    """
    text = fix_encoding_artifacts(text)
    text = normalize_unicode(text)

    if remove_urls_flag:
        text = remove_urls(text)
    if remove_emails_flag:
        text = remove_emails(text)

    text = remove_special_characters(text, keep_punctuation=keep_punctuation)
    text = remove_extra_whitespace(text)

    return text

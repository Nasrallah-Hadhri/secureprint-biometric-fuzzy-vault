"""
Module 5: Chiffrement des templates avec Fernet (AES-128-CBC + HMAC-SHA256).

La bibliothèque `cryptography.fernet` fournit une interface simple et sécurisée.
La clé maîtresse est générée une fois et stockée dans database/master.key.
EN PRODUCTION : utiliser un HSM ou un keystore système.
"""

import os
from cryptography.fernet import Fernet

KEY_FILE = os.path.join(os.path.dirname(__file__), "..", "database", "master.key")


def _key_path() -> str:
    return os.path.abspath(KEY_FILE)


def generate_key() -> bytes:
    """
    Génère une nouvelle clé Fernet et la sauvegarde dans database/master.key.
    Si le fichier existe déjà, retourne la clé existante sans l'écraser.
    """
    path = _key_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.exists(path):
        return load_key()

    key = Fernet.generate_key()
    with open(path, "wb") as f:
        f.write(key)
    return key


def load_key() -> bytes:
    """Charge la clé depuis le fichier. Lève une erreur si absente."""
    path = _key_path()
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Clé de chiffrement introuvable : {path}\n"
            "Lancez generate_key() en premier."
        )
    with open(path, "rb") as f:
        return f.read()


def encrypt_template(template_bytes: bytes) -> bytes:
    """
    Chiffre les bytes du template avec Fernet.
    Retourne le token chiffré (bytes) prêt pour stockage en BLOB SQLite.
    """
    key = load_key()
    f = Fernet(key)
    return f.encrypt(template_bytes)


def decrypt_template(token: bytes) -> bytes:
    """
    Déchiffre un token Fernet et retourne les bytes originaux du template.
    Lève cryptography.fernet.InvalidToken si la clé est incorrecte.
    """
    key = load_key()
    f = Fernet(key)
    return f.decrypt(token)

"""
Module 6: Gestion du stockage SQLite chiffré.

Tables :
  users     (id, name UNIQUE, created_at)
  templates (id, user_id FK, encrypted_blob BLOB, created_at)

Le droit à l'oubli (RGPD) est implémenté via delete_user() qui supprime
en cascade toutes les données liées à un utilisateur.
"""

import os
import sqlite3
from datetime import datetime
from typing import List, Optional, Tuple

DB_FILE = os.path.join(os.path.dirname(__file__), "..", "database", "secureprint.db")


def _db_path() -> str:
    return os.path.abspath(DB_FILE)


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path())
    conn.execute("PRAGMA foreign_keys = ON")  # Activer les contraintes FK
    return conn


def init_db() -> None:
    """
    Crée la base de données et les tables si elles n'existent pas.
    À appeler au démarrage de l'application.
    """
    os.makedirs(os.path.dirname(_db_path()), exist_ok=True)
    conn = _connect()
    with conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                name       TEXT    NOT NULL UNIQUE,
                role       TEXT    NOT NULL DEFAULT 'user',
                created_at TEXT    NOT NULL
            );

            CREATE TABLE IF NOT EXISTS templates (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id        INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                encrypted_blob BLOB    NOT NULL,
                created_at     TEXT    NOT NULL
            );
        """)
        # Migration: add role column if it doesn't exist yet (existing DBs)
        try:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT NOT NULL DEFAULT 'user'")
        except Exception:
            pass  # column already exists
    conn.close()


def enroll_user(name: str, encrypted_blob: bytes, role: str = "user") -> int:
    """
    Enregistre un utilisateur et son template chiffré.

    Args:
        name:           Nom de l'utilisateur.
        encrypted_blob: Token Fernet chiffré du template.
        role:           'admin' ou 'user' (défaut: 'user').

    Returns:
        L'ID de l'utilisateur créé.

    Raises:
        ValueError: Si l'utilisateur existe déjà.
    """
    now = datetime.utcnow().isoformat()
    conn = _connect()
    try:
        with conn:
            cur = conn.execute(
                "INSERT INTO users (name, role, created_at) VALUES (?, ?, ?)",
                (name, role, now)
            )
            user_id = cur.lastrowid
            conn.execute(
                "INSERT INTO templates (user_id, encrypted_blob, created_at) VALUES (?, ?, ?)",
                (user_id, encrypted_blob, now)
            )
        return user_id
    except sqlite3.IntegrityError:
        raise ValueError(f"L'utilisateur '{name}' est déjà enregistré.")
    finally:
        conn.close()


def get_all_users() -> List[Tuple[int, str, str]]:
    """Retourne la liste [(id, name, role), ...] de tous les utilisateurs enregistrés."""
    conn = _connect()
    rows = conn.execute("SELECT id, name, role FROM users ORDER BY name").fetchall()
    conn.close()
    return rows


def get_template(user_id: int) -> Optional[bytes]:
    """
    Retourne le blob chiffré du template d'un utilisateur.
    Retourne None si introuvable.
    """
    conn = _connect()
    row = conn.execute(
        "SELECT encrypted_blob FROM templates WHERE user_id = ? LIMIT 1",
        (user_id,)
    ).fetchone()
    conn.close()
    return row[0] if row else None


def get_user_by_name(name: str) -> Optional[Tuple[int, str, str]]:
    """Retourne (id, name, role) pour un nom donné, ou None."""
    conn = _connect()
    row = conn.execute(
        "SELECT id, name, role FROM users WHERE name = ?", (name,)
    ).fetchone()
    conn.close()
    return row


def delete_user(user_id: int) -> bool:
    """
    Supprime un utilisateur et tous ses templates (droit à l'oubli RGPD).
    Retourne True si un utilisateur a été supprimé, False sinon.
    """
    conn = _connect()
    with conn:
        cur = conn.execute("DELETE FROM users WHERE id = ?", (user_id,))
        deleted = cur.rowcount > 0
    conn.close()
    return deleted


def user_count() -> int:
    """Retourne le nombre total d'utilisateurs enregistrés."""
    conn = _connect()
    count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    conn.close()
    return count

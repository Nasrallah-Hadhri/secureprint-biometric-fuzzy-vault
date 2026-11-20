"""
Promote an enrolled user to the 'admin' role in the local SecurePrint database.

Usage (from the project root):
    python scripts/set_admin_role.py <user-name>
"""

import os
import sqlite3
import sys

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "secureprint.db")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/set_admin_role.py <user-name>")
        sys.exit(1)
    name = sys.argv[1]
    db = os.path.abspath(DB_PATH)
    if not os.path.exists(db):
        print(f"Database not found: {db}")
        sys.exit(1)
    conn = sqlite3.connect(db)
    with conn:
        conn.execute("UPDATE users SET role='admin' WHERE name=?", (name,))
    row = conn.execute("SELECT id, name, role FROM users WHERE name=?", (name,)).fetchone()
    conn.close()
    print("Result:", row if row else f"no user named '{name}'")


if __name__ == "__main__":
    main()

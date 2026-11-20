"""
Delete every enrolled user and template from the local SecurePrint database
(useful after changing the template format). The encryption key is kept.

Usage (from the project root):
    python scripts/wipe_db.py
"""

import os
import sqlite3

db_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "database", "secureprint.db"))

if os.path.exists(db_path):
    print("Wiping database to force re-enrollment...")
    conn = sqlite3.connect(db_path)
    with conn:
        conn.execute("DELETE FROM templates")
        conn.execute("DELETE FROM users")
    conn.close()
    print("Database wiped successfully! You can now restart the app and re-enroll.")
else:
    print("Database not found.")

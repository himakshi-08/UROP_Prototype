"""
Persistent Database Manager for Secure Fingerprint Template Storage.
Persists enrolled user cryptographic records (W, H, Y) to SQLite database.
STRICT SECURITY RULE: Never stores raw images, minutiae coordinates, or unencrypted templates.
"""

import os
import sqlite3
import json
import numpy as np

class SQLiteAuthDatabase:
    def __init__(self, db_path=None):
        if db_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            db_dir = os.path.join(base_dir, "database")
            os.makedirs(db_dir, exist_ok=True)
            db_path = os.path.join(db_dir, "auth_system.db")
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS enrolled_users (
                    username TEXT PRIMARY KEY,
                    helper_data_hex TEXT NOT NULL,
                    commitment_hash TEXT NOT NULL,
                    public_key_hex TEXT NOT NULL,
                    enrolled_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def store_user(self, username, helper_data_vector, commitment_hash, public_key_int):
        """
        Stores only protected cryptographic references.
        """
        # Pack binary vector (256 bits) into 32-byte hex string
        helper_bytes = np.packbits(helper_data_vector).tobytes()
        helper_hex = helper_bytes.hex()
        
        # Convert large integer public key to hex
        pk_hex = hex(public_key_int)

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO enrolled_users 
                (username, helper_data_hex, commitment_hash, public_key_hex)
                VALUES (?, ?, ?, ?)
            """, (username, helper_hex, commitment_hash, pk_hex))
            conn.commit()

    def get_user(self, username):
        """
        Retrieves protected enrollment records for a user.
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT helper_data_hex, commitment_hash, public_key_hex, enrolled_timestamp 
                FROM enrolled_users 
                WHERE username = ?
            """, (username,))
            row = cursor.fetchone()

        if row is None:
            return None

        helper_hex, commitment_hash, pk_hex, timestamp = row
        helper_bytes = bytes.fromhex(helper_hex)
        helper_vector = np.unpackbits(np.frombuffer(helper_bytes, dtype=np.uint8))
        # packbits/unpackbits is byte-aligned; keep the original 256-bit helper.
        if len(helper_vector) > 256:
            helper_vector = helper_vector[:256]
        public_key_int = int(pk_hex, 16)

        return {
            "username": username,
            "helper_data": helper_vector,
            "commitment": commitment_hash,
            "public_key": public_key_int,
            "enrolled_at": timestamp
        }

    def list_users(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT username, commitment_hash, enrolled_timestamp FROM enrolled_users")
            return cursor.fetchall()

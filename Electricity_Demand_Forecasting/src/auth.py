import hashlib
import hmac
import secrets
import sqlite3
from contextlib import closing
from pathlib import Path

USER_DB_PATH = Path(__file__).resolve().parent.parent / "users.db"
PASSWORD_HASH_ITERATIONS = 600_000


def _initialize_users_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            user_id TEXT PRIMARY KEY COLLATE NOCASE,
            password_salt BLOB NOT NULL,
            password_hash BLOB NOT NULL
        )
        """
    )


def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        PASSWORD_HASH_ITERATIONS,
    )


def register_user(user_id: str, password: str, db_path: str | Path = USER_DB_PATH) -> bool:
    """Create a user with a salted password hash; return False if the ID is taken."""
    normalized_user_id = user_id.strip()
    if not normalized_user_id:
        raise ValueError("User ID is required.")
    if len(password) < 8:
        raise ValueError("Password must be at least 8 characters.")

    salt = secrets.token_bytes(16)
    password_hash = _hash_password(password, salt)
    with closing(sqlite3.connect(db_path, timeout=10)) as connection:
        with connection:
            _initialize_users_table(connection)
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO users (user_id, password_salt, password_hash)
                VALUES (?, ?, ?)
                """,
                (normalized_user_id, salt, password_hash),
            )
            return cursor.rowcount == 1


def verify_user(user_id: str, password: str, db_path: str | Path = USER_DB_PATH) -> bool:
    """Verify a registered user's password against its stored salted hash."""
    normalized_user_id = user_id.strip()
    if not normalized_user_id or not password:
        return False

    with closing(sqlite3.connect(db_path, timeout=10)) as connection:
        with connection:
            _initialize_users_table(connection)
            row = connection.execute(
                "SELECT password_salt, password_hash FROM users WHERE user_id = ?",
                (normalized_user_id,),
            ).fetchone()
    if row is None:
        return False

    salt, stored_hash = row
    return hmac.compare_digest(_hash_password(password, salt), stored_hash)


def credentials_match(
    user_id: str,
    password: str,
    expected_user_id: str,
    expected_password: str,
) -> bool:
    """Compare optional environment-configured credentials in constant time."""
    if not user_id or not password or not expected_user_id or not expected_password:
        return False
    return hmac.compare_digest(user_id, expected_user_id) and hmac.compare_digest(
        password, expected_password
    )

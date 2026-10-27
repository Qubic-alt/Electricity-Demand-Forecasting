import sqlite3

import pytest

from src.auth import credentials_match, register_user, verify_user


def test_credentials_match_configured_values():
    assert credentials_match("operator", "s3cure-value", "operator", "s3cure-value")


def test_credentials_reject_mismatched_values():
    assert not credentials_match("operator", "wrong", "operator", "s3cure-value")
    assert not credentials_match("wrong", "s3cure-value", "operator", "s3cure-value")


def test_credentials_reject_empty_values():
    assert not credentials_match("", "password", "operator", "password")
    assert not credentials_match("operator", "", "operator", "password")
    assert not credentials_match("operator", "password", "", "password")


def test_registered_user_can_sign_in_with_case_insensitive_user_id(tmp_path):
    database = tmp_path / "users.db"

    assert register_user("operator", "correct-horse", database)
    assert verify_user("OPERATOR", "correct-horse", database)
    assert not verify_user("operator", "wrong-password", database)


def test_registration_rejects_duplicate_user_id(tmp_path):
    database = tmp_path / "users.db"

    assert register_user("operator", "correct-horse", database)
    assert not register_user("OPERATOR", "another-password", database)


def test_registration_stores_only_a_salted_hash(tmp_path):
    database = tmp_path / "users.db"

    register_user("operator", "correct-horse", database)
    with sqlite3.connect(database) as connection:
        salt, password_hash = connection.execute(
            "SELECT password_salt, password_hash FROM users WHERE user_id = ?",
            ("operator",),
        ).fetchone()

    assert salt != b"correct-horse"
    assert password_hash != b"correct-horse"
    assert salt != password_hash


def test_registration_validates_required_fields(tmp_path):
    with pytest.raises(ValueError, match="User ID is required"):
        register_user(" ", "correct-horse", tmp_path / "users.db")
    with pytest.raises(ValueError, match="at least 8 characters"):
        register_user("operator", "short", tmp_path / "users.db")

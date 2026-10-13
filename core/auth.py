"""Small, framework-free auth helpers: API key generation/hashing and role comparison.
JWT verification lives in app/deps.py since it needs Settings/Request."""
from __future__ import annotations

import hashlib
import secrets

ROLE_RANK = {"viewer": 0, "member": 1, "admin": 2}


def generate_api_key() -> str:
    return "cxk_" + secrets.token_urlsafe(32)


def hash_api_key(raw: str) -> str:
    return hashlib.sha256(raw.encode()).hexdigest()

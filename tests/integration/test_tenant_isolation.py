"""Proves workspace A cannot read workspace B's documents/chunks under RLS.

Skipped unless DATABASE_URL is set (so the default `pytest` run stays network-free,
per the project's unit-test rule). Run against the local docker-compose postgres:

    docker compose up -d db
    DATABASE_URL=postgresql://postgres:postgres@localhost:55432/contexta python scripts/migrate.py
    DATABASE_URL=postgresql://postgres:postgres@localhost:55432/contexta pytest tests/integration
"""
from __future__ import annotations

import os
import uuid

import psycopg2
import pytest

DATABASE_URL = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="set DATABASE_URL to run DB integration tests")


def _conn():
    return psycopg2.connect(DATABASE_URL)


def _as_user(cur, user_id: str) -> None:
    cur.execute("SET ROLE authenticated")
    cur.execute("SET LOCAL request.jwt.claims = %s", ('{"sub": "%s"}' % user_id,))


def test_workspace_isolation():
    org_id, ws_a, ws_b = str(uuid.uuid4()), str(uuid.uuid4()), str(uuid.uuid4())
    user_a, user_b = str(uuid.uuid4()), str(uuid.uuid4())
    doc_a = str(uuid.uuid4())

    with _conn() as conn, conn.cursor() as cur:
        cur.execute("INSERT INTO organizations (id, name) VALUES (%s, 'Test Org')", (org_id,))
        cur.execute(
            "INSERT INTO workspaces (id, org_id, name) VALUES (%s, %s, 'A'), (%s, %s, 'B')",
            (ws_a, org_id, ws_b, org_id),
        )
        cur.execute(
            "INSERT INTO auth.users (id, email) VALUES (%s, 'a@test.com'), (%s, 'b@test.com')",
            (user_a, user_b),
        )
        cur.execute(
            "INSERT INTO memberships (user_id, workspace_id, role) VALUES (%s, %s, 'member'), (%s, %s, 'member')",
            (user_a, ws_a, user_b, ws_b),
        )
        cur.execute(
            "INSERT INTO documents (id, workspace_id, filename, content_type, storage_path, content_hash, status) "
            "VALUES (%s, %s, 'secret.pdf', 'application/pdf', 'x', 'hash-a', 'ready')",
            (doc_a, ws_a),
        )
        cur.execute(
            "INSERT INTO chunks (id, document_id, workspace_id, chunk_index, content) VALUES (%s, %s, %s, 0, 'Workspace A secret content')",
            (str(uuid.uuid4()), doc_a, ws_a),
        )

    try:
        with _conn() as conn, conn.cursor() as cur:
            _as_user(cur, user_b)
            cur.execute("SELECT count(*) FROM chunks WHERE workspace_id = %s", (ws_a,))
            assert cur.fetchone()[0] == 0, "workspace B must not see workspace A's chunks"
            cur.execute("SELECT count(*) FROM documents")
            assert cur.fetchone()[0] == 0, "workspace B must not see workspace A's documents"

        with _conn() as conn, conn.cursor() as cur:
            _as_user(cur, user_a)
            cur.execute("SELECT count(*) FROM chunks WHERE workspace_id = %s", (ws_a,))
            assert cur.fetchone()[0] == 1, "workspace A must see its own chunk"
    finally:
        with _conn() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM organizations WHERE id = %s", (org_id,))
            cur.execute("DELETE FROM auth.users WHERE id IN (%s, %s)", (user_a, user_b))

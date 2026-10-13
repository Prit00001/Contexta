"""End-to-end ingestion: upsert_document -> run_ingestion -> chunks in the DB, plus
dedupe (identical re-upload is a no-op) and replace (changed content swaps chunks).

Skipped unless DATABASE_URL is set (see tests/integration/test_tenant_isolation.py
for how to run these against the local docker-compose postgres).
"""
from __future__ import annotations

import hashlib
import os
import uuid

import psycopg2
import pytest

from app.config import get_settings
from core import db
from core.storage import LocalStorage
from services.ingestion import run_ingestion

DATABASE_URL = os.environ.get("DATABASE_URL")
pytestmark = pytest.mark.skipif(not DATABASE_URL, reason="set DATABASE_URL to run DB integration tests")


def _conn():
    return psycopg2.connect(DATABASE_URL)


def test_ingest_then_dedupe_then_replace(tmp_path, monkeypatch):
    monkeypatch.setenv("STORAGE_BACKEND", "local")
    monkeypatch.setenv("STORAGE_DIR", str(tmp_path))
    get_settings.cache_clear()
    settings = get_settings()
    storage = LocalStorage(tmp_path)

    org_id, ws_id = str(uuid.uuid4()), str(uuid.uuid4())
    conn = db.get_conn(DATABASE_URL)
    try:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO organizations (id, name) VALUES (%s, 'T')", (org_id,))
            cur.execute("INSERT INTO workspaces (id, org_id, name) VALUES (%s, %s, 'W')", (ws_id, org_id))
        conn.commit()

        original = b"# Intro\nRetrieval augmented generation grounds answers in real documents."
        storage_path = f"{ws_id}/handbook.md"
        storage.save(storage_path, original)
        doc_id, needs_ingest = db.upsert_document(
            conn, workspace_id=ws_id, filename="handbook.md", content_type="text/markdown",
            storage_path=storage_path, content_hash=hashlib.sha256(original).hexdigest(),
        )
        assert needs_ingest is True

        run_ingestion(conn, settings, doc_id)
        doc = db.get_document(conn, doc_id)
        assert doc["status"] == "ready"
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM chunks WHERE document_id=%s", (doc_id,))
            assert cur.fetchone()[0] == 1

        # Re-uploading identical bytes is a no-op.
        doc_id2, needs_ingest2 = db.upsert_document(
            conn, workspace_id=ws_id, filename="handbook.md", content_type="text/markdown",
            storage_path=storage_path, content_hash=hashlib.sha256(original).hexdigest(),
        )
        assert doc_id2 == doc_id and needs_ingest2 is False

        # Changed content under the same filename replaces the chunks.
        changed = b"# Intro\nCompletely different handbook content now, several sentences long."
        storage.save(storage_path, changed)
        doc_id3, needs_ingest3 = db.upsert_document(
            conn, workspace_id=ws_id, filename="handbook.md", content_type="text/markdown",
            storage_path=storage_path, content_hash=hashlib.sha256(changed).hexdigest(),
        )
        assert doc_id3 == doc_id and needs_ingest3 is True
        run_ingestion(conn, settings, doc_id)
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM chunks WHERE document_id=%s", (doc_id,))
            rows = [r[0] for r in cur.fetchall()]
        assert len(rows) == 1 and "Completely different" in rows[0]
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM organizations WHERE id = %s", (org_id,))
        conn.commit()
        conn.close()
        get_settings.cache_clear()

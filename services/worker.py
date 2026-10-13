"""Background ingestion worker: polls the jobs table and runs the pipeline. Runs as
its own process (python -m services.worker), separate from the FastAPI request path.

ponytail: single long-lived connection, reconnect-on-drop is not handled. Upgrade
path: wrap the loop body in a retry on psycopg2.OperationalError if idle connections
get dropped by the DB (e.g. Supabase's pooler) in practice.
"""
from __future__ import annotations

import logging
import time

from app.config import get_settings
from core import db
from services.ingestion import run_ingestion

log = logging.getLogger("contexta.worker")


def run_once(conn, settings) -> bool:
    """Claims and processes one job. Returns False if there was nothing to do."""
    job = db.claim_job(conn)
    if job is None:
        return False
    try:
        run_ingestion(conn, settings, job["document_id"])
        db.mark_job(conn, job["id"], "done")
    except Exception as exc:  # noqa: BLE001 - one bad document must not kill the worker loop
        log.exception("ingestion failed for document %s", job["document_id"])
        db.mark_job(conn, job["id"], "failed", str(exc))
        db.set_document_status(conn, job["document_id"], "failed", error=str(exc))
    return True


def run_forever(poll_interval: float = 2.0) -> None:
    settings = get_settings()
    conn = db.get_conn(settings.database_url)
    log.info("Worker ready, polling every %.1fs", poll_interval)
    while True:
        if not run_once(conn, settings):
            time.sleep(poll_interval)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_forever()

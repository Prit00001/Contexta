"""Apply supabase/migrations/*.sql, in filename order, against DATABASE_URL.

Plain SQL files + a tracking table, no framework: works unmodified against local
docker-compose postgres, CI, and the real Supabase project (same migration files,
just a different DATABASE_URL). Already-applied files are skipped.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import psycopg2

MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "supabase" / "migrations"


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("DATABASE_URL is not set")

    conn = psycopg2.connect(dsn)
    try:
        with conn.cursor() as cur:
            cur.execute(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "filename text PRIMARY KEY, applied_at timestamptz NOT NULL DEFAULT now())"
            )
        conn.commit()

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM schema_migrations WHERE filename = %s", (path.name,))
                if cur.fetchone():
                    continue
            print(f"Applying {path.name}")
            with conn.cursor() as cur:
                cur.execute(path.read_text())
                cur.execute("INSERT INTO schema_migrations (filename) VALUES (%s)", (path.name,))
            conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    main()

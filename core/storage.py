"""Original-file storage abstraction. Local filesystem for dev/tests; Supabase Storage
(via its REST API, over plain httpx - no extra SDK) for production, matching the rest
of the codebase's style of talking to OpenAI-compatible/arXiv HTTP APIs directly."""
from __future__ import annotations

from pathlib import Path

import httpx


class LocalStorage:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, path: str, data: bytes) -> str:
        p = self.root / path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        return path

    def load(self, path: str) -> bytes:
        return (self.root / path).read_bytes()

    def delete(self, path: str) -> None:
        (self.root / path).unlink(missing_ok=True)


class SupabaseStorage:
    def __init__(self, base_url: str, bucket: str, service_key: str, client: httpx.Client):
        self.base_url = base_url.rstrip("/")
        self.bucket = bucket
        self.client = client
        self.headers = {"Authorization": f"Bearer {service_key}", "apikey": service_key}

    def save(self, path: str, data: bytes) -> str:
        r = self.client.post(
            f"{self.base_url}/storage/v1/object/{self.bucket}/{path}",
            headers={**self.headers, "Content-Type": "application/octet-stream"},
            content=data,
        )
        r.raise_for_status()
        return path

    def load(self, path: str) -> bytes:
        r = self.client.get(f"{self.base_url}/storage/v1/object/{self.bucket}/{path}", headers=self.headers)
        r.raise_for_status()
        return r.content

    def delete(self, path: str) -> None:
        r = self.client.delete(f"{self.base_url}/storage/v1/object/{self.bucket}/{path}", headers=self.headers)
        if r.status_code not in (200, 404):
            r.raise_for_status()


def get_storage(settings):
    if settings.storage_backend == "supabase":
        return SupabaseStorage(
            settings.supabase_url, settings.storage_bucket, settings.supabase_service_key, httpx.Client(timeout=30.0)
        )
    return LocalStorage(Path(settings.storage_dir))

"""File storage: local disk (default) ya S3-compatible (Vercel / production).

Vercel serverless par local disk persistent nahi hai aur request body 4.5 MB tak
hi ja sakta hai - isliye production me browser seedha storage par upload karta
hai (presigned PUT) aur API ko sirf URL milta hai.

Config:
  WM_STORAGE=local|s3
  S3_BUCKET, S3_REGION, S3_ENDPOINT_URL (R2/MinIO ke liye),
  AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Optional

from . import config as cfgmod

cfg = cfgmod.cfg


class Storage:
    """Interface used by the API."""

    mode = "local"

    def put(self, data: bytes, key: Optional[str] = None,
            content_type: str = "application/octet-stream") -> str:
        raise NotImplementedError

    def path(self, key: str) -> str:
        """Local filesystem path (raises on remote backends)."""
        raise NotImplementedError

    def url(self, key: str, ttl: int = 3600) -> str:
        raise NotImplementedError

    def presign_put(self, key: str, content_type: str = "application/octet-stream",
                    ttl: int = 900) -> Optional[str]:
        """Direct-to-storage upload URL, or None if not supported."""
        return None

    def read(self, key: str) -> bytes:
        raise NotImplementedError


class LocalStorage(Storage):
    mode = "local"

    def __init__(self, root: Optional[Path] = None):
        self.root = Path(root or cfg.workdir / "blobs")
        self.root.mkdir(parents=True, exist_ok=True)

    def _abs(self, key: str) -> Path:
        p = self.root / key.replace("..", "").lstrip("/")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def put(self, data: bytes, key: Optional[str] = None, content_type: str = "") -> str:
        key = key or f"{uuid.uuid4().hex[:12]}"
        self._abs(key).write_bytes(data)
        return key

    def path(self, key: str) -> str:
        return str(self._abs(key))

    def url(self, key: str, ttl: int = 3600) -> str:
        return f"/media/blobs/{key}"

    def read(self, key: str) -> bytes:
        return self._abs(key).read_bytes()


class S3Storage(Storage):
    mode = "s3"

    def __init__(self):
        import boto3  # type: ignore

        self.bucket = os.environ["S3_BUCKET"]
        kwargs = {"region_name": os.environ.get("S3_REGION", "auto")}
        if os.environ.get("S3_ENDPOINT_URL"):
            kwargs["endpoint_url"] = os.environ["S3_ENDPOINT_URL"]
        self.s3 = boto3.client(
            "s3",
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
            **kwargs,
        )
        self.public_base = os.environ.get("S3_PUBLIC_BASE_URL", "").rstrip("/")

    def put(self, data: bytes, key: Optional[str] = None, content_type: str = "") -> str:
        key = key or uuid.uuid4().hex[:12]
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=data,
                           ContentType=content_type or "application/octet-stream")
        return key

    def url(self, key: str, ttl: int = 3600) -> str:
        if self.public_base:
            return f"{self.public_base}/{key}"
        return self.s3.generate_presigned_url(
            "get_object", Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=ttl)

    def presign_put(self, key: str, content_type: str = "application/octet-stream",
                    ttl: int = 900) -> Optional[str]:
        return self.s3.generate_presigned_url(
            "put_object", Params={"Bucket": self.bucket, "Key": key,
                                  "ContentType": content_type}, ExpiresIn=ttl)

    def read(self, key: str) -> bytes:
        return self.s3.get_object(Bucket=self.bucket, Key=key)["Body"].read()


_STORE: Optional[Storage] = None


def get_storage() -> Storage:
    global _STORE
    if _STORE is None:
        mode = os.environ.get("WM_STORAGE", "local").lower()
        if mode == "s3":
            try:
                _STORE = S3Storage()
            except Exception as exc:                       # noqa: BLE001
                print(f"[storage] s3 unavailable ({exc}) -> local")
                _STORE = LocalStorage()
        else:
            _STORE = LocalStorage()
    return _STORE


def download(url: str, dest: str, timeout: int = 300) -> str:
    """Fetch a remote asset to a local temp path (used by the worker)."""
    import urllib.request

    Path(dest).parent.mkdir(parents=True, exist_ok=True)
    if url.startswith("/"):                                # same-origin media path
        rel = url.split("/media/blobs/", 1)[-1]
        data = get_storage().read(rel)
        Path(dest).write_bytes(data)
        return dest
    req = urllib.request.Request(url, headers={"User-Agent": "watermark-remover/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r, open(dest, "wb") as f:
        f.write(r.read())
    return dest

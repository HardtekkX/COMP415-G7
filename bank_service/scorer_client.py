"""HTTP client for the scoring service.

Registers the bank's public context on first use, caches the context_id in
keys_dir/scorer_context.json, and registers again when the scorer answers 404
(it keeps contexts in memory, so a restart or eviction forgets them).
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from pathlib import Path

import httpx

from common.schemas import ContextResponse, ScoreRequest, ScoreResponse, b64encode

log = logging.getLogger("bank_service")


class ScorerError(RuntimeError):
    """The scoring service failed or answered unexpectedly."""


class ScorerClient:
    def __init__(
        self,
        base_url: str,
        public_context: bytes,
        cache_path: Path,
        http: httpx.Client | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self._public_context = public_context
        self._public_sha256 = hashlib.sha256(public_context).hexdigest()
        self._cache_path = Path(cache_path)
        self._http = http or httpx.Client(timeout=timeout)
        self._lock = threading.Lock()
        self._context_id: str | None = self._read_cache()

    # --- context_id cache -------------------------------------------------

    def _read_cache(self) -> str | None:
        try:
            data = json.loads(self._cache_path.read_text())
        except (OSError, ValueError):
            return None
        if data.get("scorer_url") != self.base_url or data.get("public_sha256") != self._public_sha256:
            return None
        return data.get("context_id")

    def _write_cache(self, context_id: str) -> None:
        tmp = self._cache_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({
            "scorer_url": self.base_url,
            "public_sha256": self._public_sha256,
            "context_id": context_id,
        }))
        tmp.replace(self._cache_path)

    # --- API calls ---------------------------------------------------------

    def _register(self) -> str:
        resp = self._post("/contexts", {"context": b64encode(self._public_context)})
        if resp.status_code != 200:
            raise ScorerError(f"POST /contexts returned {resp.status_code}")
        context_id = ContextResponse.model_validate(resp.json()).context_id
        self._write_cache(context_id)
        log.info(
            "context registered context_id=%s size_bytes=%d",
            context_id, len(self._public_context),
        )
        return context_id

    def context_id(self) -> str:
        with self._lock:
            if self._context_id is None:
                self._context_id = self._register()
            return self._context_id

    def _reregister(self, stale: str) -> str:
        with self._lock:
            # Another thread may already have replaced the stale id.
            if self._context_id == stale:
                log.info("context unknown to scorer context_id=%s; registering again", stale)
                self._context_id = self._register()
            return self._context_id

    def score(self, pseudonym: str, ciphertext: bytes) -> bytes:
        """POST /score; returns the encrypted logit. Retries once after a 404."""
        context_id = self.context_id()
        resp = self._post_score(pseudonym, context_id, ciphertext)
        if resp.status_code == 404:
            context_id = self._reregister(context_id)
            resp = self._post_score(pseudonym, context_id, ciphertext)
        if resp.status_code != 200:
            raise ScorerError(f"POST /score returned {resp.status_code}")
        body = ScoreResponse.model_validate(resp.json())
        if body.pseudonym != pseudonym:
            raise ScorerError("scorer answered for a different pseudonym")
        return body.encrypted_logit

    def _post_score(self, pseudonym: str, context_id: str, ciphertext: bytes) -> httpx.Response:
        req = ScoreRequest(pseudonym=pseudonym, context_id=context_id, ciphertext=ciphertext)
        return self._post("/score", req.model_dump(mode="json"))

    def _post(self, path: str, body: dict) -> httpx.Response:
        try:
            return self._http.post(self.base_url + path, json=body)
        except httpx.HTTPError as exc:
            raise ScorerError(f"cannot reach scorer at {self.base_url}: {type(exc).__name__}") from None

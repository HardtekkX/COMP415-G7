"""Scoring service (plays KKB / Findeks): encrypted logistic-regression logit.

Holds the model weights in plaintext and never sees the applicant's data. The
bank registers its public CKKS context once (POST /contexts); each POST /score
then carries a pseudonym, the context_id and one encrypted feature vector, and
gets back the encrypted logit w . x + b. Only the bank can decrypt it.

Logging is limited to pseudonyms, context IDs, byte sizes, timings and status
codes. Request bodies are never logged, and validation errors do not echo input.

Run:  uvicorn scoring_service.main:app --port 8001
Env:  SCORER_MODEL_PATH    model JSON (default training/artifacts/model.json)
      SCORER_MAX_CONTEXTS  contexts kept in memory, oldest evicted (default 100)
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import tenseal as ts
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from common import he
from common.schemas import (
    ContextRequest,
    ContextResponse,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "model.json"
DEFAULT_MAX_CONTEXTS = 100

log = logging.getLogger("scoring_service")


@dataclass(frozen=True)
class Model:
    weights: list[float]
    bias: float

    @classmethod
    def load(cls, path: Path) -> Model:
        data = json.loads(Path(path).read_text())
        weights = [float(w) for w in data["weights"]]
        if len(weights) != len(data["feature_order"]):
            raise ValueError(f"{path}: weights and feature_order differ in length")
        return cls(weights=weights, bias=float(data["bias"]))


class ContextStore:
    """Public contexts in memory, capped at max_size; the oldest is evicted first.

    About 3 MB each, so the cap bounds memory however many uploads arrive.
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._items: OrderedDict[str, ts.Context] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, ctx: ts.Context) -> str:
        context_id = secrets.token_hex(16)
        with self._lock:
            self._items[context_id] = ctx
            while len(self._items) > self.max_size:
                evicted, _ = self._items.popitem(last=False)
                log.info("context evicted context_id=%s", evicted)
        return context_id

    def get(self, context_id: str) -> ts.Context | None:
        with self._lock:
            return self._items.get(context_id)

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)

    def contexts(self) -> list[ts.Context]:
        with self._lock:
            return list(self._items.values())


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def create_app(model_path: Path | str | None = None, max_contexts: int | None = None) -> FastAPI:
    model = Model.load(Path(model_path or os.environ.get("SCORER_MODEL_PATH", DEFAULT_MODEL_PATH)))
    store = ContextStore(
        max_contexts or int(os.environ.get("SCORER_MAX_CONTEXTS", DEFAULT_MAX_CONTEXTS))
    )

    app = FastAPI(title="Scoring service")
    app.state.model = model
    app.state.contexts = store

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default handler echoes the offending input; return locations only.
        fields = [".".join(str(p) for p in err["loc"]) for err in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "invalid request", "fields": fields})

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    # Plain `def` endpoints: FastAPI runs them in its threadpool, so the HE work
    # does not block the event loop.
    @app.post("/contexts", response_model=ContextResponse)
    def register_context(req: ContextRequest) -> ContextResponse:
        start = time.perf_counter()
        try:
            ctx = he.load_public_context(req.context)
        except Exception:
            # Covers unparsable bytes and a context that holds a secret key.
            log.info("context rejected size_bytes=%d status=400", len(req.context))
            raise HTTPException(400, "not a valid public CKKS context") from None
        context_id = store.add(ctx)
        log.info(
            "context registered context_id=%s size_bytes=%d ms=%.1f",
            context_id, len(req.context), _ms_since(start),
        )
        return ContextResponse(context_id=context_id)

    @app.post("/score", response_model=ScoreResponse)
    def score(req: ScoreRequest) -> ScoreResponse:
        start = time.perf_counter()
        size = len(req.ciphertext)

        def reject(status: int, detail: str) -> HTTPException:
            log.info(
                "score rejected pseudonym=%s context_id=%s ciphertext_bytes=%d status=%d",
                req.pseudonym, req.context_id, size, status,
            )
            return HTTPException(status, detail)

        ctx = store.get(req.context_id)
        if ctx is None:
            raise reject(404, "unknown context_id; register the context again")
        try:
            enc_x = he.load_ciphertext(ctx, req.ciphertext)
        except Exception:
            raise reject(400, "ciphertext cannot be read with this context") from None
        if enc_x.size() != len(model.weights):
            raise reject(422, f"vector length must be {len(model.weights)}")
        try:
            out = he.encrypted_logit(ctx, enc_x, model.weights, model.bias)
        except Exception:
            raise reject(400, "ciphertext cannot be scored") from None

        log.info(
            "score pseudonym=%s context_id=%s ciphertext_bytes=%d logit_bytes=%d ms=%.1f",
            req.pseudonym, req.context_id, size, len(out), _ms_since(start),
        )
        return ScoreResponse(pseudonym=req.pseudonym, encrypted_logit=out)

    return app


def _configure_logging() -> None:
    # uvicorn configures only its own loggers; give ours an INFO handler too.
    log.setLevel(logging.INFO)
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        log.addHandler(handler)


_configure_logging()
app = create_app()

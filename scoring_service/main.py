"""Scoring service (plays KKB / Findeks): encrypted logistic-regression logit.

Holds the model weights in plaintext and never sees the applicant's data. The
bank registers its public CKKS context once (POST /contexts); each POST /score
then carries a pseudonym, the context_id and one encrypted feature vector, and
gets back the encrypted logit w . x + b. Only the bank can decrypt it.

Logging is limited to pseudonyms, context IDs, byte sizes, timings and status
codes. Request bodies are never logged, and validation errors do not echo input.

Run:  uvicorn scoring_service.main:app --port 8001
Env:  SCORER_MODEL_PATH       weights and bias (default training/artifacts/model.json)
      SCORER_MAX_CONTEXTS     contexts kept in memory, oldest evicted (default 100)
      SCORER_DEMO_ENDPOINTS   GET /activity and GET /contexts/summary for the demo
                              screen (default true; false answers 404). They have
                              no authentication: a real deployment would turn them
                              off or put them behind authentication.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import tenseal as ts
from fastapi import FastAPI, HTTPException

from common import he
from common.api import configure_logger, quiet_validation_errors
from common.schemas import (
    ACTIVITY_PREFIX_CHARS,
    ActivityRecord,
    ActivityResponse,
    ContextInfo,
    ContextRequest,
    ContextResponse,
    ContextSummary,
    HealthResponse,
    ScoreRequest,
    ScoreResponse,
    b64encode,
)

DEFAULT_MODEL_PATH = Path(__file__).resolve().parent.parent / "training" / "artifacts" / "model.json"
DEFAULT_MAX_CONTEXTS = 100
ACTIVITY_SIZE = 50

log = logging.getLogger("scoring_service")


@dataclass(frozen=True)
class Model:
    weights: list[float]
    bias: float

    @classmethod
    def load(cls, path: Path) -> Model:
        """model.json holds only weights and bias; the feature count is len(weights)."""
        data = json.loads(Path(path).read_text())
        weights = [float(w) for w in data["weights"]]
        if not weights:
            raise ValueError(f"{path}: no weights")
        return cls(weights=weights, bias=float(data["bias"]))

    @property
    def n_features(self) -> int:
        return len(self.weights)


class ContextStore:
    """Public contexts in memory, capped at max_size; the oldest is evicted first.

    About 3 MB each, so the cap bounds memory however many uploads arrive.
    """

    def __init__(self, max_size: int):
        if max_size < 1:
            raise ValueError("max_size must be at least 1")
        self.max_size = max_size
        self._items: OrderedDict[str, ts.Context] = OrderedDict()
        # Serialised size and registration time per context, for GET /contexts/summary.
        # The context bytes themselves are not kept.
        self._info: dict[str, ContextInfo] = {}
        self._lock = threading.Lock()

    def add(self, ctx: ts.Context, size_bytes: int) -> str:
        context_id = secrets.token_hex(16)
        info = ContextInfo(context_id=context_id, size_bytes=size_bytes, registered_at=_now())
        with self._lock:
            self._items[context_id] = ctx
            self._info[context_id] = info
            while len(self._items) > self.max_size:
                evicted, _ = self._items.popitem(last=False)
                del self._info[evicted]
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

    def summary(self) -> ContextSummary:
        with self._lock:
            infos = [self._info[cid] for cid in self._items]
        return ContextSummary(count=len(infos), max_contexts=self.max_size, contexts=infos)


class ActivityLog:
    """The last `max_size` successful scores, for the demo's scorer screen.

    Records hold only what the scorer may log plus short base64 prefixes of the
    ciphertext it received and the encrypted logit it returned.
    """

    def __init__(self, max_size: int = ACTIVITY_SIZE):
        self._records: deque[ActivityRecord] = deque(maxlen=max_size)
        self._lock = threading.Lock()

    def add(self, record: ActivityRecord) -> None:
        with self._lock:
            self._records.append(record)

    def newest_first(self) -> list[ActivityRecord]:
        with self._lock:
            return list(reversed(self._records))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _b64_prefix(data: bytes) -> str:
    # Encode just enough bytes for the prefix (3 bytes -> 4 characters).
    return b64encode(data[: ACTIVITY_PREFIX_CHARS // 4 * 3])[:ACTIVITY_PREFIX_CHARS]


def _ms_since(start: float) -> float:
    return (time.perf_counter() - start) * 1000


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def create_app(
    model_path: Path | str | None = None,
    max_contexts: int | None = None,
    demo_endpoints: bool | None = None,
) -> FastAPI:
    model = Model.load(Path(model_path or os.environ.get("SCORER_MODEL_PATH", DEFAULT_MODEL_PATH)))
    store = ContextStore(
        max_contexts or int(os.environ.get("SCORER_MAX_CONTEXTS", DEFAULT_MAX_CONTEXTS))
    )
    activity = ActivityLog()
    if demo_endpoints is None:
        demo_endpoints = _env_flag("SCORER_DEMO_ENDPOINTS", True)

    app = FastAPI(title="Scoring service")
    app.state.model = model
    app.state.contexts = store
    app.state.activity = activity

    quiet_validation_errors(app)

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
        context_id = store.add(ctx, len(req.context))
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
        if enc_x.size() != model.n_features:
            raise reject(422, f"vector length must be {model.n_features}")
        try:
            out = he.encrypted_logit(ctx, enc_x, model.weights, model.bias)
        except Exception:
            raise reject(400, "ciphertext cannot be scored") from None

        ms = _ms_since(start)
        log.info(
            "score pseudonym=%s context_id=%s ciphertext_bytes=%d logit_bytes=%d ms=%.1f",
            req.pseudonym, req.context_id, size, len(out), ms,
        )
        activity.add(
            ActivityRecord(
                timestamp=_now(),
                pseudonym=req.pseudonym,
                context_id=req.context_id,
                ciphertext_bytes=size,
                ciphertext_prefix=_b64_prefix(req.ciphertext),
                result_prefix=_b64_prefix(out),
                compute_ms=round(ms, 3),
            )
        )
        return ScoreResponse(pseudonym=req.pseudonym, encrypted_logit=out)

    # Demo screen only. No authentication: a real deployment would set
    # SCORER_DEMO_ENDPOINTS=false (both routes then answer 404) or put them
    # behind authentication.
    if demo_endpoints:

        @app.get("/activity", response_model=ActivityResponse)
        def get_activity() -> ActivityResponse:
            return ActivityResponse(records=activity.newest_first())

        @app.get("/contexts/summary", response_model=ContextSummary)
        def contexts_summary() -> ContextSummary:
            return store.summary()

    return app


configure_logger("scoring_service")
app = create_app()

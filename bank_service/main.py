"""Bank service: scores an applicant through the encrypted scoring service.

POST /apply takes the applicant's national ID and plaintext features. The bank
scales them, replaces the ID with HMAC-SHA256(bank key, national ID), encrypts,
calls the scorer, decrypts the logit, applies the sigmoid and decides.

Logs only pseudonyms, decisions, sizes and timings (no national IDs, feature
values, logits or probabilities). Validation errors do not echo input.

Run:  uvicorn bank_service.main:app --port 8000
Config: bank_service/config.toml (env BANK_SCORER_URL, BANK_KEYS_DIR, BANK_CONFIG)
"""

from __future__ import annotations

import logging

import httpx
from fastapi import FastAPI, HTTPException

from bank_service.config import BankConfig
from bank_service.core import Bank
from bank_service.schemas import ApplicantFeatures, ApplyRequest, ApplyResponse
from bank_service.scorer_client import ScorerError
from common.api import configure_logger, quiet_validation_errors
from common.schemas import HealthResponse

log = logging.getLogger("bank_service")


def create_app(config: BankConfig | None = None, http: httpx.Client | None = None) -> FastAPI:
    bank = Bank.from_config(config or BankConfig.load(), http=http)
    fields = list(ApplicantFeatures.model_fields)
    if fields != bank.preprocessing.feature_order:
        raise RuntimeError(
            f"ApplicantFeatures fields {fields} do not match preprocessing.json "
            f"feature_order {bank.preprocessing.feature_order}"
        )

    app = FastAPI(title="Bank service")
    app.state.bank = bank
    quiet_validation_errors(app)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse()

    @app.post("/apply", response_model=ApplyResponse)
    def apply(req: ApplyRequest) -> ApplyResponse:
        try:
            result = bank.score(req.national_id, req.features.model_dump())
        except ScorerError as exc:
            log.info("apply failed pseudonym=%s reason=scorer", bank.pseudonym(req.national_id))
            raise HTTPException(502, str(exc)) from None
        return ApplyResponse(**result.__dict__)

    return app


configure_logger("bank_service")


def __getattr__(name: str):
    # `uvicorn bank_service.main:app` builds the app (and the keys) on first
    # access, so importing this module in tests has no side effects.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)

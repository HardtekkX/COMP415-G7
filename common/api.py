"""FastAPI helpers shared by the bank and the scoring service."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse


def quiet_validation_errors(app: FastAPI) -> None:
    """Answer 422 with field locations only.

    FastAPI's default handler echoes the offending input (a national ID, a
    feature value, a ciphertext) back in the response.
    """

    @app.exception_handler(RequestValidationError)
    async def _handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = [".".join(str(p) for p in err["loc"]) for err in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": "invalid request", "fields": fields})


def configure_logger(name: str) -> logging.Logger:
    """INFO logger with its own handler; uvicorn only configures its own loggers."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    return logger

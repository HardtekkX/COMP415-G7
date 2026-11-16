"""Request and response models shared by the bank and the scoring service.

Binary values travel as standard base64 strings (PROJECT_PLAN.md, "API").
Inside Python they are bytes: requests decode on validation, responses encode
on serialisation.
"""

from __future__ import annotations

import base64
import binascii
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer, WithJsonSchema

# HMAC-SHA256(bank_key, national_id).hexdigest(): 64 lowercase hex characters.
# Anything else (a name, a raw national ID) is rejected before it reaches the scorer.
PSEUDONYM_PATTERN = r"^[0-9a-f]{64}$"
# secrets.token_hex(16), issued by POST /contexts.
CONTEXT_ID_PATTERN = r"^[0-9a-f]{32}$"


def _b64decode(value: Any) -> bytes:
    if isinstance(value, bytes):  # built in Python; JSON input is always str
        return value
    if not isinstance(value, str):
        raise ValueError("must be a base64 string")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("invalid base64") from None


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


Base64Bytes = Annotated[
    bytes,
    BeforeValidator(_b64decode),
    PlainSerializer(b64encode, return_type=str),
    WithJsonSchema({"type": "string", "contentEncoding": "base64"}),
]
Pseudonym = Annotated[str, Field(pattern=PSEUDONYM_PATTERN)]
ContextId = Annotated[str, Field(pattern=CONTEXT_ID_PATTERN)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ContextRequest(_Model):
    """POST /contexts: the bank's public CKKS context, without the secret key."""

    context: Base64Bytes


class ContextResponse(_Model):
    context_id: ContextId


class ScoreRequest(_Model):
    """POST /score: one encrypted, scaled feature vector."""

    pseudonym: Pseudonym
    context_id: ContextId
    ciphertext: Base64Bytes


class ScoreResponse(_Model):
    pseudonym: Pseudonym
    encrypted_logit: Base64Bytes


class HealthResponse(_Model):
    status: str = "ok"


# --- Scorer demo endpoints (GET /activity, GET /contexts/summary) -------------
# Only what the scorer may log (pseudonyms, context IDs, sizes, timings,
# timestamps) plus short base64 prefixes of the ciphertexts it handled. Never
# plaintext features, national IDs or decrypted values. extra="forbid" keeps a
# new field from slipping in without a schema change.

ACTIVITY_PREFIX_CHARS = 64


class ActivityRecord(_Model):
    timestamp: str
    pseudonym: Pseudonym
    context_id: ContextId
    ciphertext_bytes: int
    ciphertext_prefix: Annotated[str, Field(max_length=ACTIVITY_PREFIX_CHARS)]
    result_prefix: Annotated[str, Field(max_length=ACTIVITY_PREFIX_CHARS)]
    compute_ms: float


class ActivityResponse(_Model):
    records: list[ActivityRecord]


class ContextInfo(_Model):
    context_id: ContextId
    size_bytes: int
    registered_at: str


class ContextSummary(_Model):
    count: int
    max_contexts: int
    contexts: list[ContextInfo]

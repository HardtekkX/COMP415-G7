"""Bank API models (POST /apply). Plaintext data: these never go to the scorer."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

# 11 digits, like a Turkish T.C. kimlik number. Tests and the CLI use IDs that
# start with 0, which no real T.C. number does.
NationalId = Annotated[str, Field(pattern=r"^[0-9]{11}$")]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ApplicantFeatures(_Model):
    """Field names must equal preprocessing.json's feature_order (checked at startup)."""

    duration_months: int = Field(ge=1, le=120)
    credit_amount: int = Field(ge=1, le=1_000_000)
    installment_rate: int = Field(ge=1, le=4)
    existing_credits: int = Field(ge=0, le=10)
    checking_status: int = Field(ge=0, le=3)
    savings_band: int = Field(ge=0, le=4)
    employment_band: int = Field(ge=0, le=4)
    other_installment_plans: int = Field(ge=0, le=1)
    owns_home: int = Field(ge=0, le=1)
    has_guarantor: int = Field(ge=0, le=1)


class ApplyRequest(_Model):
    national_id: NationalId
    features: ApplicantFeatures


class ApplyResponse(_Model):
    pseudonym: str
    probability_of_default: float
    decision: Literal["approve", "review", "reject"]
    logit: float
    ciphertext_bytes: int
    encrypted_logit_bytes: int
    timings_ms: dict[str, float]

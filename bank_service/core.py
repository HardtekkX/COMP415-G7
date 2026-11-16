"""Bank-side scoring: scale, pseudonymise, encrypt, call the scorer, decrypt, decide.

The bank holds the HE secret key, the HMAC key and the scaler. It never reads
the model weights (training/artifacts/model.json belongs to the scorer).
"""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np

from bank_service import keys as keystore
from bank_service.config import BankConfig, Decision, Thresholds
from bank_service.scorer_client import ScorerClient
from common import he

log = logging.getLogger("bank_service")


@dataclass(frozen=True)
class Preprocessing:
    feature_order: list[str]
    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def load(cls, path: Path) -> Preprocessing:
        data = json.loads(Path(path).read_text())
        order = list(data["feature_order"])
        mean, scale = np.array(data["scaler_mean"], float), np.array(data["scaler_scale"], float)
        if not len(order) == len(mean) == len(scale):
            raise ValueError(f"{path}: feature_order, scaler_mean and scaler_scale differ in length")
        return cls(order, mean, scale)

    @property
    def n_features(self) -> int:
        return len(self.feature_order)

    def transform(self, features: Mapping[str, float]) -> np.ndarray:
        """Scale named features into a vector in the exported feature order."""
        if set(features) != set(self.feature_order):
            raise ValueError("features do not match feature_order")
        raw = np.array([float(features[name]) for name in self.feature_order])
        return (raw - self.mean) / self.scale


def sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def make_pseudonym(hmac_key: bytes, national_id: str) -> str:
    return hmac.new(hmac_key, national_id.encode("utf-8"), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class ScoreResult:
    pseudonym: str
    probability_of_default: float
    decision: Decision
    logit: float
    ciphertext_bytes: int
    encrypted_logit_bytes: int
    timings_ms: dict[str, float]


class Bank:
    def __init__(
        self,
        preprocessing: Preprocessing,
        keys: keystore.BankKeys,
        scorer: ScorerClient,
        thresholds: Thresholds,
    ):
        self.preprocessing = preprocessing
        self.keys = keys
        self.scorer = scorer
        self.thresholds = thresholds

    @classmethod
    def from_config(cls, config: BankConfig, http: httpx.Client | None = None) -> Bank:
        pre = Preprocessing.load(config.preprocessing_path)
        keys = keystore.load_or_create(config.keys_dir, pre.n_features)
        keystore.check_pair(keys, pre.n_features)
        scorer = ScorerClient(
            config.scorer_url, keys.public_context, keys.scorer_context_path, http=http
        )
        return cls(pre, keys, scorer, config.thresholds)

    def pseudonym(self, national_id: str) -> str:
        return make_pseudonym(self.keys.hmac_key, national_id)

    def score(self, national_id: str, features: Mapping[str, float]) -> ScoreResult:
        timings: dict[str, float] = {}
        start = t = time.perf_counter()

        def lap(name: str) -> None:
            nonlocal t
            now = time.perf_counter()
            timings[name] = round((now - t) * 1000, 2)
            t = now

        pseudonym = self.pseudonym(national_id)
        x = self.preprocessing.transform(features)
        lap("prepare")
        ciphertext = he.encrypt_vector(self.keys.he_context, x)
        lap("encrypt")
        encrypted = self.scorer.score(pseudonym, ciphertext)
        lap("scorer_call")
        logit = he.decrypt_logit(self.keys.he_context, encrypted)
        probability = sigmoid(logit)
        decision = self.thresholds.decide(probability)
        lap("decrypt")
        timings["total"] = round((time.perf_counter() - start) * 1000, 2)

        # Pseudonym, decision, sizes and timings only: no national ID, features,
        # logit or probability.
        log.info(
            "apply pseudonym=%s decision=%s ciphertext_bytes=%d logit_bytes=%d "
            "encrypt_ms=%.1f scorer_ms=%.1f total_ms=%.1f",
            pseudonym, decision, len(ciphertext), len(encrypted),
            timings["encrypt"], timings["scorer_call"], timings["total"],
        )
        return ScoreResult(
            pseudonym=pseudonym,
            probability_of_default=probability,
            decision=decision,
            logit=logit,
            ciphertext_bytes=len(ciphertext),
            encrypted_logit_bytes=len(encrypted),
            timings_ms=timings,
        )

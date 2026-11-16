"""Bank configuration: bank_service/config.toml plus environment overrides."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "config.toml"

Decision = Literal["approve", "review", "reject"]


@dataclass(frozen=True)
class Thresholds:
    approve_below: float
    reject_from: float

    def __post_init__(self) -> None:
        if not 0.0 < self.approve_below < self.reject_from <= 1.0:
            raise ValueError("need 0 < approve_below < reject_from <= 1")

    def decide(self, probability_of_default: float) -> Decision:
        if probability_of_default < self.approve_below:
            return "approve"
        if probability_of_default >= self.reject_from:
            return "reject"
        return "review"


@dataclass(frozen=True)
class BankConfig:
    scorer_url: str
    keys_dir: Path
    preprocessing_path: Path
    thresholds: Thresholds

    @classmethod
    def load(cls, path: Path | str | None = None) -> BankConfig:
        path = Path(path or os.environ.get("BANK_CONFIG", DEFAULT_CONFIG_PATH))
        data = tomllib.loads(path.read_text())
        decision = data["decision"]
        return cls(
            scorer_url=os.environ.get("BANK_SCORER_URL", data["scorer_url"]).rstrip("/"),
            keys_dir=_resolve(os.environ.get("BANK_KEYS_DIR", data["keys_dir"])),
            preprocessing_path=_resolve(data["preprocessing_path"]),
            thresholds=Thresholds(
                approve_below=float(decision["approve_below"]),
                reject_from=float(decision["reject_from"]),
            ),
        )


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p

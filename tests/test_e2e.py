"""End to end: both services as separate uvicorn processes talking over HTTP.

Scores test applicants through the bank's POST /apply and compares with the
plaintext probability that training stored in test_split.csv (the bank never
computes it from weights). Needs `python -m training.train` to have been run.
"""

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
import pytest

from bank_service.cli import fake_national_id
from bank_service.config import REPO_ROOT, Thresholds

TEST_SPLIT = REPO_ROOT / "training" / "artifacts" / "test_split.csv"
N_APPLICANTS = 20
TOLERANCE = 1e-3

pytestmark = pytest.mark.skipif(
    not TEST_SPLIT.exists(), reason="test_split.csv missing; run python -m training.train"
)


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Service:
    def __init__(self, app: str, port: int, env: dict, log_path: Path):
        self.app, self.port, self.env, self.log_path = app, port, env, log_path
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 60) -> None:
        log = self.log_path.open("a")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", self.app, "--host", "127.0.0.1", "--port", str(self.port)],
            cwd=REPO_ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT,
        )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                pytest.fail(f"{self.app} exited:\n{self.log_path.read_text()}")
            try:
                if httpx.get(self.url + "/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                time.sleep(0.2)
        pytest.fail(f"{self.app} did not start:\n{self.log_path.read_text()}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            self.proc.wait(timeout=10)

    def log(self) -> str:
        return self.log_path.read_text()


@pytest.fixture(scope="module")
def services(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("e2e")
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    scorer = Service("scoring_service.main:app", free_port(), env, tmp / "scorer.log")
    bank_env = {**env, "BANK_SCORER_URL": scorer.url, "BANK_KEYS_DIR": str(tmp / "keys")}
    bank = Service("bank_service.main:app", free_port(), bank_env, tmp / "bank.log")
    scorer.start()
    bank.start()
    yield scorer, bank
    bank.stop()
    scorer.stop()


@pytest.fixture(scope="module")
def applicants():
    rows = pd.read_csv(TEST_SPLIT).head(N_APPLICANTS)
    assert len(rows) == N_APPLICANTS
    return rows


def features_of(row) -> dict:
    names = [c for c in row.index if c not in {"row_id", "default", "plaintext_probability"}]
    return {name: int(row[name]) for name in names}


def apply(bank, row) -> dict:
    resp = httpx.post(
        bank.url + "/apply",
        json={"national_id": fake_national_id(int(row["row_id"])), "features": features_of(row)},
        timeout=60,
    )
    assert resp.status_code == 200, resp.text
    return resp.json()


def test_encrypted_probability_matches_plaintext(services, applicants):
    scorer, bank = services
    thresholds = Thresholds(0.25, 0.50)
    errors = []
    for _, row in applicants.iterrows():
        body = apply(bank, row)
        plaintext = float(row["plaintext_probability"])
        errors.append(abs(body["probability_of_default"] - plaintext))
        assert body["probability_of_default"] == pytest.approx(plaintext, abs=TOLERANCE), row["row_id"]
        assert body["decision"] == thresholds.decide(plaintext), row["row_id"]
    print(f"\n{N_APPLICANTS} applicants, max |encrypted - plaintext| = {max(errors):.2e}")

    # Registered once, then reused for every applicant.
    assert scorer.log().count("context registered") == 1
    assert scorer.log().count("INFO scoring_service score ") == N_APPLICANTS


def test_logs_hold_no_national_ids_or_features(services, applicants):
    scorer, bank = services
    for service in services:
        text = service.log()
        for _, row in applicants.iterrows():
            assert fake_national_id(int(row["row_id"])) not in text
        assert "credit_amount" not in text and "duration_months" not in text
    for line in bank.log().splitlines():
        if " bank_service apply " in line:
            assert re.search(r"apply pseudonym=[0-9a-f]{64} decision=\w+ ciphertext_bytes=\d+ ", line)


def test_bank_reregisters_after_scorer_restart(services, applicants):
    scorer, bank = services
    scorer.stop()
    scorer.start()                               # empty context store
    body = apply(bank, applicants.iloc[0])
    assert body["probability_of_default"] == pytest.approx(
        float(applicants.iloc[0]["plaintext_probability"]), abs=TOLERANCE
    )
    assert "registering again" in bank.log()


def test_cli_scores_one_applicant(services, applicants):
    scorer, bank = services
    row = applicants.iloc[1]
    result = subprocess.run(
        [sys.executable, "-m", "bank_service.cli", "--row-id", str(int(row["row_id"]))],
        cwd=REPO_ROOT, env=bank.env, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stderr
    encrypted = float(re.search(r"encrypted probability\s+([\d.]+)", result.stdout).group(1))
    assert encrypted == pytest.approx(float(row["plaintext_probability"]), abs=TOLERANCE)
    assert fake_national_id(int(row["row_id"])) in result.stdout   # shown to the operator only

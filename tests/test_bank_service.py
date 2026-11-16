import ast
import builtins
import hashlib
import hmac
import io
import json
import logging
import re
import stat
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bank_service import keys as keystore
from bank_service.config import DEFAULT_CONFIG_PATH, REPO_ROOT, BankConfig, Thresholds
from bank_service.core import Bank, Preprocessing, make_pseudonym, sigmoid
from bank_service.main import create_app as create_bank_app
from common import he
from scoring_service.main import create_app as create_scorer_app

PREPROCESSING_PATH = REPO_ROOT / "training" / "artifacts" / "preprocessing.json"
PRE = Preprocessing.load(PREPROCESSING_PATH)

# Clearly fake: no real T.C. kimlik number starts with 0.
FAKE_ID = "00000000001"
OTHER_FAKE_ID = "00000000002"
FEATURES = {
    "duration_months": 24,
    "credit_amount": 3500,
    "installment_rate": 3,
    "existing_credits": 1,
    "checking_status": 1,
    "savings_band": 0,
    "employment_band": 2,
    "other_installment_plans": 0,
    "owns_home": 1,
    "has_guarantor": 0,
}

ALLOWED_BANK_LOG_LINES = [
    r"context registered context_id=[0-9a-f]{32} size_bytes=\d+",
    r"context unknown to scorer context_id=[0-9a-f]{32}; registering again",
    r"apply pseudonym=[0-9a-f]{64} decision=(approve|review|reject) ciphertext_bytes=\d+ "
    r"logit_bytes=\d+ encrypt_ms=[\d.]+ scorer_ms=[\d.]+ total_ms=[\d.]+",
    r"apply failed pseudonym=[0-9a-f]{64} reason=scorer",
]


@pytest.fixture
def scorer():
    """The scoring service in-process; TestClient is an httpx.Client."""
    return TestClient(create_scorer_app())


@pytest.fixture
def config(tmp_path):
    return BankConfig(
        scorer_url="http://testserver",
        keys_dir=tmp_path / "keys",
        preprocessing_path=PREPROCESSING_PATH,
        thresholds=Thresholds(approve_below=0.25, reject_from=0.50),
    )


@pytest.fixture
def bank(config, scorer):
    return Bank.from_config(config, http=scorer)


@pytest.fixture
def bank_api(config, scorer):
    return TestClient(create_bank_app(config, http=scorer))


# --- keys ------------------------------------------------------------------

def test_first_run_creates_private_key_files(bank, config):
    assert stat.S_IMODE(config.keys_dir.stat().st_mode) == 0o700
    for name in ("he_context.seal", "public_context.seal", "hmac.key", "meta.json"):
        assert stat.S_IMODE((config.keys_dir / name).stat().st_mode) == 0o600, name
    assert len(bank.keys.hmac_key) == 32


def test_keys_are_reused_not_regenerated(bank, config, scorer):
    ct = he.encrypt_vector(bank.keys.he_context, [0.5] * PRE.n_features)
    again = Bank.from_config(config, http=scorer)
    assert again.keys.hmac_key == bank.keys.hmac_key
    assert again.keys.public_context == bank.keys.public_context
    assert again.pseudonym(FAKE_ID) == bank.pseudonym(FAKE_ID)
    assert he.load_ciphertext(again.keys.he_context, ct).decrypt() == pytest.approx(
        [0.5] * PRE.n_features, abs=1e-6
    )


def test_public_context_has_no_secret_key(bank):
    assert not he.load_public_context(bank.keys.public_context).is_private()
    assert len(bank.keys.public_context) < 4_000_000  # minimal Galois keys, not the 17 MB set


def test_incomplete_keys_dir_is_refused(bank, config):
    (config.keys_dir / "hmac.key").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        keystore.load_or_create(config.keys_dir, PRE.n_features)


def test_keys_for_other_feature_count_are_refused(bank, config):
    with pytest.raises(RuntimeError, match="delete the folder"):
        keystore.load_or_create(config.keys_dir, PRE.n_features + 1)


def test_default_keys_dir_is_git_ignored():
    cfg = BankConfig.load(DEFAULT_CONFIG_PATH)
    for name in ("hmac.key", "he_context.seal", "scorer_context.json"):
        path = (cfg.keys_dir / name).relative_to(REPO_ROOT)
        result = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO_ROOT)
        assert result.returncode == 0, f"{path} is not git-ignored"


# --- pseudonym, scaling, decision -------------------------------------------

def test_pseudonym_is_hmac_sha256_of_national_id(bank):
    expected = hmac.new(bank.keys.hmac_key, FAKE_ID.encode(), hashlib.sha256).hexdigest()
    assert bank.pseudonym(FAKE_ID) == expected
    assert bank.pseudonym(OTHER_FAKE_ID) != expected
    assert make_pseudonym(b"\x00" * 32, FAKE_ID) != expected


def test_scaling_uses_exported_order():
    shuffled = dict(reversed(list(FEATURES.items())))
    raw = np.array([FEATURES[name] for name in PRE.feature_order], float)
    assert PRE.transform(shuffled) == pytest.approx((raw - PRE.mean) / PRE.scale)


def test_scaling_rejects_missing_or_extra_features():
    with pytest.raises(ValueError):
        PRE.transform({k: v for k, v in FEATURES.items() if k != "owns_home"})
    with pytest.raises(ValueError):
        PRE.transform({**FEATURES, "age": 40})


@pytest.mark.parametrize(
    "p, decision",
    [(0.0, "approve"), (0.2499, "approve"), (0.25, "review"), (0.4999, "review"),
     (0.50, "reject"), (1.0, "reject")],
)
def test_decision_bands(p, decision):
    assert Thresholds(0.25, 0.50).decide(p) == decision


def test_config_file_thresholds():
    data = tomllib.loads(DEFAULT_CONFIG_PATH.read_text())
    assert data["decision"] == {"approve_below": 0.25, "reject_from": 0.50}


@pytest.mark.parametrize("approve, reject", [(0.5, 0.5), (0.6, 0.4), (0.0, 0.5), (0.2, 1.1)])
def test_bad_thresholds_rejected(approve, reject):
    with pytest.raises(ValueError):
        Thresholds(approve, reject)


def test_sigmoid_is_stable():
    assert sigmoid(0) == 0.5
    assert sigmoid(-1000) == 0.0 and sigmoid(1000) == 1.0


# --- scoring through the scorer ---------------------------------------------

def test_score_returns_probability_and_decision(bank):
    result = bank.score(FAKE_ID, FEATURES)
    assert 0 < result.probability_of_default < 1
    assert result.probability_of_default == pytest.approx(sigmoid(result.logit))
    assert result.decision == bank.thresholds.decide(result.probability_of_default)
    assert result.pseudonym == bank.pseudonym(FAKE_ID)


def test_context_id_is_cached_across_restarts(bank, config, scorer):
    bank.score(FAKE_ID, FEATURES)
    assert len(scorer.app.state.contexts) == 1
    Bank.from_config(config, http=scorer).score(FAKE_ID, FEATURES)
    assert len(scorer.app.state.contexts) == 1


def test_reregisters_after_404(bank, config, scorer, caplog):
    caplog.set_level(logging.INFO, logger="bank_service")
    bank.score(FAKE_ID, FEATURES)
    old_id = json.loads((config.keys_dir / "scorer_context.json").read_text())["context_id"]
    scorer.app.state.contexts._items.clear()     # as if the scorer restarted

    bank.score(FAKE_ID, FEATURES)
    new_id = json.loads((config.keys_dir / "scorer_context.json").read_text())["context_id"]
    assert new_id != old_id
    assert f"context unknown to scorer context_id={old_id}" in caplog.text


def test_scorer_unreachable_is_502(config):
    import httpx

    def refuse(request):
        raise httpx.ConnectError("refused")

    app = create_bank_app(config, http=httpx.Client(transport=httpx.MockTransport(refuse)))
    resp = TestClient(app).post("/apply", json={"national_id": FAKE_ID, "features": FEATURES})
    assert resp.status_code == 502


# --- HTTP API ----------------------------------------------------------------

def test_health(bank_api):
    assert bank_api.get("/health").json() == {"status": "ok"}


def test_apply(bank_api):
    resp = bank_api.post("/apply", json={"national_id": FAKE_ID, "features": FEATURES})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] in {"approve", "review", "reject"}
    assert FAKE_ID not in resp.text
    assert set(body["timings_ms"]) == {"prepare", "encrypt", "scorer_call", "decrypt", "total"}


@pytest.mark.parametrize(
    "national_id, features, field",
    [
        ("0000000001", FEATURES, "body.national_id"),            # 10 digits
        ("0000000000A", FEATURES, "body.national_id"),
        (FAKE_ID, {**FEATURES, "checking_status": 4}, "body.features.checking_status"),
        (FAKE_ID, {**FEATURES, "savings_band": -1}, "body.features.savings_band"),
        (FAKE_ID, {**FEATURES, "age": 40}, "body.features.age"),
    ],
)
def test_apply_rejects_bad_input_without_echo(bank_api, national_id, features, field):
    resp = bank_api.post("/apply", json={"national_id": national_id, "features": features})
    assert resp.status_code == 422
    assert resp.json()["fields"] == [field]
    assert national_id not in resp.text


def test_bank_logs_no_national_id_or_features(bank_api, scorer, caplog):
    caplog.set_level(logging.INFO, logger="bank_service")
    bank_api.post("/apply", json={"national_id": FAKE_ID, "features": FEATURES})
    scorer.app.state.contexts._items.clear()
    bank_api.post("/apply", json={"national_id": OTHER_FAKE_ID, "features": FEATURES})

    lines = [r.getMessage() for r in caplog.records if r.name == "bank_service"]
    assert any(line.startswith("apply ") for line in lines)
    for line in lines:
        assert any(re.fullmatch(p, line) for p in ALLOWED_BANK_LOG_LINES), line
    assert FAKE_ID not in caplog.text and OTHER_FAKE_ID not in caplog.text
    assert not any(name in caplog.text for name in FEATURES)


# --- the bank never reads the model weights ----------------------------------

FORBIDDEN_BANK_NAMES = {"weights", "bias", "MODEL_PATH", "DEFAULT_MODEL_PATH", "coef_", "intercept_"}


def _bank_sources() -> list[Path]:
    return sorted((REPO_ROOT / "bank_service").glob("*.py"))


def _docstrings(tree: ast.AST) -> set[int]:
    ids = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = node.body[0] if node.body else None
            if isinstance(doc, ast.Expr) and isinstance(doc.value, ast.Constant):
                ids.add(id(doc.value))
    return ids


@pytest.mark.parametrize("path", _bank_sources(), ids=lambda p: p.name)
def test_bank_code_never_opens_model_json(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    docstrings = _docstrings(tree)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            modules = []
        for module in modules:
            assert not module.startswith(("scoring_service", "training")), (
                f"{path.name}:{node.lineno} imports {module}"
            )
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
            assert "model.json" not in node.value, f"{path.name}:{node.lineno}"
            assert "model" != node.value.strip("/"), f"{path.name}:{node.lineno}"
        if isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_BANK_NAMES, f"{path.name}:{node.lineno} .{node.attr}"
        if isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_BANK_NAMES, f"{path.name}:{node.lineno} {node.id}"


def test_bank_config_never_points_at_model_json():
    def values(d):
        for v in d.values():
            yield from values(v) if isinstance(v, dict) else [v]

    data = tomllib.loads(DEFAULT_CONFIG_PATH.read_text())
    assert not any("model.json" in str(v) for v in values(data))


def test_bank_never_opens_model_json_at_runtime(config, scorer, monkeypatch):
    """Score an applicant with every file open checked; the scorer app already exists."""
    opened = []
    real_open = io.open

    def guarded_open(file, *args, **kwargs):
        opened.append(str(file))
        if str(file).endswith("model.json"):
            raise AssertionError(f"bank opened {file}")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(io, "open", guarded_open)
    monkeypatch.setattr(builtins, "open", guarded_open)
    Bank.from_config(config, http=scorer).score(FAKE_ID, FEATURES)
    assert any(p.endswith("preprocessing.json") for p in opened)

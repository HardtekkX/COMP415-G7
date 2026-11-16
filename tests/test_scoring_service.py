import ast
import base64
import hashlib
import hmac
import json
import logging
import re
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from common import he
from scoring_service.main import DEFAULT_MODEL_PATH, create_app

ROOT = Path(__file__).resolve().parent.parent
MODEL = json.loads(DEFAULT_MODEL_PATH.read_text())
N = len(MODEL["weights"])
W, B = np.array(MODEL["weights"]), MODEL["bias"]

PSEUDONYM = hmac.new(b"test-bank-key", b"12345678901", hashlib.sha256).hexdigest()
UNKNOWN_CONTEXT_ID = "0" * 32

# Every line the service may log: pseudonyms, context IDs, sizes, timings, status.
ALLOWED_LOG_LINES = [
    r"context registered context_id=[0-9a-f]{32} size_bytes=\d+ ms=[\d.]+",
    r"context rejected size_bytes=\d+ status=\d{3}",
    r"context evicted context_id=[0-9a-f]{32}",
    r"score pseudonym=[0-9a-f]{64} context_id=[0-9a-f]{32} ciphertext_bytes=\d+ "
    r"logit_bytes=\d+ ms=[\d.]+",
    r"score rejected pseudonym=[0-9a-f]{64} context_id=[0-9a-f]{32} "
    r"ciphertext_bytes=\d+ status=\d{3}",
]


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


@pytest.fixture(scope="module")
def bank():
    """The bank's side: private context and serialised public context."""
    return he.make_contexts(N)


@pytest.fixture
def client():
    return TestClient(create_app())


@pytest.fixture
def context_id(client, bank):
    resp = client.post("/contexts", json={"context": b64(bank[1])})
    assert resp.status_code == 200
    return resp.json()["context_id"]


def scaled_row(seed: int = 0) -> np.ndarray:
    """A standardised feature vector, as the bank would send (the scorer has no scaler)."""
    return np.random.default_rng(seed).normal(size=N)


def score(client, context_id, ciphertext: bytes, pseudonym=PSEUDONYM):
    return client.post(
        "/score",
        json={"pseudonym": pseudonym, "context_id": context_id, "ciphertext": b64(ciphertext)},
    )


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_score_matches_plaintext_logit(client, bank, context_id):
    bank_ctx = bank[0]
    for seed in range(3):
        x = scaled_row(seed)
        resp = score(client, context_id, he.encrypt_vector(bank_ctx, x))
        assert resp.status_code == 200
        body = resp.json()
        assert body["pseudonym"] == PSEUDONYM
        logit = he.decrypt_logit(bank_ctx, base64.b64decode(body["encrypted_logit"]))
        assert logit == pytest.approx(W @ x + B, abs=1e-4)


def test_returns_single_summed_logit_not_products(client, bank, context_id):
    resp = score(client, context_id, he.encrypt_vector(bank[0], scaled_row()))
    out = he.load_ciphertext(bank[0], base64.b64decode(resp.json()["encrypted_logit"]))
    assert out.size() == 1


def test_unknown_context_id_is_404(client, bank):
    resp = score(client, UNKNOWN_CONTEXT_ID, he.encrypt_vector(bank[0], scaled_row()))
    assert resp.status_code == 404


@pytest.mark.parametrize("length", [N - 1, N + 1])
def test_wrong_vector_length_is_rejected(client, bank, context_id, length):
    resp = score(client, context_id, he.encrypt_vector(bank[0], np.ones(length)))
    assert resp.status_code == 422
    assert str(N) in resp.json()["detail"]


def test_garbage_ciphertext_is_400(client, context_id):
    assert score(client, context_id, b"not a ciphertext").status_code == 400


def test_context_with_secret_key_is_refused(client, bank):
    private = bank[0].serialize(save_secret_key=True)
    resp = client.post("/contexts", json={"context": b64(private)})
    assert resp.status_code == 400
    assert len(client.app.state.contexts) == 0


def test_garbage_context_is_400(client):
    assert client.post("/contexts", json={"context": b64(b"junk")}).status_code == 400


@pytest.mark.parametrize(
    "field, value",
    [
        ("pseudonym", "Ayse Yilmaz"),              # a name instead of a pseudonym
        ("pseudonym", "12345678901"),              # a raw national ID
        ("pseudonym", PSEUDONYM.upper()),
        ("context_id", "../etc"),
        ("ciphertext", "not base64!"),
    ],
)
def test_malformed_score_request_is_422_without_echo(client, bank, context_id, field, value):
    body = {
        "pseudonym": PSEUDONYM,
        "context_id": context_id,
        "ciphertext": b64(he.encrypt_vector(bank[0], scaled_row())),
    }
    body[field] = value
    resp = client.post("/score", json=body)
    assert resp.status_code == 422
    assert value not in resp.text
    assert resp.json()["fields"] == [f"body.{field}"]


def test_extra_fields_are_rejected(client, bank, context_id):
    body = {
        "pseudonym": PSEUDONYM,
        "context_id": context_id,
        "ciphertext": b64(he.encrypt_vector(bank[0], scaled_row())),
        "national_id": "12345678901",
    }
    resp = client.post("/score", json=body)
    assert resp.status_code == 422
    assert "12345678901" not in resp.text


def test_context_store_evicts_oldest(bank):
    client = TestClient(create_app(max_contexts=2))
    ids = [
        client.post("/contexts", json={"context": b64(bank[1])}).json()["context_id"]
        for _ in range(3)
    ]
    assert len(client.app.state.contexts) == 2
    ct = he.encrypt_vector(bank[0], scaled_row())
    assert score(client, ids[0], ct).status_code == 404
    assert score(client, ids[1], ct).status_code == 200
    assert score(client, ids[2], ct).status_code == 200


def test_logs_only_allowed_fields(client, bank, caplog):
    caplog.set_level(logging.INFO, logger="scoring_service")
    x = scaled_row()
    ct = he.encrypt_vector(bank[0], x)

    context_id = client.post("/contexts", json={"context": b64(bank[1])}).json()["context_id"]
    score(client, context_id, ct)                                  # ok
    score(client, UNKNOWN_CONTEXT_ID, ct)                          # 404
    score(client, context_id, he.encrypt_vector(bank[0], np.ones(N + 1)))  # 422
    client.post("/contexts", json={"context": b64(b"junk")})      # 400

    lines = [r.getMessage() for r in caplog.records if r.name == "scoring_service"]
    assert len(lines) == 5
    for line in lines:
        assert any(re.fullmatch(p, line) for p in ALLOWED_LOG_LINES), line
    text = "\n".join(lines)
    assert PSEUDONYM in text and context_id in text
    assert b64(ct)[:32] not in text
    assert not any(f"{v:.3f}" in text for v in x)


# --- The scoring service never touches a secret key ---------------------------

FORBIDDEN_NAMES = {
    "secret_key",       # TenSEAL / SEAL accessor
    "save_secret_key",
    "decrypt",
    "decrypt_logit",
    "make_contexts",    # creates the key pair: bank only
    "encrypt_vector",
    "make_context_private",
    "KeyGenerator",
    "Decryptor",
}


def _scoring_sources() -> list[Path]:
    return sorted((ROOT / "scoring_service").rglob("*.py"))


@pytest.mark.parametrize("path", _scoring_sources(), ids=lambda p: p.name)
def test_service_code_never_uses_secret_key(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
            names = {a.name for a in node.names}
            assert not names & FORBIDDEN_NAMES, f"{path.name}:{node.lineno} imports {names}"
        else:
            modules = []
        for module in modules:
            assert not module.startswith("bank_service"), f"{path.name}:{node.lineno}"
            assert "sealapi" not in module, f"{path.name}:{node.lineno}"
        if isinstance(node, ast.Attribute):
            assert node.attr not in FORBIDDEN_NAMES, f"{path.name}:{node.lineno} uses .{node.attr}"
        if isinstance(node, ast.Name):
            assert node.id not in FORBIDDEN_NAMES, f"{path.name}:{node.lineno} uses {node.id}"
        if isinstance(node, ast.keyword) and node.arg:
            assert node.arg not in FORBIDDEN_NAMES, f"{path.name}:{node.lineno} passes {node.arg}="


def test_stored_contexts_hold_no_secret_key(client, bank, context_id):
    contexts = client.app.state.contexts.contexts()
    assert contexts
    for ctx in contexts:
        assert not ctx.is_private()
        with pytest.raises(Exception):
            ctx.secret_key()


def test_model_file_holds_only_weights_and_bias():
    assert set(MODEL) == {"weights", "bias"}


# --- Demo endpoints: GET /activity, GET /contexts/summary ---------------------

ACTIVITY_FIELDS = {
    "timestamp": r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d{3}\+00:00",
    "pseudonym": r"[0-9a-f]{64}",
    "context_id": r"[0-9a-f]{32}",
    "ciphertext_bytes": r"\d+",
    "ciphertext_prefix": r"[A-Za-z0-9+/]{64}",
    "result_prefix": r"[A-Za-z0-9+/]{64}",
    "compute_ms": r"[\d.]+(e-?\d+)?",
}


def other_pseudonym(i: int) -> str:
    return hmac.new(b"test-bank-key", f"{i:011d}".encode(), hashlib.sha256).hexdigest()


def test_activity_starts_empty(client):
    resp = client.get("/activity")
    assert resp.status_code == 200
    assert resp.json() == {"records": []}


def test_activity_records_only_allowed_fields(client, bank, context_id):
    x = scaled_row()
    ct = he.encrypt_vector(bank[0], x)
    resp = score(client, context_id, ct)
    encrypted_logit = resp.json()["encrypted_logit"]

    records = client.get("/activity").json()["records"]
    assert len(records) == 1
    rec = records[0]
    assert set(rec) == set(ACTIVITY_FIELDS)
    for key, pattern in ACTIVITY_FIELDS.items():
        assert re.fullmatch(pattern, str(rec[key])), (key, rec[key])
    assert rec["pseudonym"] == PSEUDONYM
    assert rec["context_id"] == context_id
    assert rec["ciphertext_bytes"] == len(ct)
    assert rec["ciphertext_prefix"] == b64(ct)[:64]
    assert rec["result_prefix"] == encrypted_logit[:64]


def test_activity_never_holds_plaintext_or_decrypted_values(client, bank, context_id):
    bank_ctx = bank[0]
    x = scaled_row(7)
    resp = score(client, context_id, he.encrypt_vector(bank_ctx, x))
    logit = he.decrypt_logit(bank_ctx, base64.b64decode(resp.json()["encrypted_logit"]))
    probability = 1 / (1 + np.exp(-logit))

    text = client.get("/activity").text
    for value in [*x, logit, probability]:
        for fmt in ("{:.3f}", "{:.4f}", "{!r}"):
            assert fmt.format(float(value)) not in text
    assert "12345678901" not in text   # national ID behind PSEUDONYM


def test_activity_keeps_last_50_newest_first(client, bank, context_id):
    ct = he.encrypt_vector(bank[0], scaled_row())
    for i in range(55):
        assert score(client, context_id, ct, pseudonym=other_pseudonym(i)).status_code == 200
    records = client.get("/activity").json()["records"]
    assert len(records) == 50
    assert [r["pseudonym"] for r in records] == [other_pseudonym(i) for i in range(54, 4, -1)]


def test_rejected_scores_are_not_recorded(client, bank, context_id):
    score(client, UNKNOWN_CONTEXT_ID, he.encrypt_vector(bank[0], scaled_row()))   # 404
    score(client, context_id, he.encrypt_vector(bank[0], np.ones(N + 1)))       # 422
    score(client, context_id, b"not a ciphertext")                              # 400
    assert client.get("/activity").json() == {"records": []}


def test_contexts_summary(bank):
    client = TestClient(create_app(max_contexts=2))
    assert client.get("/contexts/summary").json() == {"count": 0, "max_contexts": 2, "contexts": []}

    ids = [
        client.post("/contexts", json={"context": b64(bank[1])}).json()["context_id"]
        for _ in range(3)
    ]
    summary = client.get("/contexts/summary").json()
    assert summary["count"] == 2
    assert summary["max_contexts"] == 2
    assert [c["context_id"] for c in summary["contexts"]] == ids[1:]   # oldest evicted
    for c in summary["contexts"]:
        assert set(c) == {"context_id", "size_bytes", "registered_at"}
        assert c["size_bytes"] == len(bank[1])


def test_demo_endpoints_off_return_404(bank, monkeypatch):
    monkeypatch.setenv("SCORER_DEMO_ENDPOINTS", "false")
    client = TestClient(create_app())
    context_id = client.post("/contexts", json={"context": b64(bank[1])}).json()["context_id"]
    assert score(client, context_id, he.encrypt_vector(bank[0], scaled_row())).status_code == 200
    assert client.get("/activity").status_code == 404
    assert client.get("/contexts/summary").status_code == 404
    assert client.get("/health").status_code == 200

"""Scorer page smoke tests: Streamlit AppTest with a mocked HTTP layer (no services)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

import ui.client

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "ui" / "app.py")
PAGE = "views/scorer.py"

CTX = "c0ffee" + "0" * 26


def record(i: int) -> dict:
    return {
        "timestamp": f"2026-11-16T19:00:{i:02d}.123+00:00",
        "pseudonym": f"{i:02x}" * 32,
        "context_id": CTX,
        "ciphertext_bytes": 235_000 + i,
        "ciphertext_slice": f"CT{i}" + "A" * 50,
        "ciphertext_sha256": f"{i:x}" * 16,
        "result_slice": f"RS{i}" + "B" * 50,
        "result_sha256": f"{i + 8:x}" * 16,
        "compute_ms": 4.0 + i / 10,
    }


class FakeScorer:
    def __init__(self):
        self.records = [record(3), record(2), record(1)]     # newest first, as served
        self.contexts = [{"context_id": CTX, "size_bytes": 2_912_345,
                          "registered_at": "2026-11-16T18:59:00.000+00:00"}]
        self.status = 200
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        assert request.method == "GET"
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "Not Found"})
        if request.url.path == "/activity":
            return httpx.Response(200, json={"records": self.records})
        if request.url.path == "/contexts/summary":
            return httpx.Response(200, json={"count": len(self.contexts), "max_contexts": 100,
                                             "contexts": self.contexts})
        raise AssertionError(f"unexpected request {request.url}")


@pytest.fixture
def fake_scorer(monkeypatch) -> FakeScorer:
    fake = FakeScorer()
    monkeypatch.setattr(ui.client, "is_up", lambda s: True)
    monkeypatch.setattr(ui.client, "_TRANSPORT", httpx.MockTransport(fake))
    return fake


def open_page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    return at


def page_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption] + [c.value for c in at.code]
    parts += [e.value for e in at.error] + [w.value for w in at.warning] + [i.value for i in at.info]
    return "\n".join(parts)


def test_table_lists_requests_newest_first(fake_scorer):
    at = open_page()
    [table] = at.dataframe
    df = table.value
    assert list(df.columns) == ["Time", "Pseudonym", "Context ID", "Ciphertext", "Ciphertext SHA-256",
                                "Compute (ms)"]
    assert list(df["Ciphertext SHA-256"]) == [r["ciphertext_sha256"] for r in fake_scorer.records]
    assert list(df["Pseudonym"]) == [r["pseudonym"][:16] + "…" for r in fake_scorer.records]
    assert list(df["Context ID"]) == [CTX] * 3
    assert list(df["Ciphertext"]) == ["235.0 KB"] * 3
    assert list(df["Compute (ms)"]) == [4.3, 4.2, 4.1]
    assert all(t.endswith(".123") for t in df["Time"])


def test_newest_request_slices_shown_by_default(fake_scorer):
    at = open_page()
    codes = [c.value for c in at.code]
    assert codes == [record(3)["ciphertext_slice"], record(3)["result_slice"]]
    text = page_text(at)
    assert f"SHA-256 `{record(3)['ciphertext_sha256']}…`" in text
    assert f"SHA-256 `{record(3)['result_sha256']}…`" in text
    assert "Newest request" in page_text(at)
    assert f"Pseudonym `{record(3)['pseudonym']}`" in page_text(at)


def test_summary_line(fake_scorer):
    text = page_text(open_page())
    assert "**1 public context registered** (max 100) · 2.9 MB" in text
    assert "no secret key" in text


def test_fixed_note_lists_what_the_scorer_lacks(fake_scorer):
    text = page_text(open_page()).lower()
    for item in ["names", "national ids", "feature values", "the score", "the decision"]:
        assert item in text, item


def test_refresh_fetches_again(fake_scorer):
    at = open_page()
    before = fake_scorer.paths.count("/activity")
    fake_scorer.records.insert(0, record(4))
    at.button(key="scorer_refresh").click().run()
    assert fake_scorer.paths.count("/activity") == before + 1
    assert at.dataframe[0].value["Pseudonym"].iloc[0] == record(4)["pseudonym"][:16] + "…"
    assert at.code[0].value == record(4)["ciphertext_slice"]


def test_empty_log_points_to_bank_page(fake_scorer):
    fake_scorer.records = []
    fake_scorer.contexts = []
    at = open_page()
    text = page_text(at)
    assert "No public context registered yet" in text
    assert "Score an applicant on the **Bank** page" in text
    assert not at.dataframe and not at.code


def test_demo_endpoints_off_says_how_to_turn_them_on(fake_scorer):
    fake_scorer.status = 404
    at = open_page()
    assert len(at.warning) == 1 and not at.error
    msg = at.warning[0].value
    assert "demo endpoints are turned off" in msg
    assert "SCORER_DEMO_ENDPOINTS=false" in msg
    assert ui.client.SCORER.start_command in msg
    assert not at.dataframe


def test_other_error_is_a_message_not_a_traceback(fake_scorer):
    fake_scorer.status = 500
    at = open_page()
    assert "returned HTTP 500" in at.error[0].value
    assert "Traceback" not in at.error[0].value


def test_never_calls_the_bank_or_shows_demo_plaintext(fake_scorer):
    at = open_page()
    assert set(fake_scorer.paths) == {"/activity", "/contexts/summary"}
    path = ui.client.DEMO_APPLICANTS_PATH
    if path.exists():
        text = page_text(at)
        for a in json.loads(path.read_text())["applicants"]:
            assert a["national_id"] not in text and a["label"] not in text
            assert f"{a['features']['credit_amount']:,}" not in text

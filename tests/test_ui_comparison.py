"""Comparison page smoke tests: Streamlit AppTest with a mocked HTTP layer (no services)."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
from streamlit.testing.v1 import AppTest

import ui.client

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "ui" / "app.py")
PAGE = "views/comparison.py"
DEMO = json.loads(ui.client.DEMO_APPLICANTS_PATH.read_text()) if ui.client.DEMO_APPLICANTS_PATH.exists() else None

pytestmark = pytest.mark.skipif(DEMO is None, reason="needs ui/demo_applicants.json (python -m ui.demo_applicants)")

OFFSET = 1.5e-7
TIMINGS = {"prepare": 0.1, "encrypt": 4.0, "scorer_call": 13.0, "decrypt": 0.5, "total": 17.6}


class FakeBank:
    """POST /apply: the plaintext probability shifted by OFFSET, as CKKS would."""

    def __init__(self):
        self.requests: list[dict] = []
        self.flip: str | None = None      # national_id whose decision comes back wrong
        self.status = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/apply"
        body = json.loads(request.content)
        self.requests.append(body)
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "error"})
        a = next(a for a in DEMO["applicants"] if a["national_id"] == body["national_id"])
        decision = a["expected_decision"]
        if body["national_id"] == self.flip:
            decision = "review" if decision != "review" else "approve"
        return httpx.Response(200, json={
            "pseudonym": "ab" * 32,
            "probability_of_default": a["plaintext_probability"] - OFFSET,
            "decision": decision,
            "logit": 0.0,
            "ciphertext_bytes": 235_123,
            "encrypted_logit_bytes": 117_456,
            "timings_ms": TIMINGS,
        })


@pytest.fixture
def fake_bank(monkeypatch) -> FakeBank:
    fake = FakeBank()
    monkeypatch.setattr(ui.client, "is_up", lambda s: True)
    monkeypatch.setattr(ui.client, "_TRANSPORT", httpx.MockTransport(fake))
    return fake


def open_page() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=15).run()
    at.switch_page(PAGE).run()
    assert not at.exception, at.exception
    return at


def compare(at: AppTest, index: int = 0) -> AppTest:
    at.selectbox(key="cmp_applicant").set_value(index).run()
    at.button(key="cmp_compare").click().run()
    assert not at.exception, at.exception
    return at


def metrics(at: AppTest) -> dict[str, str]:
    return {m.label: m.value for m in at.metric}


def text(at: AppTest) -> str:
    return "\n".join([m.value for m in at.markdown] + [c.value for c in at.caption])


def test_page_starts_empty(fake_bank):
    at = open_page()
    assert "press **Compare**" in at.info[0].value
    assert not at.metric and not fake_bank.requests


def test_compare_shows_both_probabilities_difference_and_match(fake_bank):
    a = DEMO["applicants"][2]
    at = compare(open_page(), 2)
    assert fake_bank.requests == [{"national_id": a["national_id"], "features": a["features"]}]
    m = metrics(at)
    assert m["Plaintext probability"] == f"{a['plaintext_probability']:.9f}"
    assert m["Encrypted-path probability"] == f"{a['plaintext_probability'] - OFFSET:.9f}"
    assert m["Absolute difference"] == "1.50e-07"
    assert m["Decisions match"] == f"✅ Yes: {a['expected_decision']}"


def test_mismatch_is_shown(fake_bank):
    a = DEMO["applicants"][0]
    fake_bank.flip = a["national_id"]
    at = compare(open_page(), 0)
    assert metrics(at)["Decisions match"].startswith("❌ No: approve vs review")


def test_request_size_and_timing_chart(fake_bank):
    a = DEMO["applicants"][0]
    at = compare(open_page(), 0)
    plain = len(json.dumps({"features": a["features"]}, separators=(",", ":")))
    t = text(at)
    assert f"| Plain JSON (the ten features) | {plain:,} bytes |" in t
    assert "| CKKS ciphertext | 235,123 bytes |" in t
    assert f"about {235_123 / plain:,.0f}× larger" in t
    [chart] = at.get("vega_lite_chart")
    spec = json.loads(chart.proto.spec)
    mark = spec["mark"]
    assert (mark["type"] if isinstance(mark, dict) else mark) == "bar"
    assert "Total 17.6 ms" in t


def test_switching_applicant_clears_the_result(fake_bank):
    at = compare(open_page(), 0)
    at.selectbox(key="cmp_applicant").set_value(1).run()
    assert not at.metric


def test_run_all_six_fills_one_row_per_applicant(fake_bank):
    at = open_page()
    at.button(key="cmp_run_all").click().run()
    assert not at.exception, at.exception
    assert [r["national_id"] for r in fake_bank.requests] == [a["national_id"] for a in DEMO["applicants"]]
    df = at.dataframe[0].value
    assert list(df["Applicant"]) == [a["label"] for a in DEMO["applicants"]]
    assert list(df["Match"]) == ["✅"] * 6
    assert list(df["|Difference|"]) == ["1.50e-07"] * 6
    assert list(df["Plaintext decision"]) == [a["expected_decision"] for a in DEMO["applicants"]]
    assert "**Largest difference 1.50e-07** · **6/6 decisions match**" in text(at)


def test_run_all_counts_a_mismatch(fake_bank):
    fake_bank.flip = DEMO["applicants"][4]["national_id"]
    at = open_page()
    at.button(key="cmp_run_all").click().run()
    assert at.dataframe[0].value["Match"].iloc[4] == "❌"
    assert "**5/6 decisions match**" in text(at)


@pytest.mark.parametrize("button", ["cmp_compare", "cmp_run_all"])
def test_bank_error_is_a_message_not_a_traceback(fake_bank, button):
    fake_bank.status = 502
    at = open_page()
    at.button(key=button).click().run()
    assert not at.exception
    assert len(at.error) == 1 and "could not get a score" in at.error[0].value
    assert not at.metric and not at.dataframe

"""Bank page smoke tests: Streamlit AppTest with a mocked HTTP layer (no services)."""

from __future__ import annotations

import json
from pathlib import Path

import annotated_types
import httpx
import pytest
from streamlit.testing.v1 import AppTest

import ui.client
from ui.features import FEATURES

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "ui" / "app.py")
PAGE = "views/bank.py"
DEMO = json.loads(ui.client.DEMO_APPLICANTS_PATH.read_text()) if ui.client.DEMO_APPLICANTS_PATH.exists() else None

pytestmark = pytest.mark.skipif(DEMO is None, reason="needs ui/demo_applicants.json (python -m ui.demo_applicants)")

PSEUDONYM = "ab" * 32
NORMAL_TIMINGS = {"prepare": 0.1, "encrypt": 4.0, "scorer_call": 13.0, "decrypt": 0.5, "total": 17.6}


class FakeBank:
    """Answers POST /apply like the bank would; records every request body."""

    def __init__(self):
        self.requests: list[dict] = []
        self.status = 200
        self.raise_exc: Exception | None = None
        self.body: dict | None = None            # overrides the normal 200 body
        self.scorer_ms: list[float] = []         # per-call scorer_call timings, then 13 ms

    def __call__(self, request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/apply"
        body = json.loads(request.content)
        self.requests.append(body)
        if self.raise_exc:
            raise self.raise_exc
        if self.status != 200:
            return httpx.Response(self.status, json=self.body or {"detail": "error"})
        if self.body is not None:
            return httpx.Response(200, json=self.body)
        applicant = next(
            (a for a in DEMO["applicants"] if a["features"] == body["features"]), DEMO["applicants"][0]
        )
        timings = dict(NORMAL_TIMINGS)
        if self.scorer_ms:
            timings["scorer_call"] = self.scorer_ms.pop(0)
        return httpx.Response(200, json={
            "pseudonym": PSEUDONYM,
            "probability_of_default": applicant["plaintext_probability"],
            "decision": applicant["expected_decision"],
            "logit": 0.0,
            "ciphertext_bytes": 235_123,
            "encrypted_logit_bytes": 117_456,
            "timings_ms": timings,
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


def submit(at: AppTest) -> AppTest:
    [button] = [b for b in at.button if b.key and b.key.startswith("bank_submit_")]
    button.click().run()
    assert not at.exception, at.exception
    return at


def pick(at: AppTest, index: int) -> AppTest:
    at.selectbox(key="bank_applicant").set_value(index).run()
    return at


def field(at: AppTest, name: str):
    """The form widget for a feature name, or "national_id", of the selected applicant."""
    index = at.session_state["bank_applicant"]
    if name == "national_id":
        return at.text_input(key=f"bank_national_id_{index}")
    f = next(f for f in FEATURES if f.name == name)
    return (at.selectbox if f.options else at.number_input)(key=f"bank_f_{name}_{index}")


def page_text(at: AppTest) -> str:
    parts = [m.value for m in at.markdown] + [c.value for c in at.caption]
    parts += [e.value for e in at.error] + [i.value for i in at.info]
    return "\n".join(parts)


def sent_markdown(at: AppTest) -> str:
    [sent] = [m.value for m in at.markdown if "**Pseudonym**" in m.value]
    return sent


# --- Happy path ---------------------------------------------------------------------

def test_page_starts_with_first_applicant_and_no_result(fake_bank):
    at = open_page()
    first = DEMO["applicants"][0]
    assert field(at, "national_id").value == first["national_id"]
    assert "Request score" in at.info[0].value
    assert not fake_bank.requests


@pytest.mark.parametrize("decision", ["approve", "review", "reject"])
def test_each_decision_renders_label_probability_and_what_left(fake_bank, decision):
    index = next(i for i, a in enumerate(DEMO["applicants"]) if a["expected_decision"] == decision)
    applicant = DEMO["applicants"][index]
    at = submit(pick(open_page(), index))

    assert fake_bank.requests == [{"national_id": applicant["national_id"], "features": applicant["features"]}]
    color, word = {"approve": ("green", "Approve"), "review": ("orange", "Review"),
                   "reject": ("red", "Reject")}[decision]
    text = page_text(at)
    assert f":{color}-background[" in text and word in text
    assert at.metric[0].value == f"{applicant['plaintext_probability']:.2%}"
    assert "Approve** below 25%" in text and "Reject** from 50%" in text
    assert f"Result for **{applicant['label']}**" in text

    # What left the bank: pseudonym, sizes, timings; never the ID or the features.
    sent = sent_markdown(at)
    assert PSEUDONYM in sent and "235.1 KB" in sent and "117.5 KB" in sent
    assert applicant["national_id"] not in sent
    for f in FEATURES:
        assert f"| {f.label} |" not in sent
    assert f"{applicant['features']['credit_amount']:,}" not in sent
    assert "Not sent: stayed in the bank" in text
    assert f"National ID **{applicant['national_id']}**" in text
    assert "| Scorer call (network + encrypted dot product) | 13.0 ms |" in text
    assert "| Total | 17.6 ms |" in text
    assert not at.error
    assert "one-time key registration" not in text


def test_picking_an_applicant_fills_the_form(fake_bank):
    at = pick(open_page(), 2)
    applicant = DEMO["applicants"][2]
    assert field(at, "national_id").value == applicant["national_id"]
    for f in FEATURES:
        assert field(at, f.name).value == applicant["features"][f.name], f.name


def test_edited_fields_are_sent(fake_bank):
    at = open_page()
    field(at, "credit_amount").set_value(5000)
    field(at, "owns_home").set_value(0)
    submit(at)
    sent = fake_bank.requests[0]["features"]
    assert sent["credit_amount"] == 5000 and sent["owns_home"] == 0
    assert "Result for **Edited applicant**" in page_text(at)


def test_form_keeps_values_after_requests(fake_bank):
    at = open_page()
    field(at, "credit_amount").set_value(5000)
    submit(at)
    submit(at)
    assert field(at, "credit_amount").value == 5000
    assert field(at, "national_id").value == DEMO["applicants"][0]["national_id"]
    assert [r["features"]["credit_amount"] for r in fake_bank.requests] == [5000, 5000]


def test_switching_applicant_clears_the_result(fake_bank):
    at = submit(open_page())
    assert at.metric
    pick(at, 3)
    assert not at.metric
    assert "Request score" in at.info[0].value


# --- Double submit and spinner state ----------------------------------------------

def test_one_click_sends_one_request_and_reenables_the_button(fake_bank):
    at = submit(open_page())
    assert len(fake_bank.requests) == 1
    [button] = [b for b in at.button if b.key and b.key.startswith("bank_submit_")]
    assert button.key == "bank_submit_1"      # new generation: old-button clicks are ignored
    assert not button.disabled
    assert not at.session_state["bank_busy"]


def test_button_and_applicant_picker_disabled_while_busy(fake_bank):
    at = open_page()
    at.session_state["bank_busy"] = True       # as during an in-flight request
    at.run()
    [button] = [b for b in at.button if b.key and b.key.startswith("bank_submit_")]
    assert button.disabled and button.label == "Scoring…"
    assert at.selectbox(key="bank_applicant").disabled
    assert not field(at, "credit_amount").disabled   # toggling would reset edits
    assert not fake_bank.requests


# --- Slow first call: key registration note --------------------------------------

def test_slow_scorer_call_shows_registration_note_once(fake_bank):
    fake_bank.scorer_ms = [144.0, 12.5]
    at = submit(open_page())
    note = " ".join(i.value for i in at.info)
    assert "one-time key registration" in note and "144 ms" in note
    submit(at)
    assert "one-time key registration" not in page_text(at)


def test_modest_slowdown_is_not_flagged(fake_bank):
    fake_bank.scorer_ms = [12.0, 13.0, 40.0]   # 40 ms: about 3x, but only 27 ms more
    at = open_page()
    for _ in range(3):
        submit(at)
    assert "one-time key registration" not in page_text(at)


# --- Errors: readable message, no stack trace -----------------------------------------

@pytest.mark.parametrize(
    "setup, expected",
    [
        (lambda f: setattr(f, "raise_exc", httpx.ConnectError("refused")), "Bank service is not reachable"),
        (lambda f: setattr(f, "raise_exc", httpx.ReadTimeout("slow")), "did not answer within"),
        (lambda f: setattr(f, "status", 502), "could not get a score from the scoring service"),
        (lambda f: (setattr(f, "status", 422),
                    setattr(f, "body", {"detail": "invalid request",
                                        "fields": ["body.features.credit_amount"]})),
         "invalid:** Credit amount (DM)"),
        (lambda f: setattr(f, "status", 500), "returned HTTP 500"),
        (lambda f: setattr(f, "body", {"unexpected": True}), "does not understand"),
    ],
    ids=["connect", "timeout", "502", "422", "500", "bad-body"],
)
def test_errors_show_a_message_not_a_traceback(fake_bank, setup, expected):
    setup(fake_bank)
    at = submit(open_page())
    assert len(at.error) == 1 and expected in at.error[0].value
    assert "Traceback" not in at.error[0].value
    assert not at.metric
    assert not at.session_state["bank_busy"]
    assert len(fake_bank.requests) == 1


def test_502_names_the_scorer_start_command(fake_bank):
    fake_bank.status = 502
    at = submit(open_page())
    assert ui.client.SCORER.start_command in at.error[0].value


def test_invalid_national_id_sends_nothing(fake_bank):
    at = open_page()
    field(at, "national_id").input("12345")
    submit(at)
    assert "exactly 11 digits" in at.error[0].value
    assert not fake_bank.requests


# --- ui/features.py stays in step with the bank's schema and the training codes -------

def test_features_match_bank_schema_and_feature_order():
    from bank_service.schemas import ApplicantFeatures

    order = json.loads((ROOT / "training/artifacts/preprocessing.json").read_text())["feature_order"]
    assert [f.name for f in FEATURES] == list(ApplicantFeatures.model_fields) == order
    for f in FEATURES:
        meta = ApplicantFeatures.model_fields[f.name].metadata
        ge = next(m.ge for m in meta if isinstance(m, annotated_types.Ge))
        le = next(m.le for m in meta if isinstance(m, annotated_types.Le))
        assert (f.min, f.max) == (ge, le), f.name
        if f.options:
            assert sorted(f.options) == list(range(f.min, f.max + 1)), f.name


def test_category_codes_match_training():
    from training import train

    by_name = {f.name: f for f in FEATURES}
    for name, mapping in [("checking_status", train.CHECKING), ("savings_band", train.SAVINGS),
                          ("employment_band", train.EMPLOYMENT)]:
        assert sorted(by_name[name].options) == sorted(mapping.values()), name
    # Spot-check the meaning of the reordered codes (ordered by default rate).
    assert train.SAVINGS["A65"] == 2 and "Unknown" in by_name["savings_band"].options[2]
    assert train.CHECKING["A14"] == 3 and "No checking" in by_name["checking_status"].options[3]
    assert train.CHECKING["A11"] == 0 and "Overdrawn" in by_name["checking_status"].options[0]

"""HTTP helpers for the Streamlit screens.

The UI is a thin client: it talks to the bank and the scoring service over
HTTP only and never imports bank_service or scoring_service code (enforced by
tests/test_ui.py).

Env:  UI_BANK_URL    bank service   (default http://127.0.0.1:8000)
      UI_SCORER_URL  scoring service (default http://127.0.0.1:8001)
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import httpx
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMO_APPLICANTS_PATH = REPO_ROOT / "ui" / "demo_applicants.json"
TEST_SPLIT_PATH = REPO_ROOT / "training" / "artifacts" / "test_split.csv"

DEMO_APPLICANTS_COMMAND = "python -m ui.demo_applicants"
SETUP_COMMAND = "python scripts/setup_demo.py"
HEALTH_TIMEOUT_S = 2.0


@dataclass(frozen=True)
class Service:
    name: str
    env_var: str
    default_url: str
    start_command: str

    @property
    def url(self) -> str:
        return os.environ.get(self.env_var, self.default_url).rstrip("/")


BANK = Service(
    "Bank service", "UI_BANK_URL", "http://127.0.0.1:8000",
    "uvicorn bank_service.main:app --port 8000",
)
SCORER = Service(
    "Scoring service", "UI_SCORER_URL", "http://127.0.0.1:8001",
    "uvicorn scoring_service.main:app --port 8001",
)


def is_up(service: Service) -> bool:
    try:
        resp = httpx.get(f"{service.url}/health", timeout=HEALTH_TIMEOUT_S)
    except httpx.HTTPError:
        return False
    return resp.status_code == 200 and resp.json().get("status") == "ok"


def require_service(service: Service) -> None:
    """Show a clear error and stop the page if `service` does not answer /health."""
    if is_up(service):
        return
    st.error(
        f"**{service.name} is not reachable at {service.url}.**\n\n"
        f"Start it from the repository root (with `.venv` activated):\n\n"
        f"```\n{service.start_command}\n```\n"
        f"Or point the UI elsewhere with `{service.env_var}`."
    )
    st.stop()


def require_demo_applicants() -> list[dict]:
    """The six demo applicants, or show the command that creates them and stop."""
    return require_demo_data()["applicants"]


def require_demo_data() -> dict:
    """Load ui/demo_applicants.json ({thresholds, applicants}), or show the
    command that creates it and stop."""
    if not DEMO_APPLICANTS_PATH.exists():
        if TEST_SPLIT_PATH.exists():
            command = DEMO_APPLICANTS_COMMAND
        else:
            command = SETUP_COMMAND + "   # trains the model, then picks the applicants"
        st.error(
            "**Demo applicants not found** (`ui/demo_applicants.json`).\n\n"
            f"Create them from the repository root:\n\n```\n{command}\n```"
        )
        st.stop()
    return json.loads(DEMO_APPLICANTS_PATH.read_text())


# --- POST /apply on the bank -----------------------------------------------------

APPLY_TIMEOUT_S = 60.0
# Tests set this to an httpx.MockTransport; None means real HTTP.
_TRANSPORT: httpx.BaseTransport | None = None


APPLY_RESPONSE_FIELDS = {
    "pseudonym", "probability_of_default", "decision", "ciphertext_bytes",
    "encrypted_logit_bytes", "timings_ms",
}


class ApplyError(Exception):
    """POST /apply failed. The message is safe to show on screen as Markdown."""


def _field_label(loc: str) -> str:
    from ui.features import BY_NAME  # local import: features.py is UI data only

    name = loc.rsplit(".", 1)[-1]
    if name == "national_id":
        return "National ID"
    return BY_NAME[name].label if name in BY_NAME else loc


def apply(national_id: str, features: dict[str, int]) -> dict:
    """Score one applicant through the bank. Returns the ApplyResponse JSON.

    Raises ApplyError with a readable message for every failure, never with
    the request body (it holds the national ID and the features).
    """
    body = {"national_id": national_id, "features": features}
    try:
        with httpx.Client(transport=_TRANSPORT, timeout=APPLY_TIMEOUT_S) as http:
            resp = http.post(f"{BANK.url}/apply", json=body)
    except httpx.TimeoutException:
        raise ApplyError(
            f"**The bank service did not answer within {APPLY_TIMEOUT_S:.0f} s** ({BANK.url})."
        ) from None
    except httpx.HTTPError:
        raise ApplyError(
            f"**Bank service is not reachable at {BANK.url}.**\n\n"
            f"Start it from the repository root:\n\n```\n{BANK.start_command}\n```"
        ) from None

    if resp.status_code == 200:
        try:
            data = resp.json()
        except ValueError:
            data = None
        if not isinstance(data, dict) or not APPLY_RESPONSE_FIELDS <= data.keys():
            raise ApplyError("**The bank service sent a response the UI does not understand.**")
        return data
    if resp.status_code == 502:
        raise ApplyError(
            f"**The bank could not get a score from the scoring service.**\n\n"
            f"Check that it is running:\n\n```\n{SCORER.start_command}\n```"
        )
    if resp.status_code == 422:
        try:
            fields = resp.json().get("fields", [])
        except ValueError:
            fields = []
        names = ", ".join(_field_label(f) for f in fields) or "unknown fields"
        raise ApplyError(f"**The bank rejected the application as invalid:** {names}.")
    raise ApplyError(f"**The bank service returned HTTP {resp.status_code}.**")


# --- Scorer demo endpoints: GET /activity, GET /contexts/summary ----------------

SCORER_TIMEOUT_S = 10.0
DEMO_ENDPOINTS_ENV = "SCORER_DEMO_ENDPOINTS"


class ScorerDemoError(Exception):
    """A scorer demo endpoint failed. The message is safe to show as Markdown.
    `off` is True when the scorer answered 404 (demo endpoints turned off)."""

    def __init__(self, message: str, off: bool = False):
        super().__init__(message)
        self.off = off


def _scorer_get(path: str, field: str) -> dict:
    try:
        with httpx.Client(transport=_TRANSPORT, timeout=SCORER_TIMEOUT_S) as http:
            resp = http.get(f"{SCORER.url}{path}")
    except httpx.HTTPError:
        raise ScorerDemoError(
            f"**Scoring service is not reachable at {SCORER.url}.**\n\n"
            f"Start it from the repository root:\n\n```\n{SCORER.start_command}\n```"
        ) from None
    if resp.status_code == 404:
        raise ScorerDemoError(
            f"**The scorer's demo endpoints are turned off** (`GET {path}` answered 404).\n\n"
            f"The scoring service was started with `{DEMO_ENDPOINTS_ENV}=false`. Restart it "
            f"without that setting (they are on by default) to see this screen:\n\n"
            f"```\n{SCORER.start_command}\n```",
            off=True,
        )
    if resp.status_code != 200:
        raise ScorerDemoError(f"**The scoring service returned HTTP {resp.status_code}** for `{path}`.")
    try:
        data = resp.json()
    except ValueError:
        data = None
    if not isinstance(data, dict) or field not in data:
        raise ScorerDemoError("**The scoring service sent a response the UI does not understand.**")
    return data


def scorer_activity() -> list[dict]:
    """GET /activity: the scorer's last successful scores, newest first."""
    return _scorer_get("/activity", "records")["records"]


def scorer_contexts_summary() -> dict:
    """GET /contexts/summary: {count, max_contexts, contexts: [...]}."""
    return _scorer_get("/contexts/summary", "contexts")

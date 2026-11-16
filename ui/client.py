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
    """Load ui/demo_applicants.json, or show the command that creates it and stop."""
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
    return json.loads(DEMO_APPLICANTS_PATH.read_text())["applicants"]

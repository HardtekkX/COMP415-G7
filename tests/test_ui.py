"""Streamlit UI: thin-client rule, page errors, demo applicant picker."""

import ast
import json
import socket
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import ui.client
from ui import demo_applicants

ROOT = Path(__file__).resolve().parent.parent
APP = str(ROOT / "ui" / "app.py")


# --- The UI talks HTTP only -------------------------------------------------

def _ui_sources() -> list[Path]:
    return sorted((ROOT / "ui").rglob("*.py"))


@pytest.mark.parametrize("path", _ui_sources(), ids=lambda p: str(p.relative_to(ROOT)))
def test_ui_never_imports_service_code(path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [node.module or ""]
        else:
            continue
        for module in modules:
            assert not module.startswith(("bank_service", "scoring_service")), (
                f"{path.name}:{node.lineno} imports {module}"
            )


# --- Pages show a clear error when a service is down --------------------------

@pytest.fixture
def services_down(monkeypatch):
    with socket.socket() as s:            # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    monkeypatch.setenv("UI_BANK_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setenv("UI_SCORER_URL", f"http://127.0.0.1:{port}")
    return f"http://127.0.0.1:{port}"


def run_page(page: str | None = None) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=15).run()
    if page:
        at.switch_page(page).run()
    assert not at.exception, at.exception
    return at


def test_home_renders_and_reports_services_down(services_down):
    at = run_page()
    assert at.title[0].value == "Encrypted credit scoring"
    errors = " ".join(e.value for e in at.error)
    assert "Bank service: not reachable" in errors
    assert "Scoring service: not reachable" in errors


@pytest.mark.parametrize(
    "page, service",
    [
        ("views/bank.py", "Bank service"),
        ("views/scorer.py", "Scoring service"),
        ("views/comparison.py", "Bank service"),
    ],
)
def test_page_errors_when_its_service_is_down(services_down, page, service):
    at = run_page(page)
    assert len(at.error) == 1
    msg = at.error[0].value
    assert f"{service} is not reachable at {services_down}" in msg
    assert "uvicorn" in msg
    assert not at.info                      # the page stopped before its content


def test_comparison_needs_the_scorer_too(services_down, monkeypatch):
    monkeypatch.setattr(ui.client, "is_up", lambda s: s is ui.client.BANK)
    at = run_page("views/comparison.py")
    assert "Scoring service is not reachable" in at.error[0].value


@pytest.mark.parametrize("page", ["views/bank.py", "views/comparison.py"])
@pytest.mark.parametrize("trained", [True, False])
def test_page_says_how_to_create_missing_demo_applicants(monkeypatch, tmp_path, page, trained):
    monkeypatch.setattr(ui.client, "is_up", lambda s: True)
    monkeypatch.setattr(ui.client, "DEMO_APPLICANTS_PATH", tmp_path / "missing.json")
    split = tmp_path / "test_split.csv"
    if trained:
        split.write_text("row_id\n")
    monkeypatch.setattr(ui.client, "TEST_SPLIT_PATH", split)

    at = run_page(page)
    msg = at.error[0].value
    assert "Demo applicants not found" in msg
    expected = ui.client.DEMO_APPLICANTS_COMMAND if trained else ui.client.SETUP_COMMAND
    assert expected in msg


def test_scorer_page_needs_no_demo_applicants(monkeypatch, tmp_path):
    monkeypatch.setattr(ui.client, "is_up", lambda s: True)
    monkeypatch.setattr(ui.client, "DEMO_APPLICANTS_PATH", tmp_path / "missing.json")
    at = run_page("views/scorer.py")
    assert not at.error


# --- demo_applicants.py --------------------------------------------------------

FEATURES = ["a", "b"]


def write_inputs(tmp_path: Path, probabilities: list[float]) -> dict:
    split = tmp_path / "test_split.csv"
    lines = ["row_id,a,b,default,plaintext_probability"]
    lines += [f"{i},{i % 5},{i % 2},0,{p}" for i, p in enumerate(probabilities)]
    split.write_text("\n".join(lines) + "\n")
    pre = tmp_path / "preprocessing.json"
    pre.write_text(json.dumps({"feature_order": FEATURES}))
    cfg = tmp_path / "config.toml"
    cfg.write_text("[decision]\napprove_below = 0.25\nreject_from = 0.50\n")
    return {"test_split": split, "preprocessing": pre, "config": cfg, "out": tmp_path / "out.json"}


def run_picker(paths: dict) -> int:
    return demo_applicants.main([
        "--test-split", str(paths["test_split"]), "--preprocessing", str(paths["preprocessing"]),
        "--config", str(paths["config"]), "--out", str(paths["out"]),
    ])


def test_picks_two_per_decision_clear_of_thresholds(tmp_path):
    probs = [i / 100 for i in range(1, 100)]
    paths = write_inputs(tmp_path, probs)
    assert run_picker(paths) == 0
    data = json.loads(paths["out"].read_text())
    apps = data["applicants"]

    assert [a["expected_decision"] for a in apps] == ["approve"] * 2 + ["review"] * 2 + ["reject"] * 2
    assert len({a["row_id"] for a in apps}) == 6
    assert len({a["national_id"] for a in apps}) == 6
    for a in apps:
        p = a["plaintext_probability"]
        assert demo_applicants.decide(p, 0.25, 0.50) == a["expected_decision"]
        assert abs(p - 0.25) >= demo_applicants.MARGIN and abs(p - 0.50) >= demo_applicants.MARGIN
        assert len(a["national_id"]) == 11 and a["national_id"].isdigit()
        assert a["national_id"].startswith("0")            # never a real T.C. number
        assert list(a["features"]) == FEATURES
    assert data["thresholds"] == {"approve_below": 0.25, "reject_from": 0.50}

    assert run_picker(paths) == 0                              # deterministic
    assert json.loads(paths["out"].read_text()) == data


def test_too_few_rows_in_a_band_fails(tmp_path):
    paths = write_inputs(tmp_path, [0.05, 0.1, 0.3, 0.4, 0.6])  # one reject row
    with pytest.raises(ValueError, match="reject"):
        run_picker(paths)


def test_missing_test_split_tells_user_to_train(tmp_path, capsys):
    paths = write_inputs(tmp_path, [0.1])
    paths["test_split"].unlink()
    assert run_picker(paths) == 1
    err = capsys.readouterr().err
    assert "python -m training.train" in err
    assert "python scripts/setup_demo.py" in err
    assert not paths["out"].exists()

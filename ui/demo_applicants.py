"""Pick six demo applicants from the test split: two approve, two review, two reject.

Run:  python -m ui.demo_applicants
Reads training/artifacts/test_split.csv (written by `python -m training.train`),
the decision thresholds from bank_service/config.toml (read as data, not code)
and the feature order from training/artifacts/preprocessing.json. Writes
ui/demo_applicants.json (git-ignored; regenerate after retraining).

Rows within MARGIN of a threshold are skipped, so the small CKKS error can
never flip a demo decision. In each band the rows nearest the 25th and 75th
percentile of probability are chosen; the choice is deterministic. National
IDs are made up: 11 digits starting with 0, which no real T.C. number does.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import statistics
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_SPLIT_PATH = REPO_ROOT / "training" / "artifacts" / "test_split.csv"
PREPROCESSING_PATH = REPO_ROOT / "training" / "artifacts" / "preprocessing.json"
CONFIG_PATH = REPO_ROOT / "bank_service" / "config.toml"
OUT_PATH = REPO_ROOT / "ui" / "demo_applicants.json"

DECISIONS = ("approve", "review", "reject")
PER_DECISION = 2
MARGIN = 0.02


def load_thresholds(config_path: Path) -> tuple[float, float]:
    decision = tomllib.loads(config_path.read_text())["decision"]
    return float(decision["approve_below"]), float(decision["reject_from"])


def decide(p: float, approve_below: float, reject_from: float) -> str:
    # Same rule as the bank (bank_service/config.py, Thresholds.decide).
    if p < approve_below:
        return "approve"
    if p >= reject_from:
        return "reject"
    return "review"


def fake_national_id(row_id: int) -> str:
    return "0" + f"{random.Random(row_id).randrange(10**10):010d}"


def pick(rows: list[dict], approve_below: float, reject_from: float) -> list[dict]:
    """Two rows per decision band, clear of the thresholds by MARGIN."""
    chosen = []
    for decision in DECISIONS:
        band = sorted(
            (
                r for r in rows
                if decide(r["p"], approve_below, reject_from) == decision
                and abs(r["p"] - approve_below) >= MARGIN
                and abs(r["p"] - reject_from) >= MARGIN
            ),
            key=lambda r: (r["p"], r["row_id"]),
        )
        if len(band) < PER_DECISION:
            raise ValueError(f"only {len(band)} '{decision}' rows clear of the thresholds")
        q1, _, q3 = statistics.quantiles([r["p"] for r in band], n=4)
        picked: list[dict] = []
        for target in (q1, q3):
            rest = [r for r in band if r not in picked]
            picked.append(min(rest, key=lambda r: (abs(r["p"] - target), r["row_id"])))
        chosen.extend((decision, r) for r in sorted(picked, key=lambda r: r["p"]))
    return [
        {
            "label": f"Applicant {chr(ord('A') + i)}",
            "row_id": r["row_id"],
            "national_id": fake_national_id(r["row_id"]),
            "features": r["features"],
            "expected_decision": decision,
            "plaintext_probability": r["p"],
        }
        for i, (decision, r) in enumerate(chosen)
    ]


def read_rows(test_split: Path, feature_order: list[str]) -> list[dict]:
    with test_split.open(newline="") as f:
        reader = csv.DictReader(f)
        missing = set(feature_order) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{test_split}: missing columns {sorted(missing)}")
        return [
            {
                "row_id": int(row["row_id"]),
                "p": float(row["plaintext_probability"]),
                "features": {name: int(row[name]) for name in feature_order},
            }
            for row in reader
        ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--test-split", type=Path, default=TEST_SPLIT_PATH)
    ap.add_argument("--preprocessing", type=Path, default=PREPROCESSING_PATH)
    ap.add_argument("--config", type=Path,
                    default=Path(os.environ.get("BANK_CONFIG", CONFIG_PATH)))
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    if not args.test_split.exists() or not args.preprocessing.exists():
        missing = args.test_split if not args.test_split.exists() else args.preprocessing
        print(
            f"{missing} not found. Train the model first:\n"
            f"    python -m training.train\n"
            f"or run training and this script together:\n"
            f"    python scripts/setup_demo.py",
            file=sys.stderr,
        )
        return 1

    feature_order = json.loads(args.preprocessing.read_text())["feature_order"]
    approve_below, reject_from = load_thresholds(args.config)
    applicants = pick(read_rows(args.test_split, feature_order), approve_below, reject_from)

    args.out.write_text(json.dumps({
        "thresholds": {"approve_below": approve_below, "reject_from": reject_from},
        "applicants": applicants,
    }, indent=2) + "\n")
    print(f"Wrote {len(applicants)} demo applicants to {args.out}")
    for a in applicants:
        print(f"  {a['label']}  row {a['row_id']:>4}  p={a['plaintext_probability']:.3f}  "
              f"{a['expected_decision']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

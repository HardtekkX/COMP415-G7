"""Score one applicant from training/artifacts/test_split.csv through the scorer.

Talks to the scoring service directly (the bank's HTTP API need not run) and
compares the encrypted-path probability with the plaintext probability that
training stored in test_split.csv. The bank never computes it from weights.

The test split has no national IDs, so the CLI uses a made-up one, 000 followed
by the zero-padded row ID. No real T.C. kimlik number starts with 0.

Run:  python -m bank_service.cli [--row-id 690]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from bank_service.config import REPO_ROOT, BankConfig
from bank_service.core import Bank
from bank_service.scorer_client import ScorerError

TEST_SPLIT = REPO_ROOT / "training" / "artifacts" / "test_split.csv"


def fake_national_id(row_id: int) -> str:
    return f"000{row_id:08d}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--row-id", type=int, help="row_id in test_split.csv (default: first row)")
    parser.add_argument("--test-split", type=Path, default=TEST_SPLIT)
    args = parser.parse_args(argv)

    if not args.test_split.exists():
        print(f"{args.test_split} not found; run python -m training.train", file=sys.stderr)
        return 1
    rows = pd.read_csv(args.test_split)
    if args.row_id is None:
        row = rows.iloc[0]
    else:
        match = rows[rows["row_id"] == args.row_id]
        if match.empty:
            print(f"row_id {args.row_id} not in {args.test_split}", file=sys.stderr)
            return 1
        row = match.iloc[0]

    bank = Bank.from_config(BankConfig.load())
    row_id = int(row["row_id"])
    national_id = fake_national_id(row_id)
    features = {name: row[name] for name in bank.preprocessing.feature_order}
    try:
        result = bank.score(national_id, features)
    except ScorerError as exc:
        print(f"scoring failed: {exc}", file=sys.stderr)
        return 1

    plaintext = float(row["plaintext_probability"])
    print(f"row_id                  {row_id} (actual outcome: {'default' if row['default'] else 'repaid'})")
    print(f"national_id (fake)      {national_id}")
    print(f"pseudonym sent          {result.pseudonym}")
    print(f"ciphertext              {result.ciphertext_bytes:,} bytes")
    print(f"encrypted probability   {result.probability_of_default:.6f}")
    print(f"plaintext probability   {plaintext:.6f}   (from test_split.csv)")
    print(f"difference              {abs(result.probability_of_default - plaintext):.2e}")
    print(f"decision                {result.decision}")
    print("timings (ms)            " + ", ".join(f"{k} {v:.1f}" for k, v in result.timings_ms.items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())

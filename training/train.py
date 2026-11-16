"""Train the credit-scoring model on UCI Statlog German Credit.

Downloads german.data (1000 applicants, 20 attributes), builds 10 numeric
features, fits StandardScaler + LogisticRegression on a stratified 70/30 split
and exports what both services need to training/artifacts/model.json. The
model predicts the probability of default (bad credit = 1).

Features (UCI attribute in brackets):
  duration_months          [A2]  loan duration in months
  credit_amount            [A5]  loan amount in DM
  installment_rate         [A8]  instalment as % of disposable income, 1-4
  existing_credits         [A16] number of credits already at this bank
  checking_status          [A1]  0 overdrawn, 1 0-200 DM, 2 >=200 DM,
                                 3 no checking account (order of observed risk)
  savings_band             [A6]  0 <100 DM or unknown/none, 1 100-500,
                                 2 500-1000, 3 >=1000
  employment_band          [A7]  0 unemployed, 1 <1y, 2 1-4y, 3 4-7y, 4 >=7y
  other_installment_plans  [A14] 1 if instalments at another bank or store
  owns_home                [A15] 1 if the applicant owns their home
  has_guarantor            [A10] 1 if a guarantor backs the loan
Not used: age, personal status/sex, foreign worker (protected characteristics),
credit history (labels contradict risk through selection bias), purpose
(10 categories), residence, dependants, telephone, job (no signal), property
(overlaps owns_home).

Run:  python -m training.train
"""

from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

SEED = 415
TEST_SIZE = 0.3

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
DATA_SHA256 = "b21f3d81db8071257d5ff1deaeba1fd4303b62712e6fcc9715c7a86202cb5871"

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "data" / "german.data"          # git-ignored
ARTIFACTS = HERE / "artifacts"
MODEL_PATH = ARTIFACTS / "model.json"
TEST_PATH = ARTIFACTS / "test_split.csv"            # git-ignored (*.csv)

RAW_COLUMNS = [
    "checking", "duration", "history", "purpose", "amount", "savings", "employment",
    "installment_rate", "personal_status", "debtors", "residence", "property", "age",
    "other_plans", "housing", "existing_credits", "job", "liable", "telephone",
    "foreign_worker", "target",
]

FEATURES = [
    "duration_months",
    "credit_amount",
    "installment_rate",
    "existing_credits",
    "checking_status",
    "savings_band",
    "employment_band",
    "other_installment_plans",
    "owns_home",
    "has_guarantor",
]

CHECKING = {"A11": 0, "A12": 1, "A13": 2, "A14": 3}
SAVINGS = {"A61": 0, "A62": 1, "A63": 2, "A64": 3, "A65": 0}
EMPLOYMENT = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}

MANUAL_STEP = f"""Could not download the dataset. Manual step:
  1. Open {DATA_URL}
     (or https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data,
     file german.data inside the zip)
  2. Save it as {DATA_PATH}
  3. Run python -m training.train again."""


def fetch_dataset() -> Path:
    if not DATA_PATH.exists():
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {DATA_URL}")
        try:
            with urllib.request.urlopen(DATA_URL, timeout=30) as resp:
                payload = resp.read()
        except OSError as exc:
            sys.exit(f"{exc}\n{MANUAL_STEP}")
        DATA_PATH.write_bytes(payload)
    digest = hashlib.sha256(DATA_PATH.read_bytes()).hexdigest()
    if digest != DATA_SHA256:
        sys.exit(f"{DATA_PATH} has SHA-256 {digest}, expected {DATA_SHA256}.\n{MANUAL_STEP}")
    return DATA_PATH


def build_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({
        "duration_months": raw["duration"],
        "credit_amount": raw["amount"],
        "installment_rate": raw["installment_rate"],
        "existing_credits": raw["existing_credits"],
        "checking_status": raw["checking"].map(CHECKING),
        "savings_band": raw["savings"].map(SAVINGS),
        "employment_band": raw["employment"].map(EMPLOYMENT),
        "other_installment_plans": raw["other_plans"].isin(["A141", "A142"]).astype(int),
        "owns_home": (raw["housing"] == "A152").astype(int),
        "has_guarantor": (raw["debtors"] == "A103").astype(int),
    })[FEATURES]
    if X.isna().any().any():
        raise ValueError("unmapped category in the raw data")
    y = (raw["target"] == 2).astype(int).rename("default")   # 1 = bad credit
    return X, y


def main() -> None:
    raw = pd.read_csv(fetch_dataset(), sep=" ", header=None, names=RAW_COLUMNS)
    X, y = build_features(raw)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )

    scaler = StandardScaler().fit(X_train)
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    model.fit(scaler.transform(X_train), y_train)

    proba = model.predict_proba(scaler.transform(X_test))[:, 1]
    accuracy = accuracy_score(y_test, (proba >= 0.5).astype(int))
    auc = roc_auc_score(y_test, proba)
    baseline = 1 - y_test.mean()

    print(f"Train {len(X_train)}  test {len(X_test)}  (seed {SEED}, stratified)")
    print(f"Test accuracy  {accuracy:.3f}   (always-'good' baseline {baseline:.3f})")
    print(f"Test AUC       {auc:.3f}")
    print("\nStandardised weights (positive = more likely to default):")
    for name, w in sorted(zip(FEATURES, model.coef_[0]), key=lambda t: -abs(t[1])):
        print(f"  {name:<25}{w:+.3f}")
    print(f"  {'bias':<25}{model.intercept_[0]:+.3f}")

    ARTIFACTS.mkdir(exist_ok=True)
    export = {
        "description": "Logistic regression on UCI Statlog German Credit; "
                       "output = probability of default. Apply (x - scaler_mean) / "
                       "scaler_scale in feature_order, logit = weights . x_scaled + bias.",
        "feature_order": FEATURES,
        "weights": model.coef_[0].tolist(),
        "bias": float(model.intercept_[0]),
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "positive_class": "default",
        "metrics": {"test_accuracy": round(accuracy, 4), "test_auc": round(auc, 4)},
        "training": {
            "dataset": "UCI Statlog German Credit (german.data)",
            "dataset_sha256": DATA_SHA256,
            "seed": SEED,
            "test_size": TEST_SIZE,
            "n_train": len(X_train),
            "n_test": len(X_test),
        },
    }
    MODEL_PATH.write_text(json.dumps(export, indent=2) + "\n")

    test = X_test.copy()
    test.insert(0, "row_id", X_test.index)
    test["default"] = y_test
    test["plaintext_probability"] = np.round(proba, 6)
    test.to_csv(TEST_PATH, index=False)
    print(f"\nWrote {MODEL_PATH.relative_to(HERE.parent)} and {TEST_PATH.relative_to(HERE.parent)}")


if __name__ == "__main__":
    main()

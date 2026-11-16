# COMP415 G7: Encrypted Credit Scoring with Homomorphic Encryption

A bank gets a credit score for a loan applicant from an external scoring service
(modelled on KKB / Findeks) without the scorer ever seeing the applicant's data.
Features are encrypted with CKKS (TenSEAL), the scorer runs a logistic regression
on the ciphertext, and only the bank can decrypt the result.

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for the architecture, API contract, scope
and schedule. Developer notes and commands are in [CLAUDE.md](CLAUDE.md).

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m training.train   # downloads the dataset, writes training/artifacts/
python -m pytest
```

`training/artifacts/model.json` (weights and bias, read only by the scoring
service) and `training/artifacts/preprocessing.json` (feature order and scaler,
read only by the bank service) are committed.
`training/artifacts/test_split.csv` and the downloaded dataset are git-ignored;
rerun `python -m training.train` to regenerate them.

## Run the two services

```bash
# Terminal 1: scoring service (holds the weights)
uvicorn scoring_service.main:app --port 8001

# Terminal 2: bank service (holds the keys; first run creates bank_service/keys/)
uvicorn bank_service.main:app --port 8000

# Score one applicant from the test split without the bank's HTTP API
python -m bank_service.cli --row-id 690

# Or through the bank's API (the national ID here is fake)
curl -s localhost:8000/apply -H 'content-type: application/json' -d '{
  "national_id": "00000000001",
  "features": {"duration_months": 24, "credit_amount": 3500, "installment_rate": 3,
               "existing_credits": 1, "checking_status": 1, "savings_band": 0,
               "employment_band": 2, "other_installment_plans": 0, "owns_home": 1,
               "has_guarantor": 0}}'
```

Decision thresholds, the scorer URL and the keys folder are set in
`bank_service/config.toml`. `bank_service/keys/` holds the HE secret key and
the HMAC key; it is git-ignored and must never be committed.

## Tests

```bash
python -m pytest                       # everything
python -m pytest tests/test_e2e.py -s  # both services as real processes, 20 applicants
```

The end-to-end test needs `python -m training.train` to have been run
(it reads `test_split.csv`) and is skipped otherwise.

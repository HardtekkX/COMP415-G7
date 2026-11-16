# CLAUDE.md

`PROJECT_PLAN.md` is the source of truth for this project. Read it before making
design decisions; if this file and the plan disagree, the plan wins.

## Goal

A bank obtains a credit score for a loan applicant from an external scoring
service (playing KKB / Findeks) without the scorer seeing the applicant's data.
The bank encrypts the features with CKKS homomorphic encryption, the scorer
computes a logistic-regression logit on the ciphertext, and only the bank can
decrypt it. Threat model: honest-but-curious scorer.

Due around 14 December 2026: ~2000-word report and ~30-minute live demo.

## Architecture

Two separate services plus one offline script.

- **Bank service** (`bank_service/`, FastAPI) — holds customer data, the HE secret
  key and the HMAC key. Scales features, builds the pseudonym
  `HMAC-SHA256(bank_key, national_id)`, encrypts, sends to the scorer, decrypts,
  applies the sigmoid and shows the decision (approve / review / reject).
- **Scoring service** (`scoring_service/`, FastAPI) — holds the model weights in
  plaintext. Receives the pseudonym, the public context and the ciphertext,
  computes `w · x + b` homomorphically, returns the encrypted logit.
- **Training script** (`training/`) — trains `LogisticRegression` on the public
  dataset and exports `model.json` (weights, bias: scorer only) and
  `preprocessing.json` (feature order, scaler: bank only).

The sigmoid runs at the bank after decryption, so the encrypted part is a single
dot product.

API (see plan; binary values as base64):
- `POST /contexts` on the scorer with `{context}` (public context, no secret
  key) → `{context_id}`. Registered once; the scorer keeps it in memory (at
  most 100, oldest evicted, so the bank must re-register on a 404). 400 for a
  context that holds a secret key.
- `POST /score` on the scorer with `{pseudonym, context_id, ciphertext}` →
  `{pseudonym, encrypted_logit}`. 404 for an unknown `context_id`, 422 for a
  wrong vector length or a malformed field (pseudonym must be 64 lowercase hex).
- `GET /health` on both services → `{"status": "ok"}`.

The scorer returns the summed logit, never element-wise `w_i * x_i` (the bank
could divide by `x_i` and recover the weights).

## CKKS configuration (fixed, see plan)

N=8192, coefficient chain `[60, 40, 60]`, scale 2^40, Galois keys only for the
rotation steps the sum needs (derived from the vector length), scorer context
without secret key, public key or relinearisation keys. Lives in `common/he.py`;
`USE_MINIMAL_GALOIS_KEYS=false` falls back to TenSEAL's `generate_galois_keys()`.
Measured for 10 features: context 2.9 MB (one-off), ciphertext 235 KB, logit
error about 1e-6, about 4 ms per score. A server-side sigmoid (stretch goal)
would need relinearisation keys and more levels.

## Security rules for the code

- The HE secret key never leaves the bank service. The context sent to the scorer must be serialized without the secret key.
- The HMAC key stays in the bank service.
- The scoring service never receives names, national IDs or plaintext features.
- The bank service never reads the model weights. Training exports them to `training/artifacts/model.json` (scoring service only); the bank reads only `training/artifacts/preprocessing.json` (feature order and scaler).
- The scoring service must not log anything except pseudonyms, context IDs (random, issued by the scorer), ciphertext sizes and timings.
- The bank service must not log national IDs or plaintext feature values. It logs pseudonyms, decisions, sizes and timings only.

## Tech stack

| Need | Tool |
|---|---|
| Homomorphic encryption | TenSEAL (CKKS). Fallback: OpenFHE |
| Model | scikit-learn `LogisticRegression` |
| Dataset | German Credit (UCI) or Give Me Some Credit (Kaggle), TBD week 1 |
| Services | FastAPI, one app per party |
| Pseudonym | Python `hmac` + `hashlib` |
| Front end | Streamlit |
| Should have | HTTPS (self-signed cert), Docker Compose |

Python only. Local Python is 3.12; confirm TenSEAL supports it before relying on
it (switch to the fallback if it doesn't install by 18 November).

## Repo layout

```
bank_service/       FastAPI app: scaling, pseudonym, encrypt, decrypt, decision
scoring_service/    FastAPI app: load weights, encrypted dot product
common/             shared request/response schemas, serialization helpers
training/           train.py, model.json (scorer), preprocessing.json (bank)
ui/                 Streamlit screens: bank, scorer, comparison
evaluation/         measurement scripts and results
notebooks/          experiments, first TenSEAL test
tests/              unit and end-to-end tests
docs/               project plan PDF, report drafts
```

Datasets, binary model artifacts (`*.pkl`, `*.joblib`), keys and certificates are
git-ignored. Never commit keys.

## Commands

Fill these in as the project is built.

```bash
# Setup
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Train (downloads german.data, writes training/artifacts/model.json,
# preprocessing.json and test_split.csv; test_split.csv is git-ignored and
# regenerated by this command)
python -m training.train

# Run scorer (SCORER_MODEL_PATH, SCORER_MAX_CONTEXTS=100 optional)
uvicorn scoring_service.main:app --port 8001

# Run bank (bank_service/config.toml: scorer URL, keys dir, decision thresholds;
# env BANK_SCORER_URL, BANK_KEYS_DIR override). First run creates the HE secret
# key and HMAC key in bank_service/keys/ (git-ignored, never commit).
uvicorn bank_service.main:app --port 8000

# Score one test applicant from the command line (scorer must be running)
python -m bank_service.cli --row-id 690

# Test (all; test_e2e.py starts both services itself and needs training run first)
python -m pytest
python -m pytest tests/test_e2e.py -s     # end to end only, prints max error

# Run UI           (TODO) streamlit run ui/app.py
# Evaluate         (TODO) python -m evaluation.run
```

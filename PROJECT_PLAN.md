# COMP415 G7: Encrypted Credit Scoring with Homomorphic Encryption

A bank gets a credit score for a loan applicant from an external scoring service, and the service never sees the applicant's data. The scenario is modelled on how Turkish banks use KKB (Kredi Kayıt Bürosu) and its Findeks score.

The bank encrypts the applicant's features with homomorphic encryption (CKKS). The scoring service runs a logistic regression on the ciphertext and returns an encrypted result. Only the bank can decrypt it.

- **Team:** 4 members, tasks unassigned
- **Due:** around 14 December 2026
- **Deliverables:** report of about 2000 words, live demo of about 30 minutes
- **Full plan with legal background and sources:** COMP415 G7 Project Plan v3 (PDF), shared with the team separately, not stored in this repo

## Why

Today KKB must read customer records in plaintext to compute a score. Turkish banking law (Banking Law No. 5411, Article 73) treats this data as a customer secret and requires sharing to be limited to what the purpose needs. Our demo shows the same scoring service working without the scorer seeing the data.

Threat model: an honest-but-curious scorer. It follows the protocol but would read anything it is given.

## Architecture

Two separate services and one offline script.

| Component | Holds | Does |
|---|---|---|
| Bank service | Customer data, HE secret key, HMAC key | Scales features, makes the pseudonym, encrypts, sends, decrypts, applies sigmoid, shows the decision |
| Scoring service (plays KKB) | Model weights in plaintext | Receives ciphertext, computes the encrypted dot product plus bias, returns ciphertext |
| Training script | Public credit dataset | Trains logistic regression, exports weights (for the scorer) and scaler (for the bank) separately |

### One score request

1. Loan officer enters or loads an applicant's features.
2. Bank service scales the features with the scaler from training.
3. Bank service replaces the national ID with `HMAC-SHA256(bank_key, national_id)`.
4. Bank service encrypts the feature vector with CKKS.
5. Bank service sends the pseudonym, the ciphertext and the `context_id` of its registered public context to the scoring service. The public context itself (about 2.9 MB, almost all rotation keys) is registered once with `POST /contexts`, not sent with every request.
6. Scoring service computes `w · x + b` on the ciphertext. The result is an encrypted logit.
7. Scoring service returns the encrypted logit.
8. Bank service decrypts, applies the sigmoid, and shows the probability and a decision: approve, review or reject.

The sigmoid runs at the bank after decryption, so the encrypted part is a single dot product.

### Security rules for the code

- The HE secret key never leaves the bank service. The context sent to the scorer must be serialized without the secret key.
- The HMAC key stays in the bank service.
- The scoring service never receives names, national IDs or plaintext features.
- The bank service never reads the model weights. Training exports them to `training/artifacts/model.json` (scoring service only); the bank reads only `training/artifacts/preprocessing.json` (feature order and scaler).
- The scoring service must not log anything except pseudonyms, context IDs (random, issued by the scorer), ciphertext sizes and timings.
- The bank service must not log national IDs or plaintext feature values. It logs pseudonyms, decisions, sizes and timings only.
- `GET /activity` and `GET /contexts/summary` on the scorer (demo screen only) expose only what the scorer may log (pseudonyms, context IDs, sizes, timings, timestamps) plus, for each ciphertext received and each encrypted result returned, 64 base64 characters from the middle (encrypted payload, not the serialization header) and the first 16 hex characters of its SHA-256. Never plaintext features, national IDs or decrypted values. They have no authentication; `SCORER_DEMO_ENDPOINTS=false` turns them off (404), and a real deployment would turn them off or put them behind authentication.

## Tech stack

| Need | Tool |
|---|---|
| Homomorphic encryption | TenSEAL (CKKS). Fallback: OpenFHE |
| Model | scikit-learn `LogisticRegression` |
| Dataset | UCI Statlog German Credit (`german.data`, 1000 rows), 10 features |
| Services | FastAPI, one app per party |
| Pseudonym | Python `hmac` and `hashlib` |
| Front end | Streamlit |
| Optional | HTTPS with a self-signed certificate, Docker Compose |

Language: Python only. Python 3.12 is required: TenSEAL 0.3.18 ships a wheel for CPython 3.12 (cp312), and versions are pinned in `requirements.txt`.

## CKKS configuration (chosen 16 November 2026)

Measured with `notebooks/he_params.py`; implemented in `common/he.py`.

| Parameter | Value | Why |
|---|---|---|
| `poly_modulus_degree` N | 8192 | 4096 slots, plenty for one feature vector; 128-bit security with room for a 40-bit level |
| Coefficient modulus chain | `[60, 40, 60]` (160 bits, limit 218) | One rescale: the ciphertext × plaintext multiply. Then the plaintext bias is added at the same scale |
| Scale | 2^40 | Matches the 40-bit middle prime |
| Galois (rotation) keys | Only the steps the encrypted sum needs, derived from the vector length: 1, 2, 4, ... below the next power of two (10 features: 1, 2, 4, 8) | TenSEAL's default makes keys for every power-of-two step; we generate only the needed ones with `sealapi`. `USE_MINIMAL_GALOIS_KEYS=false` falls back to the default |
| Scorer context | Galois keys only. No secret key, no public key, no relinearisation keys | The scorer never decrypts, never encrypts, and never multiplies two ciphertexts |

Measured numbers for 10 features:

| Measure | Value |
|---|---|
| Public context sent to the scorer | 2.9 MB, once per bank (`POST /contexts`) |
| Ciphertext per request | 235 KB |
| Logit error after decryption | about 1e-6 |
| Scorer time for `w · x + b` | about 4 ms in-process (`notebooks/he_params.py`); about 6.4 ms median as measured by the running scoring service (`evaluation/results.md`) |

Stretch goal note: a server-side polynomial sigmoid multiplies ciphertexts, so it would need the relinearisation keys in the scorer context and a longer chain (one extra level per multiplication depth), which means larger keys and ciphertexts. This configuration does not support it.

## Repo layout

```
bank_service/       FastAPI app (POST /apply): scaling, pseudonym, encrypt, decrypt, decision;
                    cli.py scores one test-set row; config.toml; keys/ (git-ignored)
scoring_service/    FastAPI app: load weights, register contexts, encrypted dot product, demo endpoints
common/             CKKS setup (he.py), shared request and response schemas, FastAPI helpers
training/           train.py; artifacts/: model.json (scorer), preprocessing.json (bank),
                    metrics.json, test_split.csv (git-ignored)
ui/                 Streamlit app (app.py): Home, Bank, Scorer and Comparison pages in views/;
                    demo_applicants.py writes the git-ignored demo_applicants.json
scripts/            setup_demo.py: training, then demo applicants
evaluation/         measurement script and results (to do)
notebooks/          he_spike.py (first TenSEAL test), he_params.py (CKKS parameter measurements)
tests/              unit, UI and end-to-end tests
docs/               report notes and drafts
```

## API contract

Binary values are standard base64 strings. Every request body rejects unknown
fields. A malformed request gets 422 with field locations only, never the
input: `{"detail": "invalid request", "fields": ["body.pseudonym"]}`.

### Scoring service (port 8001)

`POST /contexts` registers the bank's public context. The scorer keeps it in memory, at most 100 contexts (`SCORER_MAX_CONTEXTS`); the oldest is evicted first.

Request:

```json
{
  "context": "base64, public CKKS context without the secret key"
}
```

Response:

```json
{
  "context_id": "32 lowercase hex characters, random, issued by the scorer"
}
```

Returns 400 if the bytes are not a valid public CKKS context, including a context that holds a secret key.

`POST /score`

Request:

```json
{
  "pseudonym": "64 lowercase hex characters, HMAC-SHA256 of the national ID",
  "context_id": "value returned by POST /contexts",
  "ciphertext": "base64, encrypted scaled feature vector"
}
```

Response:

```json
{
  "pseudonym": "same value as in the request",
  "encrypted_logit": "base64, encrypted w · x + b"
}
```

Returns 404 if `context_id` is unknown (for example after a scorer restart or an eviction); the bank then registers its context again and retries. Returns 422 for a wrong vector length or a malformed field, and 400 if the ciphertext cannot be read or scored with that context.

The scorer must return the summed logit, not the element-wise products `w_i * x_i`: the bank chooses `x`, so it could divide each product by `x_i` and recover every model weight.

Demo-only endpoints (for the scorer screen; switched off with `SCORER_DEMO_ENDPOINTS=false`, which makes both return 404):

- `GET /activity` returns `{"records": [...]}`, the last 50 successful scores, newest first. Each record: `timestamp`, `pseudonym`, `context_id`, `ciphertext_bytes`, `ciphertext_slice` and `result_slice` (64 base64 characters from the middle of the ciphertext received and of the encrypted logit returned, so the demo shows encrypted payload rather than the serialization header), `ciphertext_sha256` and `result_sha256` (first 16 hex characters of the SHA-256 of the raw bytes), `compute_ms`.
- `GET /contexts/summary` returns `{"count", "max_contexts", "contexts": [{"context_id", "size_bytes", "registered_at"}]}`.

### Bank service (port 8000)

`POST /apply` scores one applicant through the scoring service. This is the bank's own API: it takes plaintext data, none of which is sent to the scorer.

Request:

```json
{
  "national_id": "11 digits",
  "features": {
    "duration_months": 24, "credit_amount": 3500, "installment_rate": 3,
    "existing_credits": 1, "checking_status": 1, "savings_band": 0,
    "employment_band": 2, "other_installment_plans": 0, "owns_home": 1,
    "has_guarantor": 0
  }
}
```

All ten features are integers, with ranges checked (see `bank_service/schemas.py`); their order must equal `feature_order` in `preprocessing.json`, which the bank checks at startup.

Response:

```json
{
  "pseudonym": "64 lowercase hex characters, as sent to the scorer",
  "probability_of_default": 0.3653,
  "decision": "approve | review | reject",
  "logit": -0.5524,
  "ciphertext_bytes": 235044,
  "encrypted_logit_bytes": 131216,
  "timings_ms": {"prepare": 0.1, "encrypt": 3.4, "scorer_call": 11.5, "decrypt": 0.4, "total": 15.4}
}
```

Decision thresholds come from `bank_service/config.toml`: approve below 0.25, reject from 0.50, review in between. Returns 502 if the scoring service fails.

### Both services

`GET /health` returns `{"status": "ok"}`.

## Scope

| Priority | What | Status |
|---|---|---|
| Must have | One applicant scored end to end under CKKS. Two separate services. HMAC pseudonym. Three screens. | Done, 16 November |
| Must have | Five measurements over the test set (`evaluation/`) | Open, due 6 December |
| Done early | Six pre-loaded demo applicants, two each for approve, review and reject | Done, 16 November |
| Optional | HTTPS between services. Docker Compose. | Only if the must-haves and report are on track |
| Optional (stretch) | Batch scoring with CKKS packing. Server-side polynomial sigmoid. Encrypted portfolio aggregate. Fraud-score scenario. | Only after everything above works |

## Schedule

Status on 16 November 2026: the core flow (training, both services, command-line scoring) and all three demo screens (Bank, Scorer, Comparison) are done, as are the six demo applicants.

| Week | Goal | Tasks | Done when |
|---|---|---|---|
| 1: 16 to 22 Nov | Prove the risky parts and fix the interfaces | Done 16 Nov: encrypted dot product, dataset and features, model and exports, API contract, repo structure. | Done |
| 2: 23 to 29 Nov | Core flow between two services | Done 16 Nov: bank and scoring services, CLI, CKKS parameters. | Done |
| 3: 30 Nov to 6 Dec | Measurements and report draft | Done 16 Nov: the three screens. Open: evaluation script for the five measurements over the test set; report draft. Optional: HTTPS, Docker Compose. | Feature freeze on 6 December. Full demo runs on one laptop. |
| 4: 7 to 13 Dec | Polish, rehearse, submit | Finish the report. Make the slides. Rehearse the demo twice. Record a backup video. Bug fixes only. | Report and slides final, two rehearsals done, backup video recorded |
| 14 Dec | Deliver | Submit the report and give the live demo | |

Still to do: the evaluation script, the report, slides, two rehearsals and the backup video. Optional: HTTPS, Docker Compose and the stretch goals.

Four workstreams run in parallel: model and data, HE core, services and API, screens and evaluation. The team picks tasks at a short meeting at the start of each week.

### Week 1 checklist

- [x] Install TenSEAL and run one encrypted dot product end to end
- [x] Pick the dataset and freeze the feature list
- [x] Train logistic regression, record plaintext accuracy and AUC
- [x] Export weights and scaler in a format both services can load
- [x] Freeze the API contract
- [x] Create the repo layout and a `requirements.txt` with pinned versions

## Demo screens

A Streamlit app (`streamlit run ui/app.py`) that talks to the two services over HTTP only; it never imports service code or reads the model weights.

- **Home:** what the demo shows, whether both services are up, and whether the demo applicants exist.
- **Bank screen:** pick one of the six demo applicants, edit the national ID or any feature, and request a score. Shows the decision, the probability of default and the thresholds. A "What left the bank" box puts what was sent (the pseudonym, the ciphertext size, the size of the encrypted logit that came back) next to what stayed (national ID and every feature value), plus a per-step timing table.
- **Scorer screen:** built only from `GET /activity` and `GET /contexts/summary`. Shows the registered public contexts (count, sizes, no secret key), a list of what the scorer does not have, and a table of recent requests (time, pseudonym, context ID, ciphertext size, SHA-256 fingerprint, compute time). For a selected request: the full pseudonym and, for the ciphertext received and the encrypted result returned, a 64-character base64 slice from the middle and the fingerprint.
- **Comparison screen:** one demo applicant scored through the encrypted path next to its plaintext probability from training: both probabilities, the difference, whether the decisions match, ciphertext size against plain JSON, and a timing chart. **Run all six** fills a table for all demo applicants with the largest difference and the number of matching decisions.

## What we measure

| Measure | How |
|---|---|
| Score error | Absolute difference between encrypted and plaintext probability over the test set |
| Decision agreement | Share of test applicants with the same approve, review or reject outcome |
| Latency | Time to encrypt, compute and decrypt, against plaintext scoring |
| Ciphertext size | Bytes sent per request, against plaintext |
| Model quality | Accuracy and AUC of the model itself |

Record the CKKS parameters used with every result.

# COMP415 G7: Encrypted Credit Scoring with Homomorphic Encryption

A bank gets a credit score for a loan applicant from an external scoring service, and the service never sees the applicant's data. The scenario is modelled on how Turkish banks use KKB (Kredi Kayıt Bürosu) and its Findeks score.

The bank encrypts the applicant's features with homomorphic encryption (CKKS). The scoring service runs a logistic regression on the ciphertext and returns an encrypted result. Only the bank can decrypt it.

- **Team:** 4 members, tasks unassigned
- **Due:** around 14 December 2026
- **Deliverables:** report of about 2000 words, live demo of about 30 minutes
- **Full plan with legal background and sources:** `COMP415 G7 Project Plan v2` (PDF)

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

## Tech stack

| Need | Tool |
|---|---|
| Homomorphic encryption | TenSEAL (CKKS). Fallback: OpenFHE |
| Model | scikit-learn `LogisticRegression` |
| Dataset | German Credit (UCI) or "Give Me Some Credit" (Kaggle), to be decided in week 1 |
| Services | FastAPI, one app per party |
| Pseudonym | Python `hmac` and `hashlib` |
| Front end | Streamlit |
| Should have | HTTPS with a self-signed certificate, Docker Compose |

Language: Python only. Check TenSEAL and Python version compatibility on day one. If TenSEAL does not install on the team's machines by 18 November, switch to the fallback.

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
| Scorer time for `w · x + b` | about 4 ms |

Stretch goal note: a server-side polynomial sigmoid multiplies ciphertexts, so it would need the relinearisation keys in the scorer context and a longer chain (one extra level per multiplication depth), which means larger keys and ciphertexts. This configuration does not support it.

## Proposed repo layout

A starting point. Change it in week 1 if the team prefers something else.

```
bank_service/       FastAPI app: scaling, pseudonym, encrypt, decrypt, decision
scoring_service/    FastAPI app: load weights, encrypted dot product
common/             shared request and response schemas, serialization helpers
training/           train.py, model.json (scorer), preprocessing.json (bank)
ui/                 Streamlit screens: bank, scorer, comparison
evaluation/         measurement scripts and results
notebooks/          experiments, first TenSEAL test
tests/              unit and end-to-end tests
docs/               full project plan, report drafts
```

## Proposed API contract

A draft to freeze in week 1. Binary values are base64 strings.

`POST /contexts` on the scoring service registers the bank's public context once. The scorer keeps it in memory.

Request:

```json
{
  "context": "base64, public CKKS context without the secret key"
}
```

Response:

```json
{
  "context_id": "opaque string chosen by the scorer"
}
```

`POST /score` on the scoring service

Request:

```json
{
  "pseudonym": "hex string, HMAC-SHA256 of the national ID",
  "context_id": "value returned by POST /contexts",
  "ciphertext": "base64, encrypted feature vector"
}
```

Response:

```json
{
  "pseudonym": "same value as in the request",
  "encrypted_logit": "base64, encrypted result"
}
```

Returns 404 if `context_id` is unknown (for example after a scorer restart); the bank then registers its context again.

The scorer must return the summed logit, not the element-wise products `w_i * x_i`: the bank chooses `x`, so it could divide each product by `x_i` and recover every model weight.

`GET /health` on both services returns `{"status": "ok"}`.

## Scope

| Priority | What | Rule |
|---|---|---|
| Must have | One applicant scored end to end under CKKS. Two separate services. HMAC pseudonym. Three screens. Five measurements. | Done by 6 December |
| Should have | HTTPS between services. Docker Compose. A few pre-loaded applicants with different outcomes. | Week 3, if the core is stable |
| Cut first | Batch scoring with CKKS packing. Server-side polynomial sigmoid. Encrypted portfolio aggregate. Fraud-score scenario. | Only after everything above works |

## Schedule

| Week | Goal | Tasks | Done when |
|---|---|---|---|
| 1: 16 to 22 Nov | Prove the risky parts and fix the interfaces | Run one encrypted dot product in a notebook. Pick the dataset and freeze the feature list. Train the model and export weights and scaler. Freeze the API contract. Set up the repo structure. | A notebook shows the encrypted score matching the plaintext score for one applicant |
| 2: 23 to 29 Nov | Core flow between two services | Build the bank service and the scoring service. Connect them over the API. Start the bank screen. Choose and record the CKKS parameters. | One command-line request returns the right decision |
| 3: 30 Nov to 6 Dec | Screens, measurements, report draft | Finish the three screens. Run the measurements over the test set. Add should-have items if stable. Draft the report. | Feature freeze on 6 December. Full demo runs on one laptop. |
| 4: 7 to 13 Dec | Polish, rehearse, submit | Finish the report. Rehearse the demo twice. Record a backup video. Bug fixes only. | Report final, two rehearsals done |
| 14 Dec | Deliver | Submit the report and give the live demo | |

Four workstreams run in parallel: model and data, HE core, services and API, screens and evaluation. The team picks tasks at a short meeting at the start of each week.

### Week 1 checklist

- [ ] Install TenSEAL and run one encrypted dot product end to end
- [ ] Pick the dataset and freeze the feature list
- [ ] Train logistic regression, record plaintext accuracy and AUC
- [ ] Export weights and scaler in a format both services can load
- [ ] Freeze the API contract
- [ ] Create the repo layout and a `requirements.txt` with pinned versions

## Demo screens

- **Bank screen:** applicant form, decrypted probability, decision.
- **Scorer screen:** exactly what the scoring service received and returned. A pseudonym and ciphertext bytes, no names or numbers.
- **Comparison screen:** encrypted result next to the plaintext result for the same applicant.

## What we measure

| Measure | How |
|---|---|
| Score error | Absolute difference between encrypted and plaintext probability over the test set |
| Decision agreement | Share of test applicants with the same approve, review or reject outcome |
| Latency | Time to encrypt, compute and decrypt, against plaintext scoring |
| Ciphertext size | Bytes sent per request, against plaintext |
| Model quality | Accuracy and AUC of the model itself |

Record the CKKS parameters used with every result.

## Known limits

- A real bureau combines records from many banks. Here one bank supplies all features. Cross-bank linking under encryption is out of scope.
- The bank learns the raw logit, so with many queries it could estimate the model's weights.
- CKKS is approximate. Wrong parameters give wrong scores or low security.
- Whether HE ciphertext still counts as a customer secret under Turkish law is untested. We present HE as a technical measure, not a legal exemption.

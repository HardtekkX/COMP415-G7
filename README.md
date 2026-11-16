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
python -m pytest
```

# Report notes

Material for the report that is not in the plan or the README.

## Limitations to cover in the report

### Scope and scenario

- A real bureau combines records from many banks. Here one bank supplies all
  features. Cross-bank linking under encryption is out of scope.
- Whether HE ciphertext still counts as a customer secret under Turkish law
  (Banking Law No. 5411, Article 73) is untested. We present HE as a technical
  measure, not a legal exemption.

### What each party can still learn

- **The bank learns the raw logit.** The scorer returns the encrypted
  `w · x + b`, and the bank decrypts it before applying the sigmoid. Because
  the bank chooses `x`, with enough queries (about one per feature plus one,
  since the model is linear) it could solve for the weights and bias. Returning
  the summed logit rather than each `w_i * x_i` stops the one-query version of
  this attack, not the many-query one. Possible mitigations, not implemented:
  rate limits, noise added to the logit, or a server-side polynomial sigmoid.
- **The pseudonym is stable.** It is `HMAC-SHA256(bank_key, national_id)`, so
  the same person always gets the same pseudonym. The scorer cannot reverse it
  without the bank's key, but it can link every request for the same person
  and count how often they apply. This is intentional (it plays the role of a
  bureau's customer key) but should be stated. A fresh random pseudonym per
  request would prevent linking, at the cost of the scorer being unable to keep
  any history.

### Accuracy of the numbers

- **The test split was used for feature decisions.** The test split was looked
  at while choosing the feature encodings, so the reported test accuracy 0.760
  (baseline 0.700) and AUC 0.770 are not from an untouched holdout and are
  likely a little optimistic.
- **CKKS is approximate.** Wrong parameters give wrong scores or low security.
  With the chosen parameters (N=8192, chain [60, 40, 60], scale 2^40) the
  largest probability error over all 300 test rows is about 3.3e-7, and all 300
  decisions agree with plaintext scoring.
- **The logit error is a constant offset, not random noise.** Over all 300 test
  rows the decrypted logit was below the exact plaintext logit every time, by
  1.0e-6 to 1.8e-6 (mean about 1.4e-6). Report it as a small systematic bias
  from the CKKS encoding and rescaling, not as noise that averages out. It is
  far too small to change any decision at the 0.25 / 0.50 thresholds unless a
  probability lies within about 3e-7 of a threshold.

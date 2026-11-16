"""First TenSEAL / CKKS spike.

Shows the full bank -> scorer -> bank round trip in one process:
  1. bank creates a CKKS context (holds the secret key) and encrypts features
  2. bank serialises the context WITHOUT the secret key (what the scorer gets)
  3. scorer computes w . x + b on the ciphertext
  4. bank decrypts; the scorer's copy of the context cannot

Run:  python notebooks/he_spike.py
"""

import numpy as np
import tenseal as ts

rng = np.random.default_rng(seed=415)

# --- 1. CKKS context (bank side) -------------------------------------------
context = ts.context(
    ts.SCHEME_TYPE.CKKS,
    # Ring dimension N. Security and capacity both grow with N. CKKS packs
    # N/2 = 4096 real numbers into one ciphertext ("slots"), far more than our
    # 10 features. 8192 is the smallest N that gives 128-bit security with the
    # modulus chain below (HE standard: total coeff modulus <= 218 bits).
    poly_modulus_degree=8192,
    # Bit sizes of the primes forming the coefficient modulus chain q.
    #   60 (first)  : precision of the final decrypted value (integer part + scale)
    #   40, 40      : one prime per multiplicative level. Each ciphertext x
    #                 plaintext multiply consumes one level via rescaling. The
    #                 dot product needs one multiply, so two levels is headroom.
    #   60 (last)   : "special prime" used during key switching (rotations,
    #                 relinearisation); never part of a ciphertext's modulus.
    # Total = 200 bits < 218, so 128-bit security holds for N = 8192.
    coeff_mod_bit_sizes=[60, 40, 40, 60],
)
# Scale Delta: reals are multiplied by Delta and rounded to integers before
# encoding. 2^40 matches the 40-bit middle primes, so rescaling after a
# multiply brings the scale back to ~2^40. Larger scale => more precision.
context.global_scale = 2**40
# Galois (rotation) keys are needed by dot(): after the element-wise product
# the slots are summed by repeated rotate-and-add. These are public keys.
context.generate_galois_keys()

# --- 2. Data ----------------------------------------------------------------
n_features = 10
x = rng.normal(size=n_features)         # applicant's (scaled) features - secret
w = rng.normal(size=n_features)         # model weights - known to the scorer
b = float(rng.normal())                 # model bias    - known to the scorer

enc_x = ts.ckks_vector(context, x.tolist())
ciphertext_bytes = enc_x.serialize()

# --- 3. Public context: strip the secret key --------------------------------
public_context_bytes = context.serialize(
    save_public_key=True,
    save_secret_key=False,
    save_galois_keys=True,
    save_relin_keys=True,
)
full_context_bytes = context.serialize(
    save_public_key=True,
    save_secret_key=True,
    save_galois_keys=True,
    save_relin_keys=True,
)
# Both blobs contain the Galois keys. The public one is ~50x larger: with the
# secret key present SEAL stores keys in compact "seeded" form; without it the
# keys must be stored fully expanded. The Galois keys dominate (~35 MB).
print(f"context with secret key : {len(full_context_bytes):>10,} bytes")
print(f"context without it      : {len(public_context_bytes):>10,} bytes")
print(f"ciphertext (10 floats)  : {len(ciphertext_bytes):>10,} bytes")

# --- 4. Scorer side: reload public context, compute on ciphertext -----------
scorer_context = ts.context_from(public_context_bytes)
print(f"scorer context is_private(): {scorer_context.is_private()}")

scorer_enc_x = ts.ckks_vector_from(scorer_context, ciphertext_bytes)
enc_logit = scorer_enc_x.dot(w.tolist()) + b   # w . x + b, all on ciphertext
enc_logit_bytes = enc_logit.serialize()
print(f"encrypted logit         : {len(enc_logit_bytes):>10,} bytes")

# --- 5. Scorer tries to decrypt: must fail ----------------------------------
try:
    enc_logit.decrypt()
    print("ERROR: scorer decrypted the result - secret key leaked!")
except Exception as exc:  # TenSEAL raises ValueError
    print(f"scorer decrypt failed as expected: {type(exc).__name__}: {exc}")

# --- 6. Bank side: decrypt with the secret-key context ----------------------
bank_logit = ts.ckks_vector_from(context, enc_logit_bytes)
he_result = bank_logit.decrypt()[0]
plain_result = float(np.dot(w, x) + b)

print()
print(f"encrypted result : {he_result:.12f}")
print(f"plaintext result : {plain_result:.12f}")
print(f"absolute error   : {abs(he_result - plain_result):.3e}")

"""CKKS parameter experiment: shrink keys and ciphertexts for our computation.

The scorer computes w . x + b on 10 encrypted slots: one ciphertext x plaintext
multiply, then a rotate-and-add sum. Two levers are tested:

  (a) smaller poly_modulus_degree N with a coefficient modulus chain that still
      fits the 128-bit HE-standard bound (SEAL refuses anything larger, and we
      also assert it below);
  (b) Galois keys only for the rotation steps the sum uses. TenSEAL's
      generate_galois_keys() makes keys for every power-of-two step. Its
      Python API cannot choose steps, so we generate the keys with the
      low-level sealapi KeyGenerator and load them into the TenSEAL context.
      For a 10-slot sum TenSEAL rotates by 1, 2, 4, 8 (ceil(log2 10) = 4);
      dropping step 8 fails with "Galois key not present".

The scorer's context also drops the relinearisation keys (no ciphertext x
ciphertext multiply) and the public key (it never encrypts; adding the
plaintext bias needs no key).

Run:  python notebooks/he_params.py
"""

import os
import statistics
import tempfile
import time

import numpy as np
import tenseal as ts
import tenseal.sealapi as sealapi

N_FEATURES = 10
SUM_STEPS = [1, 2, 4, 8]          # rotations TenSEAL's sum uses for 10 slots
TRIALS = 30                        # random (x, w, b) draws per configuration

# (label, N, coeff_mod_bit_sizes, scale bits, keys, rescale)
#   keys    = "all"  -> TenSEAL default, every power-of-two step (spike setup)
#             "min"  -> only SUM_STEPS
#   rescale = False  -> chain has no level to spend: keep the product at scale
#                       2^(2*scale) in the first prime instead of rescaling.
#                       "+ b" then fails (plaintext encoded at the wrong scale),
#                       so the bank appends a constant 1 to x and the scorer
#                       appends b to w: 11 slots, same rotation steps.
CONFIGS = [
    ("spike baseline",        8192, [60, 40, 40, 60], 40, "all", True),
    ("N=8192 min keys",       8192, [60, 40, 40, 60], 40, "min", True),
    ("N=8192 1 level",        8192, [60, 40, 60],     40, "min", True),
    ("N=4096 all keys",       4096, [40, 20, 40],     20, "all", True),
    ("N=4096 min keys",       4096, [40, 20, 40],     20, "min", True),
    ("N=4096 min keys s29",   4096, [40, 29, 40],     29, "min", True),
    ("N=4096 big special",    4096, [35, 30, 44],     30, "min", True),
    ("N=4096 no rescale",     4096, [60, 49],         25, "min", False),
    ("N=4096 no rescale 2",   4096, [54, 55],         24, "min", False),
    ("N=2048 no rescale",     2048, [30, 24],         13, "min", False),
]


def seal_context(n, bits):
    parms = sealapi.EncryptionParameters(sealapi.SCHEME_TYPE.CKKS)
    parms.set_poly_modulus_degree(n)
    parms.set_coeff_modulus(sealapi.CoeffModulus.Create(n, bits))
    return sealapi.SEALContext(parms, True, sealapi.SEC_LEVEL_TYPE.TC128)


def load_minimal_galois_keys(ctx, n, bits, steps):
    """Replace ctx's Galois keys with keys for `steps` only.

    TenSEAL and sealapi are separate bindings, so keys cross between them as
    files. The secret key file lives in a private temp dir for milliseconds;
    in the bank service this would stay inside the bank's own process.
    """
    sc = seal_context(n, bits)
    with tempfile.TemporaryDirectory() as tmp:
        sk_path, gk_path = os.path.join(tmp, "sk"), os.path.join(tmp, "gk")
        ctx.data.secret_key().save(sk_path)
        sk = sealapi.SecretKey()
        sk.load(sc, sk_path)
        # create_galois_keys(list) takes Galois elements, not steps.
        elts = sc.key_context_data().galois_tool().get_elts_from_steps(steps)
        keys = sealapi.GaloisKeys()
        sealapi.KeyGenerator(sc, sk).create_galois_keys(elts, keys)
        keys.save(gk_path)
        ctx.data.galois_keys().load(ctx.data.seal_context(), gk_path)


def make_context(n, bits, scale_bits, keys, rescale):
    assert sum(bits) <= sealapi.CoeffModulus.MaxBitCount(n, sealapi.SEC_LEVEL_TYPE.TC128)
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_modulus_degree=n, coeff_mod_bit_sizes=bits)
    ctx.global_scale = 2**scale_bits
    ctx.auto_rescale = rescale
    ctx.generate_galois_keys()
    if keys == "min":
        load_minimal_galois_keys(ctx, n, bits, SUM_STEPS)
    return ctx


def public_bytes(ctx, keys):
    if keys == "all":   # what he_spike.py sends
        return ctx.serialize(save_public_key=True, save_secret_key=False,
                             save_galois_keys=True, save_relin_keys=True)
    return ctx.serialize(save_public_key=False, save_secret_key=False,
                         save_galois_keys=True, save_relin_keys=False)


def run(label, n, bits, scale_bits, keys, rescale, rng):
    bank = make_context(n, bits, scale_bits, keys, rescale)
    pub_bytes = public_bytes(bank, keys)
    scorer = ts.context_from(pub_bytes)
    scorer.auto_rescale = rescale  # flag is not serialised
    assert not scorer.is_private()

    errors, times, ct_sizes, logit_sizes = [], [], [], []
    for _ in range(TRIALS):
        x = rng.normal(size=N_FEATURES)
        w = rng.normal(scale=0.5, size=N_FEATURES)
        b = float(rng.normal())
        x_sent = x.tolist() if rescale else x.tolist() + [1.0]
        ct = ts.ckks_vector(bank, x_sent).serialize()
        enc_x = ts.ckks_vector_from(scorer, ct)
        start = time.perf_counter()
        if rescale:
            enc_logit = enc_x.dot(w.tolist()) + b
        else:
            enc_logit = enc_x.dot(w.tolist() + [b])
        times.append(time.perf_counter() - start)
        out = enc_logit.serialize()
        got = ts.ckks_vector_from(bank, out).decrypt()[0]
        errors.append(abs(got - (w @ x + b)))
        ct_sizes.append(len(ct))
        logit_sizes.append(len(out))

    return {
        "label": label,
        "params": f"N={n} {bits} 2^{scale_bits}",
        "pub_ctx": len(pub_bytes),
        "ct": statistics.median(ct_sizes),
        "logit": statistics.median(logit_sizes),
        "err_max": max(errors),
        "err_med": statistics.median(errors),
        "ms": statistics.median(times) * 1000,
    }


def main():
    rng = np.random.default_rng(seed=415)
    rows = []
    for cfg in CONFIGS:
        try:
            rows.append(run(*cfg, rng))
        except Exception as exc:
            rows.append({"label": cfg[0], "fail": f"{type(exc).__name__}: {exc}"})

    head = (f"{'config':<22}{'params':<30}{'pub ctx':>13}{'ct':>11}"
            f"{'logit':>11}{'err med':>11}{'err max':>11}{'dot ms':>9}")
    print(head)
    print("-" * len(head))
    for r in rows:
        if "fail" in r:
            print(f"{r['label']:<22}FAILED  {r['fail']}")
            continue
        print(f"{r['label']:<22}{r['params']:<30}{r['pub_ctx']:>13,}{r['ct']:>11,.0f}"
              f"{r['logit']:>11,.0f}{r['err_med']:>11.1e}{r['err_max']:>11.1e}{r['ms']:>9.2f}")
    print(f"\nerror = |decrypted logit - plaintext logit| over {TRIALS} random draws; "
          "sizes in bytes; dot ms = median scorer time for dot + bias.")


if __name__ == "__main__":
    main()

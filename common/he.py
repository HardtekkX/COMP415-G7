"""CKKS configuration and helpers shared by the bank and the scoring service.

Chosen configuration (see PROJECT_PLAN.md, "CKKS configuration"):
  N = 8192, coefficient chain [60, 40, 60], scale 2^40, Galois keys only for
  the rotation steps the encrypted sum needs, and a scorer context with
  Galois keys only (no secret, public or relinearisation keys).

The scorer computes w . x + b: one ciphertext x plaintext multiply (uses the
single 40-bit level), a rotate-and-add sum, and a plaintext add. The sigmoid
runs at the bank after decryption.

Usage:
    bank_ctx, public_ctx = make_contexts(n_features)      # bank, at startup
    scorer_ctx = load_public_context(public_ctx)          # scorer, POST /contexts
    ct = encrypt_vector(bank_ctx, x)                       # bank
    out = encrypted_logit(scorer_ctx, ct, w, b)            # scorer
    logit = decrypt_logit(bank_ctx, out)                   # bank
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence

import tenseal as ts
import tenseal.sealapi as sealapi  # noqa: F401  registers SEAL types (GaloisTool) used below

POLY_MODULUS_DEGREE = 8192
COEFF_MOD_BIT_SIZES = (60, 40, 60)
GLOBAL_SCALE_BITS = 40


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


# True: Galois keys only for the steps the sum needs (2.9 MB public context for
# 10 features). False: TenSEAL's generate_galois_keys(), every power-of-two step.
USE_MINIMAL_GALOIS_KEYS = _env_flag("USE_MINIMAL_GALOIS_KEYS", True)


def rotation_steps(n_slots: int) -> list[int]:
    """Rotation steps TenSEAL's sum uses for an n-slot vector.

    The sum rotates by 1, 2, 4, ... up to (not including) the next power of two
    >= n_slots; 10 slots -> [1, 2, 4, 8], 17 slots -> [1, 2, 4, 8, 16].
    """
    if n_slots < 1:
        raise ValueError("n_slots must be at least 1")
    return [1 << i for i in range(math.ceil(math.log2(n_slots)))]


def _replace_with_minimal_galois_keys(ctx: ts.Context, steps: list[int]) -> None:
    """Overwrite ctx's Galois keys with keys for `steps` only.

    Entirely in memory: TenSEAL's context exposes its SEAL context, secret key
    and Galois key object (by reference), so the sealapi KeyGenerator writes the
    new keys straight into the context. The secret key is never serialised.
    """
    seal_ctx = ctx.data.seal_context()
    # create_galois_keys(list) takes Galois elements, not rotation steps.
    elts = seal_ctx.key_context_data().galois_tool().get_elts_from_steps(steps) if steps else []
    keygen = sealapi.KeyGenerator(seal_ctx, ctx.data.secret_key())
    keygen.create_galois_keys(elts, ctx.data.galois_keys())


def make_contexts(
    n_features: int, minimal_galois_keys: bool | None = None
) -> tuple[ts.Context, bytes]:
    """Build the bank's private context and the serialised public context.

    Returns (bank_ctx, public_ctx_bytes). bank_ctx holds the secret key and must
    stay in the bank service. public_ctx_bytes is what POST /contexts sends:
    Galois keys only, no secret, public or relinearisation keys.
    """
    if minimal_galois_keys is None:
        minimal_galois_keys = USE_MINIMAL_GALOIS_KEYS

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=list(COEFF_MOD_BIT_SIZES),
    )
    ctx.global_scale = 2**GLOBAL_SCALE_BITS
    # Allocates the context's Galois key object (~80 ms); the minimal path then
    # overwrites its contents.
    ctx.generate_galois_keys()
    if minimal_galois_keys:
        _replace_with_minimal_galois_keys(ctx, rotation_steps(n_features))

    public = ctx.serialize(
        save_public_key=False,
        save_secret_key=False,
        save_galois_keys=True,
        save_relin_keys=False,
    )
    return ctx, public


def load_public_context(public_ctx: bytes) -> ts.Context:
    """Scorer side: load the bank's public context. Refuses a private one."""
    ctx = ts.context_from(public_ctx)
    if ctx.is_private():
        raise ValueError("context contains a secret key; refusing it")
    ctx.auto_rescale = True  # flags are not serialised; keep TenSEAL's default explicit
    return ctx


def encrypt_vector(bank_ctx: ts.Context, x: Sequence[float]) -> bytes:
    return ts.ckks_vector(bank_ctx, list(map(float, x))).serialize()


def encrypted_logit(
    scorer_ctx: ts.Context, ciphertext: bytes, weights: Sequence[float], bias: float
) -> bytes:
    """Scorer side: w . x + b on the ciphertext. Returns the summed logit only."""
    enc_x = ts.ckks_vector_from(scorer_ctx, ciphertext)
    if enc_x.size() != len(weights):
        raise ValueError(f"ciphertext has {enc_x.size()} slots, model has {len(weights)} weights")
    return (enc_x.dot(list(map(float, weights))) + float(bias)).serialize()


def decrypt_logit(bank_ctx: ts.Context, encrypted: bytes) -> float:
    return ts.ckks_vector_from(bank_ctx, encrypted).decrypt()[0]

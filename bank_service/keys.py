"""The bank's key material, created on first run and kept in keys_dir.

keys_dir (git-ignored, mode 0700) holds, each file mode 0600:
  he_context.seal      CKKS context with the secret key (no Galois keys)
  public_context.seal  the matching public context for POST /contexts
                       (Galois keys only, no secret key)
  hmac.key             32 random bytes for the pseudonym
  meta.json            feature count the Galois keys were made for
  scorer_context.json  cached context_id (written by ScorerClient)

The secret key and the HMAC key are never sent anywhere.
"""

from __future__ import annotations

import json
import os
import secrets
from dataclasses import dataclass
from pathlib import Path

import tenseal as ts

from common import he

HE_CONTEXT = "he_context.seal"
PUBLIC_CONTEXT = "public_context.seal"
HMAC_KEY = "hmac.key"
META = "meta.json"
SCORER_CONTEXT = "scorer_context.json"
_KEY_FILES = (HE_CONTEXT, PUBLIC_CONTEXT, HMAC_KEY, META)


@dataclass
class BankKeys:
    he_context: ts.Context  # private: holds the secret key
    public_context: bytes   # what the scorer gets
    hmac_key: bytes
    keys_dir: Path

    @property
    def scorer_context_path(self) -> Path:
        return self.keys_dir / SCORER_CONTEXT


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(data)


def load_or_create(keys_dir: Path, n_features: int) -> BankKeys:
    keys_dir = Path(keys_dir)
    present = [name for name in _KEY_FILES if (keys_dir / name).exists()]
    if not present:
        return _create(keys_dir, n_features)
    if len(present) != len(_KEY_FILES):
        missing = sorted(set(_KEY_FILES) - set(present))
        raise RuntimeError(
            f"{keys_dir} is incomplete (missing {missing}); delete the folder to make new keys"
        )
    meta = json.loads((keys_dir / META).read_text())
    if meta["n_features"] != n_features:
        raise RuntimeError(
            f"keys in {keys_dir} were made for {meta['n_features']} features, the model "
            f"now has {n_features}; delete the folder to make new keys"
        )
    keys = BankKeys(
        he_context=he.load_private_context((keys_dir / HE_CONTEXT).read_bytes()),
        public_context=(keys_dir / PUBLIC_CONTEXT).read_bytes(),
        hmac_key=(keys_dir / HMAC_KEY).read_bytes(),
        keys_dir=keys_dir,
    )
    if len(keys.hmac_key) != 32:
        raise RuntimeError(f"{keys_dir / HMAC_KEY} must hold 32 bytes")
    return keys


def _create(keys_dir: Path, n_features: int) -> BankKeys:
    keys_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(keys_dir, 0o700)
    bank_ctx, public = he.make_contexts(n_features)
    hmac_key = secrets.token_bytes(32)
    _write_private(keys_dir / HE_CONTEXT, he.serialize_private_context(bank_ctx))
    _write_private(keys_dir / PUBLIC_CONTEXT, public)
    _write_private(keys_dir / HMAC_KEY, hmac_key)
    # meta.json last: its presence marks a complete set.
    _write_private(keys_dir / META, json.dumps({"n_features": n_features}).encode())
    return BankKeys(bank_ctx, public, hmac_key, keys_dir)


def check_pair(keys: BankKeys, n_features: int) -> None:
    """Encrypt, score with the public context and decrypt, all locally.

    Catches a public_context.seal that does not belong to he_context.seal
    before any applicant is scored. Uses unit weights, not the model's.
    """
    scorer_ctx = he.load_public_context(keys.public_context)
    ct = he.encrypt_vector(keys.he_context, [1.0] * n_features)
    out = he.encrypted_logit(scorer_ctx, ct, [1.0] * n_features, 0.0)
    if abs(he.decrypt_logit(keys.he_context, out) - n_features) > 1e-3:
        raise RuntimeError(f"keys in {keys.keys_dir} do not match; delete the folder")

import numpy as np
import pytest
import tenseal as ts

from common import he

BOTH_PATHS = pytest.mark.parametrize("minimal", [True, False], ids=["minimal-keys", "standard-keys"])


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(415)


@BOTH_PATHS
@pytest.mark.parametrize("n", [10, 20])
def test_dot_product_matches_plaintext(minimal, n, rng):
    bank, public = he.make_contexts(n, minimal_galois_keys=minimal)
    scorer = he.load_public_context(public)
    for _ in range(5):
        x, w, b = rng.normal(size=n), rng.normal(scale=0.5, size=n), float(rng.normal())
        out = he.encrypted_logit(scorer, he.encrypt_vector(bank, x), w, b)
        assert he.decrypt_logit(bank, out) == pytest.approx(w @ x + b, abs=1e-4)


@BOTH_PATHS
def test_scorer_context_cannot_decrypt(minimal):
    bank, public = he.make_contexts(10, minimal_galois_keys=minimal)
    scorer = he.load_public_context(public)
    assert not scorer.is_private()
    assert not scorer.has_public_key()
    assert not scorer.has_relin_keys()
    assert scorer.has_galois_keys()
    ct = he.encrypt_vector(bank, [1.0] * 10)
    with pytest.raises(Exception):
        ts.ckks_vector_from(scorer, ct).decrypt()
    out = he.encrypted_logit(scorer, ct, [1.0] * 10, 0.0)
    with pytest.raises(Exception):
        he.decrypt_logit(scorer, out)


def test_load_public_context_rejects_private_context():
    bank, _ = he.make_contexts(4)
    with pytest.raises(ValueError):
        he.load_public_context(bank.serialize(save_secret_key=True))


@pytest.mark.parametrize(
    "n, steps",
    [(1, []), (2, [1]), (4, [1, 2]), (5, [1, 2, 4]), (10, [1, 2, 4, 8]),
     (16, [1, 2, 4, 8]), (17, [1, 2, 4, 8, 16]), (20, [1, 2, 4, 8, 16])],
)
def test_rotation_steps(n, steps):
    assert he.rotation_steps(n) == steps


def test_minimal_keys_are_sized_to_the_vector_length():
    # 10 features -> steps 1,2,4,8; 20 features also need step 16.
    bank10, public10 = he.make_contexts(10, minimal_galois_keys=True)
    bank20, public20 = he.make_contexts(20, minimal_galois_keys=True)
    assert bank10.data.galois_keys().size() == 4
    assert bank20.data.galois_keys().size() == 5
    assert len(public10) < len(public20)

    # Keys made for 10 slots cannot sum 20 slots.
    scorer10 = he.load_public_context(public10)
    ct20 = he.encrypt_vector(bank10, [1.0] * 20)
    with pytest.raises(Exception):
        he.encrypted_logit(scorer10, ct20, [1.0] * 20, 0.0)


def test_minimal_public_context_is_smaller_than_standard():
    _, minimal = he.make_contexts(10, minimal_galois_keys=True)
    _, standard = he.make_contexts(10, minimal_galois_keys=False)
    assert len(minimal) < len(standard) / 3


def test_weight_length_mismatch_rejected():
    bank, public = he.make_contexts(10)
    scorer = he.load_public_context(public)
    with pytest.raises(ValueError):
        he.encrypted_logit(scorer, he.encrypt_vector(bank, [1.0] * 10), [1.0] * 9, 0.0)


def test_env_flag(monkeypatch):
    monkeypatch.setenv("FLAG", "false")
    assert he._env_flag("FLAG", True) is False
    monkeypatch.setenv("FLAG", "1")
    assert he._env_flag("FLAG", False) is True
    monkeypatch.delenv("FLAG")
    assert he._env_flag("FLAG", True) is True

"""evaluation.run on 10 rows: starts both services, writes both result files.

Writes to a temporary folder, so the committed evaluation/results.* stay as
they are. Needs `python -m training.train` to have been run.
"""

import json

import pytest

from evaluation.run import TEST_SPLIT, run

pytestmark = pytest.mark.skipif(
    not TEST_SPLIT.exists(), reason="test_split.csv missing; run python -m training.train"
)


def test_evaluation_on_10_rows(tmp_path):
    results = run(rows=10, out_dir=tmp_path, quiet=True)

    saved = json.loads((tmp_path / "results.json").read_text())
    assert saved["rows"] == 10 and len(saved["per_row"]) == 10
    assert saved["score_error"]["abs_probability_error"]["max"] < 1e-3
    assert saved["decision_agreement"]["agree"] == 10
    assert saved["ckks"]["poly_modulus_degree"] == 8192
    assert saved["ckks"]["coeff_mod_bit_sizes"] == [60, 40, 60]
    assert saved["versions"]["tenseal"] and saved["machine"]["python"]

    latency = saved["latency_ms"]
    assert latency["steady_state"]["total"]["n"] == 9          # first request reported separately
    assert set(latency["first_request"]) >= {"prepare", "encrypt", "scorer_call", "decrypt", "total"}
    assert latency["scorer_compute"]["n"] == 9                 # all rows matched in /activity

    sizes = saved["size_bytes"]
    assert sizes["context_one_time"] > sizes["ciphertext_per_request"]["median"] > \
        sizes["plaintext_json_per_request"]["median"]
    assert results["model_quality"]["test_auc"] == saved["model_quality"]["test_auc"]

    table = (tmp_path / "results.md").read_text()
    assert "| Measure | Value |" in table
    assert "plaintext baseline is in-process" in table and "two HTTP hops" in table
    assert latency["client_round_trip"]["median"] > latency["steady_state"]["total"]["median"]

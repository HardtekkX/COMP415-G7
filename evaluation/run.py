"""Evaluation: encrypted scoring against plaintext scoring on the test split.

Starts the scoring and bank services as separate uvicorn processes on free
ports (the bank gets a temporary keys folder, so fresh HE and HMAC keys), sends
every row of training/artifacts/test_split.csv through the bank's POST /apply,
and compares with plaintext scoring of the same rows. Measures (PROJECT_PLAN.md,
"What we measure"): score error, decision agreement, latency, size and model
quality. Writes evaluation/results.json (every number, per-row data, CKKS
parameters, versions, date, machine) and evaluation/results.md (one table).

Run:  python -m evaluation.run [--rows N] [--out-dir DIR]
Needs `python -m training.train` first.

The evaluator is not a party to the protocol, so it may read model.json to
compute the plaintext reference. bank_service and ui never read model.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import platform
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from importlib import metadata
from pathlib import Path

import httpx
import numpy as np
import pandas as pd

from bank_service.cli import fake_national_id
from bank_service.config import REPO_ROOT, BankConfig
from bank_service.core import Preprocessing, sigmoid
from common import he

ARTIFACTS = REPO_ROOT / "training" / "artifacts"
TEST_SPLIT = ARTIFACTS / "test_split.csv"
MODEL = ARTIFACTS / "model.json"           # evaluator only, see module docstring
PREPROCESSING = ARTIFACTS / "preprocessing.json"
METRICS = ARTIFACTS / "metrics.json"
DEFAULT_OUT_DIR = Path(__file__).resolve().parent

STEPS = ("prepare", "encrypt", "scorer_call", "decrypt", "total")
# The scorer's GET /activity keeps the last 50 scores; reading it every 40 rows
# means no score drops out between reads.
ACTIVITY_EVERY = 40
PACKAGES = ("tenseal", "numpy", "scikit-learn", "pandas", "fastapi", "uvicorn", "httpx", "pydantic")


# --- services ----------------------------------------------------------------

def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Service:
    """One uvicorn process; output goes to log_path."""

    def __init__(self, app: str, env: dict, log_path: Path):
        self.app, self.env, self.log_path = app, env, log_path
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.proc: subprocess.Popen | None = None

    def start(self, timeout: float = 60) -> None:
        with self.log_path.open("a") as log:
            self.proc = subprocess.Popen(
                [sys.executable, "-m", "uvicorn", self.app,
                 "--host", "127.0.0.1", "--port", str(self.port)],
                cwd=REPO_ROOT, env=self.env, stdout=log, stderr=subprocess.STDOUT,
            )
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.app} exited:\n{self.log_path.read_text()}")
            try:
                if httpx.get(self.url + "/health", timeout=1).status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"{self.app} did not start:\n{self.log_path.read_text()}")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()


# --- plaintext reference ---------------------------------------------------------

class PlaintextModel:
    """w . scale(x) + b and the sigmoid, in-process. Evaluator only."""

    def __init__(self) -> None:
        model = json.loads(MODEL.read_text())
        self.w = np.array(model["weights"], float)
        self.b = float(model["bias"])
        self.pre = Preprocessing.load(PREPROCESSING)

    def score(self, features: dict) -> tuple[float, float]:
        z = float(self.pre.transform(features) @ self.w + self.b)
        return z, sigmoid(z)


# --- statistics ------------------------------------------------------------------

def median_p95(values: list[float]) -> dict:
    return {"median": float(np.median(values)), "p95": float(np.percentile(values, 95)), "n": len(values)}


def machine_info() -> dict:
    return {
        "platform": platform.platform(),
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
    }


def versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def ckks_parameters(n_features: int) -> dict:
    return {
        "scheme": "CKKS",
        "poly_modulus_degree": he.POLY_MODULUS_DEGREE,
        "coeff_mod_bit_sizes": list(he.COEFF_MOD_BIT_SIZES),
        "global_scale": f"2^{he.GLOBAL_SCALE_BITS}",
        "minimal_galois_keys": he.USE_MINIMAL_GALOIS_KEYS,
        "rotation_steps": he.rotation_steps(n_features) if he.USE_MINIMAL_GALOIS_KEYS else "all powers of two",
        "scorer_context_keys": "Galois keys only (no secret, public or relinearisation keys)",
    }


# --- the run ---------------------------------------------------------------------

def collect_activity(scorer: Service, seen: dict) -> None:
    """Add the scorer's /activity records to `seen`, keyed by fingerprint."""
    resp = httpx.get(scorer.url + "/activity", timeout=10)
    resp.raise_for_status()
    for rec in resp.json()["records"]:
        seen[(rec["pseudonym"], rec["ciphertext_sha256"], rec["timestamp"])] = rec


def run(rows: int | None = None, out_dir: Path = DEFAULT_OUT_DIR, quiet: bool = False) -> dict:
    say = (lambda *a: None) if quiet else (lambda *a: print(*a, flush=True))
    data = pd.read_csv(TEST_SPLIT)
    if rows is not None:
        data = data.head(rows)
    n = len(data)
    if n < 2:
        raise ValueError("need at least 2 rows (the first is reported separately)")

    plain = PlaintextModel()
    names = plain.pre.feature_order
    thresholds = BankConfig.load().thresholds

    per_row: list[dict] = []
    activity: dict = {}
    with tempfile.TemporaryDirectory(prefix="he-eval-") as tmp:
        tmp = Path(tmp)
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        scorer = Service("scoring_service.main:app",
                         {**env, "SCORER_DEMO_ENDPOINTS": "true"}, tmp / "scorer.log")
        bank_env = {**env, "BANK_KEYS_DIR": str(tmp / "keys")}
        bank = None
        try:
            scorer.start()
            bank_env["BANK_SCORER_URL"] = scorer.url
            bank = Service("bank_service.main:app", bank_env, tmp / "bank.log")
            bank.start()
            say(f"scorer {scorer.url}, bank {bank.url}; scoring {n} rows")

            with httpx.Client(timeout=60) as http:
                for i, (_, row) in enumerate(data.iterrows(), start=1):
                    features = {k: int(row[k]) for k in names}
                    national_id = fake_national_id(int(row["row_id"]))
                    t0 = time.perf_counter()
                    resp = http.post(bank.url + "/apply",
                                     json={"national_id": national_id, "features": features})
                    round_trip_ms = (time.perf_counter() - t0) * 1000
                    if resp.status_code != 200:
                        raise RuntimeError(f"row {row['row_id']}: /apply {resp.status_code}")
                    body = resp.json()

                    t0 = time.perf_counter()
                    z_plain, p_plain = plain.score(features)
                    plain_ms = (time.perf_counter() - t0) * 1000

                    if not math.isclose(p_plain, float(row["plaintext_probability"]), abs_tol=1e-12):
                        raise RuntimeError(f"row {row['row_id']}: model.json disagrees with test_split.csv")
                    pseudonym_json = json.dumps(
                        {"pseudonym": body["pseudonym"], "features": features}, separators=(",", ":"))
                    per_row.append({
                        "row_id": int(row["row_id"]),
                        "pseudonym": body["pseudonym"],
                        "p_encrypted": body["probability_of_default"],
                        "p_plaintext": p_plain,
                        "logit_encrypted": body["logit"],
                        "logit_plaintext": z_plain,
                        "decision_encrypted": body["decision"],
                        "decision_plaintext": thresholds.decide(p_plain),
                        "ciphertext_bytes": body["ciphertext_bytes"],
                        "encrypted_logit_bytes": body["encrypted_logit_bytes"],
                        "plaintext_json_bytes": len(pseudonym_json.encode()),
                        "timings_ms": body["timings_ms"],
                        "round_trip_ms": round_trip_ms,
                        "plaintext_ms": plain_ms,
                    })
                    if i % ACTIVITY_EVERY == 0 or i == n:
                        collect_activity(scorer, activity)
                        say(f"  {i}/{n}")

                summary = http.get(scorer.url + "/contexts/summary").json()
        finally:
            if bank:
                bank.stop()
            scorer.stop()

    results = summarise(per_row, activity, summary, thresholds, plain)
    write_results(results, Path(out_dir))
    return results


def summarise(per_row, activity, summary, thresholds, plain) -> dict:
    n = len(per_row)
    abs_err = [abs(r["p_encrypted"] - r["p_plaintext"]) for r in per_row]
    offset = [r["logit_encrypted"] - r["logit_plaintext"] for r in per_row]
    disagree = [r["row_id"] for r in per_row if r["decision_encrypted"] != r["decision_plaintext"]]

    first, rest = per_row[0], per_row[1:]
    latency = {s: median_p95([r["timings_ms"][s] for r in rest]) for s in STEPS}

    # Scorer-side compute time, matched to our rows by pseudonym.
    by_pseudonym = {}
    for rec in activity.values():
        by_pseudonym.setdefault(rec["pseudonym"], []).append(rec["compute_ms"])
    compute = [by_pseudonym[r["pseudonym"]][0] for r in per_row if r["pseudonym"] in by_pseudonym]
    if len(compute) != n or any(len(v) != 1 for v in by_pseudonym.values()):
        raise RuntimeError(f"scorer activity covers {len(compute)} of {n} rows")
    scorer_compute = {**median_p95(compute[1:]), "first_ms": compute[0]}

    context_sizes = [c["size_bytes"] for c in summary["contexts"]]
    if len(context_sizes) != 1:
        raise RuntimeError(f"expected one registered context, scorer has {len(context_sizes)}")
    ct = [r["ciphertext_bytes"] for r in per_row]
    # POST /score body: {"pseudonym", "context_id", "ciphertext" (base64)} as JSON.
    context_id = summary["contexts"][0]["context_id"]
    envelope = len(json.dumps({"pseudonym": "0" * 64, "context_id": context_id, "ciphertext": ""},
                              separators=(",", ":")))
    request_body = [envelope + 4 * math.ceil(b / 3) for b in ct]
    plain_json = [r["plaintext_json_bytes"] for r in per_row]

    model_quality = json.loads(METRICS.read_text())
    return {
        "date_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "rows": n,
        "dataset": model_quality.get("dataset"),
        "thresholds": {"approve_below": thresholds.approve_below, "reject_from": thresholds.reject_from},
        "ckks": ckks_parameters(plain.pre.n_features),
        "versions": versions(),
        "machine": machine_info(),
        "score_error": {
            "abs_probability_error": {"median": statistics.median(abs_err),
                                      "mean": statistics.fmean(abs_err), "max": max(abs_err)},
            "logit_offset": {"mean": statistics.fmean(offset), "std": statistics.pstdev(offset),
                             "min": min(offset), "max": max(offset)},
        },
        "decision_agreement": {"agree": n - len(disagree), "rows": n,
                               "share": (n - len(disagree)) / n, "disagreeing_row_ids": disagree},
        "latency_ms": {
            "note": "steady_state: steps as timed inside the bank service (includes the bank -> "
                    "scorer HTTP hop). client_round_trip: POST /apply timed by this script, so two "
                    "HTTP hops (client -> bank -> scorer). 'first_request' includes registering "
                    "the public context with the scorer and is excluded from the medians and p95s. "
                    "The plaintext baseline is in-process (no HTTP).",
            "first_request": first["timings_ms"],
            "steady_state": latency,
            "client_round_trip": median_p95([r["round_trip_ms"] for r in rest])
            | {"first_ms": first["round_trip_ms"]},
            "scorer_compute": {**scorer_compute,
                               "note": f"compute_ms from the scorer's GET /activity, read every "
                                       f"{ACTIVITY_EVERY} rows and combined by fingerprint; covers "
                                       f"all {n} rows"},
            "plaintext_in_process": median_p95([r["plaintext_ms"] for r in per_row]),
        },
        "size_bytes": {
            "context_one_time": context_sizes[0],
            "ciphertext_per_request": median_p95(ct) | {"min": min(ct), "max": max(ct)},
            "score_request_body_per_request": median_p95(request_body),
            "encrypted_logit_per_response": median_p95([r["encrypted_logit_bytes"] for r in per_row]),
            "plaintext_json_per_request": median_p95(plain_json),
            "plaintext_json_note": "compact JSON {pseudonym, features}: what the bank would send without HE",
        },
        "model_quality": {k: model_quality[k] for k in
                          ("test_accuracy", "test_auc", "baseline_accuracy", "decision_threshold",
                           "n_train", "n_test", "seed")},
        "per_row": per_row,
    }


# --- output ----------------------------------------------------------------------

def _kb(b: float) -> str:
    return f"{b / 1000:.0f} KB" if b >= 10_000 else f"{b:.0f} B"


def markdown(r: dict) -> str:
    e, a, lat, s, m = (r["score_error"], r["decision_agreement"], r["latency_ms"],
                       r["size_bytes"], r["model_quality"])
    err, off = e["abs_probability_error"], e["logit_offset"]
    st, first = lat["steady_state"], lat["first_request"]
    sc, pl = lat["scorer_compute"], lat["plaintext_in_process"]
    ct, body, pj = s["ciphertext_per_request"], s["score_request_body_per_request"], s["plaintext_json_per_request"]
    ck, mach = r["ckks"], r["machine"]

    def ms(d):
        return f"{d['median']:.2f} ms / {d['p95']:.2f} ms"

    rows = [
        ("Score error: \\|p_enc − p_plain\\| (median / mean / max)",
         f"{err['median']:.1e} / {err['mean']:.1e} / {err['max']:.1e}"),
        ("Logit offset z_enc − z_plain (mean ± std, range)",
         f"{off['mean']:.2e} ± {off['std']:.1e} ({off['min']:.2e} to {off['max']:.2e})"),
        ("Decision agreement (approve / review / reject)",
         f"{a['agree']} / {a['rows']} ({a['share']:.1%})"),
        *[(f"Latency: {name} (median / p95)", ms(st[key])) for key, name in
          (("prepare", "prepare (scale + pseudonym)"), ("encrypt", "encrypt"),
           ("scorer_call", "scorer call (HTTP + HE dot product)"), ("decrypt", "decrypt + sigmoid + decision"),
           ("total", "total inside the bank"))],
        ("Latency: encrypted total, client round trip of POST /apply (median / p95)",
         ms(lat["client_round_trip"])),
        ("Latency: scorer HE compute only (median / p95)", ms(sc)),
        ("Latency: first request, incl. context registration (client round trip)",
         f"{lat['client_round_trip']['first_ms']:.0f} ms"),
        ("Latency: plaintext scoring, in-process (median / p95)",
         f"{pl['median'] * 1000:.1f} µs / {pl['p95'] * 1000:.1f} µs"),
        ("Size: ciphertext per request (raw / base64 JSON body)",
         f"{_kb(ct['median'])} / {_kb(body['median'])}"),
        ("Size: encrypted logit per response", _kb(s["encrypted_logit_per_response"]["median"])),
        ("Size: public context, sent once", f"{s['context_one_time'] / 1e6:.1f} MB"),
        ("Size: same features as plain JSON", f"{_kb(pj['median'])} (ciphertext ×{ct['median'] / pj['median']:,.0f})"),
        ("Model quality: test accuracy (baseline) / AUC",
         f"{m['test_accuracy']:.3f} ({m['baseline_accuracy']:.3f}) / {m['test_auc']:.3f}"),
    ]
    lines = [
        f"# Evaluation results ({r['rows']} test rows)",
        "",
        "| Measure | Value |",
        "|---|---|",
        *[f"| {k} | {v} |" for k, v in rows],
        "",
        "The plaintext baseline is in-process; the encrypted total (client round trip) includes "
        "two HTTP hops (client → bank → scorer).",
        "",
        f"Latency excludes the first request (reported on its own line). Scorer HE compute "
        f"covers all {r['rows']} rows ({sc['note'].split('; ')[0]}).",
        f"CKKS: N={ck['poly_modulus_degree']}, coefficient chain {ck['coeff_mod_bit_sizes']}, "
        f"scale {ck['global_scale']}, Galois keys for steps {ck['rotation_steps']}. "
        f"TenSEAL {r['versions']['tenseal']}, Python {mach['python']}, {mach['platform']}, "
        f"{mach['cpu_count']} CPUs. Run {r['date_utc']}. Full data: results.json.",
        "",
    ]
    return "\n".join(lines)


def write_results(results: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    (out_dir / "results.md").write_text(markdown(results))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rows", type=int, default=None, help="first N rows only (default: all)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)
    if not TEST_SPLIT.exists():
        print("test_split.csv missing; run python -m training.train", file=sys.stderr)
        return 1
    results = run(args.rows, args.out_dir)
    print()
    print(markdown(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())

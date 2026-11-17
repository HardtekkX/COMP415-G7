# Evaluation results (300 test rows)

| Measure | Value |
|---|---|
| Score error: \|p_enc − p_plain\| (median / mean / max) | 2.5e-07 / 2.4e-07 / 3.9e-07 |
| Logit offset z_enc − z_plain (mean ± std, range) | 1.38e-06 ± 1.7e-07 (1.02e-06 to 1.75e-06) |
| Decision agreement (approve / review / reject) | 300 / 300 (100.0%) |
| Latency: prepare (scale + pseudonym) (median / p95) | 0.06 ms / 0.10 ms |
| Latency: encrypt (median / p95) | 3.86 ms / 4.88 ms |
| Latency: scorer call (HTTP + HE dot product) (median / p95) | 14.72 ms / 16.80 ms |
| Latency: decrypt + sigmoid + decision (median / p95) | 0.42 ms / 0.69 ms |
| Latency: total inside the bank (median / p95) | 19.10 ms / 21.52 ms |
| Latency: encrypted total, client round trip of POST /apply (median / p95) | 21.68 ms / 24.58 ms |
| Latency: scorer HE compute only (median / p95) | 6.44 ms / 7.64 ms |
| Latency: first request, incl. context registration (client round trip) | 161 ms |
| Latency: plaintext scoring, in-process (median / p95) | 68.4 µs / 103.9 µs |
| Size: ciphertext per request (raw / base64 JSON body) | 235 KB / 314 KB |
| Size: encrypted logit per response | 131 KB |
| Size: public context, sent once | 2.9 MB |
| Size: same features as plain JSON | 294 B (ciphertext ×800) |
| Model quality: test accuracy (baseline) / AUC | 0.760 (0.700) / 0.770 |

The plaintext baseline is in-process; the encrypted total (client round trip) includes two HTTP hops (client → bank → scorer).

Latency excludes the first request (reported on its own line). Scorer HE compute covers all 300 rows (compute_ms from the scorer's GET /activity, read every 40 rows and combined by fingerprint).
CKKS: N=8192, coefficient chain [60, 40, 60], scale 2^40, Galois keys for steps [1, 2, 4, 8]. TenSEAL 0.3.18, Python 3.12.3, Linux-5.15.167.4-microsoft-standard-WSL2-x86_64-with-glibc2.39, 16 CPUs. Run 2026-11-17T06:35:57+00:00. Full data: results.json.

"""Comparison screen: encrypted result next to the plaintext result.

Uses the six demo applicants only: their plaintext probability (computed by
training, full float precision) and plaintext decision are stored in
ui/demo_applicants.json. The encrypted-path result comes from the bank's
POST /apply. This page never reads the model weights; the UI has none.
"""

from __future__ import annotations

import json

import altair as alt
import pandas as pd
import streamlit as st

from ui.client import BANK, SCORER, ApplyError, apply, require_demo_data, require_service

STEPS = [
    ("prepare", "Prepare"),
    ("encrypt", "Encrypt"),
    ("scorer_call", "Scorer call"),
    ("decrypt", "Decrypt + sigmoid"),
]

S = st.session_state


def plain_json_bytes(features: dict[str, int]) -> int:
    """Size of the same ten features sent as compact plain JSON."""
    return len(json.dumps({"features": features}, separators=(",", ":")).encode())


def score(applicant: dict) -> dict:
    """Score one demo applicant through the bank and compare. Raises ApplyError."""
    resp = apply(applicant["national_id"], applicant["features"])
    plain_p = applicant["plaintext_probability"]
    enc_p = resp["probability_of_default"]
    return {
        "label": applicant["label"],
        "plain_p": plain_p,
        "enc_p": enc_p,
        "diff": abs(enc_p - plain_p),
        "plain_decision": applicant["expected_decision"],
        "enc_decision": resp["decision"],
        "match": applicant["expected_decision"] == resp["decision"],
        "ciphertext_bytes": resp["ciphertext_bytes"],
        "plain_bytes": plain_json_bytes(applicant["features"]),
        "timings": resp["timings_ms"],
    }


def run(applicants: list[dict]) -> tuple[list[dict], str | None]:
    results = []
    try:
        for a in applicants:
            results.append(score(a))
    except ApplyError as exc:
        return results, str(exc)
    except Exception as exc:  # noqa: BLE001 - never show a stack trace on stage
        return results, f"**Unexpected error in the UI:** {type(exc).__name__}."
    return results, None


def on_applicant_change() -> None:
    S.cmp_one = None
    S.cmp_one_error = None


def timing_chart(timings: dict[str, float]) -> alt.Chart:
    df = pd.DataFrame(
        [{"Step": label, "ms": timings[key]} for key, label in STEPS if key in timings]
    )
    return (
        alt.Chart(df, height=40 * len(df) + 30)
        .mark_bar()
        .encode(
            y=alt.Y("Step:N", sort=None, title=None),
            x=alt.X("ms:Q", title="milliseconds"),
            tooltip=["Step", alt.Tooltip("ms:Q", format=".1f")],
        )
    )


# --- Page -------------------------------------------------------------------------

st.title("Comparison")
st.caption(
    "The same applicant scored two ways: in plaintext at training time, and through "
    "the encrypted path (bank encrypts, scorer computes on ciphertext, bank decrypts)."
)
require_service(BANK)
require_service(SCORER)
DATA = require_demo_data()
APPLICANTS = DATA["applicants"]

if "cmp_one" not in S:
    S.cmp_applicant = 0
    S.cmp_one = None
    S.cmp_one_error = None
    S.cmp_all = None
    S.cmp_all_error = None

# --- One applicant ------------------------------------------------------------------
st.subheader("One applicant")
pick, go = st.columns([4, 1], vertical_alignment="bottom")
with pick:
    st.selectbox(
        "Demo applicant",
        options=range(len(APPLICANTS)),
        format_func=lambda i: f"{APPLICANTS[i]['label']} (test-set row {APPLICANTS[i]['row_id']})",
        key="cmp_applicant",
        on_change=on_applicant_change,
    )
with go:
    if st.button("Compare", type="primary", key="cmp_compare", width="stretch"):
        with st.spinner("Scoring through the encrypted path…"):
            results, S.cmp_one_error = run([APPLICANTS[S.cmp_applicant]])
        S.cmp_one = results[0] if results else None

if S.cmp_one_error:
    st.error(S.cmp_one_error)
elif S.cmp_one is None:
    st.info("Pick an applicant and press **Compare**.")
else:
    r = S.cmp_one
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Plaintext probability", f"{r['plain_p']:.9f}", help="Computed by training on the plaintext features.")
    c2.metric("Encrypted-path probability", f"{r['enc_p']:.9f}", help="Decrypted by the bank from the scorer's encrypted logit.")
    c3.metric("Absolute difference", f"{r['diff']:.2e}", help="CKKS is approximate: a tiny error is expected.")
    if r["match"]:
        c4.metric("Decisions match", f"✅ Yes: {r['enc_decision']}")
    else:
        c4.metric("Decisions match", f"❌ No: {r['plain_decision']} vs {r['enc_decision']}")

    size, timing = st.columns(2, gap="large")
    with size:
        st.markdown("**Request size**")
        ratio = r["ciphertext_bytes"] / r["plain_bytes"]
        st.markdown(
            f"| Sent as | Size |\n|---|---:|\n"
            f"| Plain JSON (the ten features) | {r['plain_bytes']:,} bytes |\n"
            f"| CKKS ciphertext | {r['ciphertext_bytes']:,} bytes |"
        )
        st.caption(
            f"The ciphertext is about {ratio:,.0f}× larger. Its size is fixed by the CKKS "
            f"parameters (N=8192), whatever the feature values are."
        )
    with timing:
        st.markdown("**Timing breakdown**")
        st.altair_chart(timing_chart(r["timings"]), width="stretch")
        st.caption(f"Total {r['timings'].get('total', 0.0):,.1f} ms, measured by the bank.")

# --- All six ------------------------------------------------------------------------
st.divider()
st.subheader("All six demo applicants")
if st.button("Run all six", key="cmp_run_all"):
    with st.spinner("Scoring six applicants through the encrypted path…"):
        S.cmp_all, S.cmp_all_error = run(APPLICANTS)

if S.cmp_all_error:
    st.error(S.cmp_all_error)
if S.cmp_all:
    st.dataframe(
        [
            {
                "Applicant": r["label"],
                "Plaintext p": f"{r['plain_p']:.9f}",
                "Encrypted p": f"{r['enc_p']:.9f}",
                "|Difference|": f"{r['diff']:.2e}",
                "Plaintext decision": r["plain_decision"],
                "Encrypted decision": r["enc_decision"],
                "Match": "✅" if r["match"] else "❌",
                "Ciphertext (KB)": round(r["ciphertext_bytes"] / 1000, 1),
                "Total (ms)": round(r["timings"].get("total", 0.0), 1),
            }
            for r in S.cmp_all
        ],
        key="cmp_all_table",
        hide_index=True,
        width="stretch",
    )
    matches = sum(r["match"] for r in S.cmp_all)
    st.markdown(
        f"**Largest difference {max(r['diff'] for r in S.cmp_all):.2e}** · "
        f"**{matches}/{len(S.cmp_all)} decisions match**"
    )
elif not S.cmp_all_error:
    st.info("Press **Run all six** to score every demo applicant.")

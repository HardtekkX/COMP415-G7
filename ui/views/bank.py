"""Bank screen: applicant form, decrypted probability, decision.

This is the bank's own view, so it shows the national ID and the plaintext
features. The "What left the bank" box shows what actually went to the scorer.

Double-submit guard: the submit button's key carries a generation number that
changes when a request is accepted. A second click sent from the old button
(before the browser has the disabled one) arrives with the old generation and
is ignored, even though Streamlit runs its callback only after the first
request has finished.
"""

from __future__ import annotations

import re
import statistics

import streamlit as st

from ui.client import BANK, ApplyError, apply, require_demo_data, require_service
from ui.features import FEATURES

# Measured on the dev laptop: the scorer call is about 11-14 ms; the first
# call after the scorer starts also uploads the 2.9 MB public context and
# takes about 145 ms. Used until this session has its own measurements.
TYPICAL_SCORER_CALL_MS = 15.0
SLOW_FACTOR = 3.0
SLOW_MIN_EXTRA_MS = 50.0

DECISION_STYLE = {
    "approve": ("green", "✅", "Approve"),
    "review": ("orange", "🔎", "Review"),
    "reject": ("red", "⛔", "Reject"),
}
STEP_LABELS = [
    ("prepare", "Prepare (scale features, compute pseudonym)"),
    ("encrypt", "Encrypt (CKKS)"),
    ("scorer_call", "Scorer call (network + encrypted dot product)"),
    ("decrypt", "Decrypt, sigmoid, decision"),
    ("total", "Total"),
]

S = st.session_state


# Form widgets are keyed per applicant and get their start value through
# `value=`. Setting their values through session_state instead loses them in the
# browser when the first click interrupts the page's first run. They are never
# disabled: toggling `disabled` resets a widget to its `value=`, which would
# throw away the user's edits. Disabling the submit button is enough, since a
# form only sends its values on submit.
def feature_key(name: str) -> str:
    return f"bank_f_{name}_{S.bank_applicant}"


def id_key() -> str:
    return f"bank_national_id_{S.bank_applicant}"


def on_applicant_change() -> None:
    S.bank_result = None
    S.bank_error = None


def on_submit(generation: int) -> None:
    if S.bank_busy or generation != S.bank_gen:
        return                                   # stale second click
    national_id = str(S[id_key()]).strip()
    if not re.fullmatch(r"[0-9]{11}", national_id):
        S.bank_result = None
        S.bank_error = "**National ID must be exactly 11 digits.** Nothing was sent."
        return
    S.bank_gen += 1
    S.bank_busy = True
    S.bank_error = None
    S.bank_request = {
        "national_id": national_id,
        "features": {f.name: int(S[feature_key(f.name)]) for f in FEATURES},
    }


def is_slow(scorer_ms: float) -> tuple[bool, float]:
    """(slow, usual_ms): slow means well above this session's usual scorer call."""
    history = S.bank_scorer_ms
    usual = statistics.median(history) if history else TYPICAL_SCORER_CALL_MS
    slow = scorer_ms >= SLOW_FACTOR * usual and scorer_ms >= usual + SLOW_MIN_EXTRA_MS
    return slow, usual


def run_request() -> None:
    """Send the pending request. Only session_state is touched between the
    HTTP call returning and the state being saved, so a rerun cannot cut in."""
    request = S.bank_request
    try:
        response, error = apply(request["national_id"], request["features"]), None
    except ApplyError as exc:
        response, error = None, str(exc)
    except Exception as exc:  # noqa: BLE001 - never show a stack trace on stage
        response, error = None, f"**Unexpected error in the UI:** {type(exc).__name__}."
    slow = None
    if response is not None:
        scorer_ms = float(response["timings_ms"].get("scorer_call", 0.0))
        is_slow_call, usual = is_slow(scorer_ms)
        if is_slow_call:
            slow = (scorer_ms, usual)
        else:
            S.bank_scorer_ms = (S.bank_scorer_ms + [scorer_ms])[-20:]
        S.bank_result = {"response": response, "request": request, "slow": slow}
    S.bank_error = error
    S.bank_request = None
    S.bank_busy = False


def applicant_name(request: dict) -> str:
    for a in APPLICANTS:
        if a["national_id"] == request["national_id"] and a["features"] == request["features"]:
            return a["label"]
    return "Edited applicant"


# --- Page -------------------------------------------------------------------------

st.title("Bank")
st.caption("The bank's own view: it holds the customer data, the HE secret key and the HMAC key.")
require_service(BANK)
DATA = require_demo_data()
APPLICANTS = DATA["applicants"]
THRESHOLDS = DATA["thresholds"]

if "bank_gen" not in S:
    S.bank_gen = 0
    S.bank_busy = False
    S.bank_request = None
    S.bank_scorer_ms = []
    S.bank_applicant = 0
    S.bank_result = None
    S.bank_error = None

left, right = st.columns([5, 6], gap="large")

with left:
    st.subheader("Applicant")
    st.selectbox(
        "Demo applicant",
        options=range(len(APPLICANTS)),
        format_func=lambda i: f"{APPLICANTS[i]['label']} (test-set row {APPLICANTS[i]['row_id']})",
        key="bank_applicant",
        on_change=on_applicant_change,
        disabled=S.bank_busy,
        help="Fills the form. Every field can still be edited.",
    )
    with st.form("bank_form", border=True):
        current = APPLICANTS[S.bank_applicant]
        st.text_input(
            "National ID", value=current["national_id"], key=id_key(), max_chars=11,
            help="11 digits. Stays in the bank: only its HMAC pseudonym goes to the scorer.",
        )
        cols = st.columns(2)
        for i, f in enumerate(FEATURES):
            with cols[i % 2]:
                start = int(current["features"][f.name])
                if f.options:
                    codes = list(f.options)
                    st.selectbox(f.label, options=codes, index=codes.index(start),
                                 format_func=f.format, key=feature_key(f.name), help=f.help)
                else:
                    st.number_input(f.label, min_value=f.min, max_value=f.max, step=f.step,
                                    value=start, key=feature_key(f.name), help=f.help)
        st.form_submit_button(
            "Scoring…" if S.bank_busy else "Request score",
            type="primary",
            key=f"bank_submit_{S.bank_gen}",
            on_click=on_submit,
            args=(S.bank_gen,),
            disabled=S.bank_busy,
            width="stretch",
        )

with right:
    st.subheader("Decision")
    if S.bank_request is not None:
        with st.spinner("Encrypting, scoring on the ciphertext, decrypting…"):
            run_request()
        st.rerun()

    if S.bank_error:
        st.error(S.bank_error)
    elif S.bank_result is None:
        st.info("Pick an applicant and press **Request score**.")
    else:
        resp, req = S.bank_result["response"], S.bank_result["request"]
        color, icon, word = DECISION_STYLE[resp["decision"]]
        st.caption(f"Result for **{applicant_name(req)}**")
        st.markdown(f"# :{color}-background[{icon} {word}]")
        st.metric("Probability of default", f"{resp['probability_of_default']:.2%}")
        lo, hi = THRESHOLDS["approve_below"], THRESHOLDS["reject_from"]
        st.markdown(
            f"**Approve** below {lo:.0%} · **Review** from {lo:.0%} to under {hi:.0%} · "
            f"**Reject** from {hi:.0%}.  \n"
            f"The thresholds are the bank's own policy. The scorer returned only an "
            f"encrypted logit; the bank decrypted it and applied the sigmoid itself."
        )

        with st.container(border=True):
            st.subheader("What left the bank")
            sent, kept = st.columns(2)
            with sent:
                st.markdown(":green[**✅ Sent to the scorer**]")
                st.markdown(
                    f"- **Pseudonym** (HMAC-SHA256 of the ID with the bank's secret key):\n"
                    f"  `{resp['pseudonym']}`\n"
                    f"- **Encrypted features:** CKKS ciphertext, "
                    f"{resp['ciphertext_bytes'] / 1000:,.1f} KB\n"
                    f"- **Came back:** encrypted logit, "
                    f"{resp['encrypted_logit_bytes'] / 1000:,.1f} KB"
                )
            with kept:
                st.markdown(":red[**🚫 Not sent: stayed in the bank**]")
                rows = "\n".join(
                    f"| {f.label} | {f.format(req['features'][f.name])} |" for f in FEATURES
                )
                st.markdown(
                    f"National ID **{req['national_id']}**\n\n"
                    f"| Feature | Value |\n|---|---|\n{rows}"
                )
            st.caption(
                "The bank's public HE context (about 2.9 MB, no secret key) goes to the "
                "scorer once, when the bank registers, not with each request."
            )

            st.markdown("**Timings**")
            timings = resp["timings_ms"]
            lines = "\n".join(
                f"| {label} | {timings[key]:,.1f} ms |" for key, label in STEP_LABELS if key in timings
            )
            st.markdown(f"| Step | Time |\n|---|---:|\n{lines}")
            if S.bank_result["slow"]:
                took, usual = S.bank_result["slow"]
                st.info(
                    f"The scorer call took {took:,.0f} ms, about {took / usual:.0f}× the usual "
                    f"{usual:,.0f} ms. This was most likely the one-time key registration: the "
                    f"scorer did not know this bank's public context yet (it had just started), "
                    f"so the bank uploaded it before scoring. Later requests skip this step."
                )

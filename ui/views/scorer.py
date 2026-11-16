"""Scorer screen: exactly what the scoring service received and returned.

Built only from the scorer's GET /activity and GET /contexts/summary, which
carry what the scorer may log (pseudonyms, context IDs, sizes, timings,
timestamps) plus, per ciphertext, a 64-character base64 slice from its middle
and a SHA-256 fingerprint. The page reads no local files
and never calls the bank, so it cannot show plaintext even by mistake.
"""

from __future__ import annotations

from datetime import datetime

import streamlit as st

from ui.client import SCORER, ScorerDemoError, require_service, scorer_activity, scorer_contexts_summary

NOT_HELD = [
    "names",
    "national IDs (only a keyed HMAC pseudonym, which it cannot reverse)",
    "feature values (only CKKS ciphertext)",
    "the score: the logit it returns stays encrypted, and only the bank can decrypt it",
    "the decision (approve / review / reject is the bank's own policy)",
]


def local_time(iso: str, with_ms: bool = True) -> str:
    try:
        t = datetime.fromisoformat(iso).astimezone()
    except ValueError:
        return iso
    return t.strftime("%H:%M:%S.%f")[:-3] if with_ms else t.strftime("%H:%M:%S")


def kb(n: int) -> str:
    return f"{n / 1000:,.1f} KB"


def mb(n: int) -> str:
    return f"{n / 1_000_000:,.1f} MB"


st.title("Scorer")
st.caption(
    "Everything the scoring service sees, and nothing else. It holds the model "
    "weights in plaintext and computes on ciphertext it cannot read."
)
require_service(SCORER)

try:
    summary = scorer_contexts_summary()
    records = scorer_activity()
except ScorerDemoError as exc:
    (st.warning if exc.off else st.error)(str(exc))
    st.stop()

# --- Summary line and what the scorer lacks ------------------------------------
top_left, top_right = st.columns([3, 2], gap="large")

with top_left:
    st.button("🔄 Refresh", key="scorer_refresh")      # any click reruns the page
    contexts = summary["contexts"]
    if contexts:
        sizes = ", ".join(mb(c["size_bytes"]) for c in contexts)
        latest = max(c["registered_at"] for c in contexts)
        plural = "context" if summary["count"] == 1 else "contexts"
        st.markdown(
            f"**{summary['count']} public {plural} registered** (max {summary['max_contexts']}) · "
            f"{sizes} · last registered {local_time(latest, with_ms=False)} · "
            f"no secret key in any of them"
        )
    else:
        st.markdown(
            f"**No public context registered yet** (max {summary['max_contexts']}). "
            f"The bank registers one with its first request."
        )

with top_right:
    with st.container(border=True):
        st.markdown(":red[**🚫 The scorer does not have**]")
        st.markdown("\n".join(f"- {item}" for item in NOT_HELD))

# --- Recent requests (full width, so every column fits) -------------------------
st.subheader("Recent requests")
if not records:
    st.info("No requests yet. Score an applicant on the **Bank** page, then press **Refresh**.")
    st.stop()

table = [
    {
        "Time": local_time(r["timestamp"]),
        "Pseudonym": r["pseudonym"][:16] + "…",   # full one in the box below
        "Context ID": r["context_id"],
        "Ciphertext": kb(r["ciphertext_bytes"]),
        "Ciphertext SHA-256": r["ciphertext_sha256"],
        "Compute (ms)": round(r["compute_ms"], 2),
    }
    for r in records
]
# Keyed by the newest record: new requests clear a stale row selection.
event = st.dataframe(
    table,
    key=f"scorer_table_{records[0]['timestamp']}",
    on_select="rerun",
    selection_mode="single-row",
    hide_index=True,
    width="stretch",
)
st.caption(
    f"Last {len(records)} successful scores, newest first. Times are this computer's "
    f"local time. Select a row to see its ciphertext."
)

rows = event.selection.rows if event is not None else []
index = rows[0] if rows else 0
record = records[index]
which = "Selected request" if rows else "Newest request (select a row to change)"
with st.container(border=True):
    st.markdown(
        f"**{which}:** {local_time(record['timestamp'])}, context `{record['context_id']}`  \n"
        f"Pseudonym `{record['pseudonym']}`"
    )
    st.markdown(
        f"**Ciphertext received** ({kb(record['ciphertext_bytes'])}), "
        f"SHA-256 `{record['ciphertext_sha256']}…`, 64 base64 characters from the middle:"
    )
    st.code(record["ciphertext_slice"], language=None, wrap_lines=True)
    st.markdown(
        f"**Encrypted result returned**, SHA-256 `{record['result_sha256']}…`, "
        f"64 base64 characters from the middle:"
    )
    st.code(record["result_slice"], language=None, wrap_lines=True)
    st.caption(
        "Encrypted payload, not the serialization header. CKKS encryption is randomised, so "
        "the same applicant scored twice gives different bytes and fingerprints. The scorer "
        "cannot read either; the decrypted logit exists only at the bank."
    )

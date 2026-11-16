"""Home: what the demo shows and whether both services are up."""

import streamlit as st

from ui.client import BANK, DEMO_APPLICANTS_COMMAND, DEMO_APPLICANTS_PATH, SCORER, SETUP_COMMAND, is_up

st.title("Encrypted credit scoring")
st.markdown(
    """
A bank gets a credit score for a loan applicant from an external scoring
service (playing KKB / Findeks), and the scorer never sees the applicant's data.

1. The **bank** scales the features, replaces the national ID with an HMAC
   pseudonym and encrypts the feature vector with CKKS.
2. The **scorer** computes the logistic-regression logit `w · x + b` on the
   ciphertext and returns it, still encrypted.
3. The **bank** decrypts, applies the sigmoid and decides: approve, review or reject.

Pages: **Bank** (applicant and decision), **Scorer** (exactly what the scorer
received and returned), **Comparison** (encrypted against plaintext result).
"""
)

st.subheader("Services")
for service in (BANK, SCORER):
    if is_up(service):
        st.success(f"{service.name}: up at {service.url}")
    else:
        st.error(f"{service.name}: not reachable at {service.url}. Start it with `{service.start_command}`")

st.subheader("Demo applicants")
if DEMO_APPLICANTS_PATH.exists():
    st.success("`ui/demo_applicants.json` found.")
else:
    st.warning(
        f"`ui/demo_applicants.json` not found. Run `{DEMO_APPLICANTS_COMMAND}` "
        f"(or `{SETUP_COMMAND}` if the model is not trained yet)."
    )

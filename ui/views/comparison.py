"""Comparison screen: encrypted result next to the plaintext result."""

import streamlit as st

from ui.client import BANK, SCORER, require_demo_applicants, require_service

st.title("Comparison")
require_service(BANK)
require_service(SCORER)
applicants = require_demo_applicants()

st.info("Encrypted against plaintext probability: coming next.")

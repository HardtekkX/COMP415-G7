"""Scorer screen: exactly what the scoring service received and returned."""

import streamlit as st

from ui.client import SCORER, require_service

st.title("Scorer")
require_service(SCORER)

st.info("Recent requests (pseudonyms and ciphertext bytes only): coming next.")

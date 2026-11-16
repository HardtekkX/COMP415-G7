"""Bank screen: applicant form, decrypted probability, decision."""

import streamlit as st

from ui.client import BANK, require_demo_applicants, require_service

st.title("Bank")
require_service(BANK)
applicants = require_demo_applicants()

st.info("Applicant form and decision: coming next.")

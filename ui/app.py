"""Streamlit demo: encrypted credit scoring.

Run from the repository root:  streamlit run ui/app.py
Needs both services running (see README.md); each page says which one is missing.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `streamlit run ui/app.py` puts ui/ on sys.path, not the repository root.
_ROOT = str(Path(__file__).resolve().parent.parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import streamlit as st  # noqa: E402

st.set_page_config(page_title="Encrypted credit scoring", page_icon="🔒", layout="wide")

pages = [
    st.Page("views/home.py", title="Home", icon="🏠", default=True),
    st.Page("views/bank.py", title="Bank", icon="🏦"),
    st.Page("views/scorer.py", title="Scorer", icon="🧮"),
    st.Page("views/comparison.py", title="Comparison", icon="⚖️"),
]
st.navigation(pages).run()

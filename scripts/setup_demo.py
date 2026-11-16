"""One-step demo setup: train the model, then pick the demo applicants.

Run from anywhere:  python scripts/setup_demo.py
Equivalent to `python -m training.train && python -m ui.demo_applicants`.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for module in ("training.train", "ui.demo_applicants"):
    print(f"==> python -m {module}", flush=True)
    result = subprocess.run([sys.executable, "-m", module], cwd=ROOT)
    if result.returncode != 0:
        sys.exit(result.returncode)
print("==> Done. Start the services and run: streamlit run ui/app.py")

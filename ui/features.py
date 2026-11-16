"""The ten applicant features as the Bank screen shows them.

Limits copy bank_service/schemas.py ApplicantFeatures and the category codes
copy training/train.py. The UI may not import either (thin-client rule), so
tests/test_ui.py checks that this file still matches both.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Feature:
    name: str
    label: str
    help: str
    min: int
    max: int
    # code -> readable label for category fields; empty for plain numbers.
    options: dict[int, str] = field(default_factory=dict)
    step: int = 1

    def format(self, value: int) -> str:
        if self.options:
            return self.options.get(value, str(value))
        return f"{value:,}"


YES_NO = {0: "No", 1: "Yes"}

FEATURES: list[Feature] = [
    Feature("duration_months", "Loan duration (months)", "Length of the loan.", 1, 120),
    Feature("credit_amount", "Credit amount (DM)", "Amount requested, in Deutsche Mark.",
            1, 1_000_000, step=100),
    Feature("installment_rate", "Instalment rate (1–4)",
            "Instalment as a share of disposable income: 1 lowest, 4 highest.", 1, 4),
    Feature("existing_credits", "Existing credits at this bank",
            "Number of credits the applicant already has here.", 0, 10),
    Feature("checking_status", "Checking account", "Balance of the current account.", 0, 3,
            {0: "Overdrawn (< 0 DM)", 1: "0–200 DM", 2: "200 DM or more", 3: "No checking account"}),
    Feature("savings_band", "Savings", "Savings account or bonds.", 0, 4,
            {0: "Under 100 DM", 1: "100–500 DM", 2: "Unknown / no savings",
             3: "500–1000 DM", 4: "1000 DM or more"}),
    Feature("employment_band", "Employed for", "Time in the present job.", 0, 4,
            {0: "Unemployed", 1: "Under 1 year", 2: "1–4 years", 3: "4–7 years", 4: "7 years or more"}),
    Feature("other_installment_plans", "Instalments elsewhere",
            "Pays instalments at another bank or a store.", 0, 1, YES_NO),
    Feature("owns_home", "Owns their home", "Lives in a home they own.", 0, 1, YES_NO),
    Feature("has_guarantor", "Has a guarantor", "A guarantor backs the loan.", 0, 1, YES_NO),
]

BY_NAME = {f.name: f for f in FEATURES}

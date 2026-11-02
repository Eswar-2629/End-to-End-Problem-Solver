"""Generate a clearly labeled synthetic e-commerce churn dataset."""
from pathlib import Path
import numpy as np
import pandas as pd

RANDOM_SEED = 42


def generate_synthetic_data(n_rows: int = 6000, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    customer_ids = [f"CUST-{i:05d}" for i in range(1, n_rows + 1)]
    tenure_months = rng.integers(1, 73, n_rows)
    plan = rng.choice(["Basic", "Standard", "Premium"], n_rows, p=[0.45, 0.38, 0.17])
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n_rows, p=[0.56, 0.28, 0.16])
    payment = rng.choice(["Electronic check", "Credit card", "Bank transfer"], n_rows, p=[0.42, 0.36, 0.22])
    support_calls = np.clip(rng.poisson(1.8, n_rows), 0, 8)
    late_payments = np.clip(rng.poisson(0.7, n_rows), 0, 5)
    monthly_spend = np.clip(
        35 + 8 * (plan == "Standard") + 25 * (plan == "Premium")
        + rng.normal(0, 12, n_rows), 15, 180
    )
    satisfaction = np.clip(rng.normal(7.0, 1.8, n_rows), 1, 10).round(1)
    age = np.clip(rng.normal(43, 13, n_rows).round(), 18, 75).astype(int)

    logit = (
        -0.65
        + 0.85 * (contract == "Month-to-month")
        - 0.40 * (contract == "Two year")
        + 0.035 * support_calls
        + 0.50 * late_payments
        - 0.018 * tenure_months
        - 0.20 * (satisfaction - 7)
        + 0.004 * np.maximum(monthly_spend - 60, 0)
        + 0.30 * (payment == "Electronic check")
        - 0.0005 * (age - 43)
        + rng.normal(0, 0.22, n_rows)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    churned = rng.binomial(1, churn_probability)

    # Inject a small amount of realistic messiness.
    satisfaction[rng.choice(n_rows, size=int(0.025 * n_rows), replace=False)] = np.nan
    payment[rng.choice(n_rows, size=18, replace=False)] = "bank transfer"
    contract[rng.choice(n_rows, size=12, replace=False)] = "month-to-month"
    duplicate_rows = rng.choice(n_rows, size=35, replace=False)

    df = pd.DataFrame({
        "customer_id": customer_ids,
        "age": age,
        "plan": plan,
        "contract_type": contract,
        "payment_method": payment,
        "tenure_months": tenure_months,
        "monthly_spend": monthly_spend.round(2),
        "support_calls": support_calls,
        "late_payments": late_payments,
        "satisfaction_score": satisfaction,
        "churned": churned.astype(int),
    })
    df = pd.concat([df, df.iloc[duplicate_rows].copy()], ignore_index=True)
    return df


def save_synthetic_data(output_path: str = "data/raw/synthetic_customer_churn.csv") -> pd.DataFrame:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_data()
    df.to_csv(path, index=False)
    return df


if __name__ == "__main__":
    frame = save_synthetic_data()
    print(f"Created synthetic dataset: {len(frame):,} rows")

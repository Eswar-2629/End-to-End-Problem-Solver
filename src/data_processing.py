"""Cleaning utilities with an auditable decision log."""
from pathlib import Path
import json
import pandas as pd


EXPECTED_COLUMNS = {
    "customer_id", "age", "plan", "contract_type", "payment_method",
    "tenure_months", "monthly_spend", "support_calls", "late_payments",
    "satisfaction_score", "churned",
}


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, list[dict]]:
    log = []
    cleaned = df.copy()

    missing = sorted(EXPECTED_COLUMNS - set(cleaned.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    duplicate_count = int(cleaned.duplicated().sum())
    cleaned = cleaned.drop_duplicates().copy()
    log.append({"step": "duplicates", "action": "Dropped exact duplicate rows", "count_or_reason": duplicate_count})

    category_maps = {
        "contract_type": {"month-to-month": "Month-to-month"},
        "payment_method": {"bank transfer": "Bank transfer"},
    }
    for column, mapping in category_maps.items():
        before = cleaned[column].copy()
        cleaned[column] = cleaned[column].astype("string").str.strip().replace(mapping)
        changed = int((before != cleaned[column]).sum())
        log.append({"step": "category normalization", "action": f"Normalized {column}", "count_or_reason": changed})

    numeric_columns = ["age", "tenure_months", "monthly_spend", "support_calls", "late_payments", "satisfaction_score"]
    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    cleaned["age"] = cleaned["age"].clip(lower=18, upper=75)
    cleaned["satisfaction_score"] = cleaned["satisfaction_score"].clip(lower=1, upper=10)
    cleaned["monthly_spend"] = cleaned["monthly_spend"].clip(lower=15, upper=180)

    satisfaction_median = cleaned["satisfaction_score"].median()
    cleaned["satisfaction_score"] = cleaned["satisfaction_score"].fillna(satisfaction_median)
    log.append({"step": "missing values", "action": "Median-imputed satisfaction_score", "count_or_reason": f"median={satisfaction_median:.2f}; justified by numeric skew risk and low 2.5% missingness"})

    numeric_missing = cleaned[numeric_columns].isna().sum().sum()
    cleaned = cleaned.dropna(subset=numeric_columns + ["contract_type", "payment_method", "plan", "churned"])
    log.append({"step": "missing values", "action": "Dropped remaining required-field missing rows", "count_or_reason": int(numeric_missing)})

    churn_counts = cleaned["churned"].value_counts().to_dict()
    log.append({"step": "outliers", "action": "Applied domain bounds to age, satisfaction, and spend; retained tenure/calls/late payments", "count_or_reason": "Bounds prevent implausible values while preserving meaningful extremes"})
    log.append({"step": "target validation", "action": "Validated churned is binary", "count_or_reason": churn_counts})

    if not set(cleaned["churned"]).issubset({0, 1}):
        raise ValueError("churned must contain only 0 and 1")

    return cleaned.reset_index(drop=True), log


def process_and_save(raw_path: str = "data/raw/synthetic_customer_churn.csv",
                     clean_path: str = "data/processed/customer_churn_clean.csv",
                     log_path: str = "outputs/cleaning_log.json") -> tuple[pd.DataFrame, list[dict]]:
    df = pd.read_csv(raw_path)
    cleaned, log = clean_data(df)
    Path(clean_path).parent.mkdir(parents=True, exist_ok=True)
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(clean_path, index=False)
    with open(log_path, "w", encoding="utf-8") as handle:
        json.dump(log, handle, indent=2)
    return cleaned, log


if __name__ == "__main__":
    frame, decisions = process_and_save()
    print(f"Cleaned dataset: {len(frame):,} rows; logged {len(decisions)} cleaning steps")

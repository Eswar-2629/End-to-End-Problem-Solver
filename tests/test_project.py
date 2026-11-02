import pandas as pd
from src.data_generation import generate_synthetic_data
from src.data_processing import clean_data


def test_synthetic_data_is_valid():
    df = generate_synthetic_data(n_rows=500, seed=42)
    assert len(df) >= 500
    assert set(df["churned"].unique()) <= {0, 1}


def test_cleaning_handles_duplicates_and_categories():
    df = generate_synthetic_data(n_rows=500, seed=42)
    cleaned, log = clean_data(df)
    assert len(cleaned) == 500
    assert cleaned["contract_type"].str.strip().eq("month-to-month").sum() == 0
    assert not cleaned.duplicated().any()
    assert cleaned["satisfaction_score"].notna().all()
    assert len(log) >= 6


def test_holdout_metrics_exist_and_are_realistic():
    frame = pd.read_csv("outputs/models/model_metrics.json") if False else None
    import json
    with open("outputs/models/model_metrics.json", encoding="utf-8") as handle:
        metrics = json.load(handle)
    assert 0 <= metrics["baseline_test"]["roc_auc"] <= 1
    assert 0 <= metrics["improved_test"]["roc_auc"] <= 1
    assert metrics["champion"] == "baseline_logistic_regression"

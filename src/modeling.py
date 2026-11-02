"""Leakage-safe baseline and improved churn models."""
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_SEED = 42


def build_models(X: pd.DataFrame):
    categorical = X.select_dtypes(include=["object", "string"]).columns.tolist()
    numeric = X.select_dtypes(include="number").columns.tolist()
    preprocessing = ColumnTransformer([
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("numeric", StandardScaler(), numeric),
    ])

    baseline = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=RANDOM_SEED)),
    ])
    improved = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", RandomForestClassifier(n_estimators=300, min_samples_leaf=4,
                                              class_weight="balanced_subsample", random_state=RANDOM_SEED,
                                              n_jobs=-1)),
    ])
    return baseline, improved


def train_and_evaluate(df: pd.DataFrame, output_dir: str = "outputs/models") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    feature_columns = [
        "age", "plan", "contract_type", "payment_method", "tenure_months",
        "monthly_spend", "support_calls", "late_payments", "satisfaction_score",
    ]
    X = df[feature_columns]
    y = df["churned"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_SEED
    )

    baseline, improved = build_models(X_train)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    baseline_cv = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc")
    improved_cv = cross_val_score(improved, X_train, y_train, cv=cv, scoring="roc_auc")

    baseline.fit(X_train, y_train)
    improved.fit(X_train, y_train)

    def evaluate(model, name):
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average="binary", zero_division=0)
        return {
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
            "confusion_matrix": confusion_matrix(y_test, predictions).tolist(),
            "classification_report": classification_report(y_test, predictions, output_dict=True, zero_division=0),
        }

    baseline_metrics = evaluate(baseline, "baseline")
    improved_metrics = evaluate(improved, "improved")
    permutation = permutation_importance(improved, X_test, y_test, n_repeats=10,
                                         random_state=RANDOM_SEED, scoring="roc_auc")
    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance_mean": permutation.importances_mean,
        "importance_std": permutation.importances_std,
    }).sort_values("importance_mean", ascending=False)
    importance.to_csv(output / "feature_importance.csv", index=False)

    results = {
        "random_seed": RANDOM_SEED,
        "train_rows": len(X_train), "test_rows": len(X_test),
        "target_rate": float(y.mean()),
        "baseline_cv_roc_auc_mean": float(baseline_cv.mean()),
        "baseline_cv_roc_auc_std": float(baseline_cv.std()),
        "improved_cv_roc_auc_mean": float(improved_cv.mean()),
        "improved_cv_roc_auc_std": float(improved_cv.std()),
        "baseline_test": baseline_metrics,
        "improved_test": improved_metrics,
    }
    champion = baseline if baseline_metrics["roc_auc"] >= improved_metrics["roc_auc"] else improved
    joblib.dump(champion, output / "churn_model.joblib")
    results["champion"] = "baseline_logistic_regression" if champion is baseline else "random_forest"
    with open(output / "model_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


if __name__ == "__main__":
    frame = pd.read_csv("data/processed/customer_churn_clean.csv")
    print(json.dumps(train_and_evaluate(frame), indent=2))

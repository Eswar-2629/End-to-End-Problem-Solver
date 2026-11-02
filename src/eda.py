"""Exploratory analysis, hypothesis tests, and portfolio visualizations."""
from pathlib import Path
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, ttest_ind

sns.set_theme(style="whitegrid")


def run_eda(df: pd.DataFrame, output_dir: str = "outputs/eda") -> dict:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    results = {}
    contract_table = pd.crosstab(df["contract_type"], df["churned"])
    chi2, p_value, _, _ = chi2_contingency(contract_table)
    results["contract_churn_chi_square"] = {"chi2": float(chi2), "p_value": float(p_value)}

    churned = df[df["churned"] == 1]["tenure_months"]
    retained = df[df["churned"] == 0]["tenure_months"]
    t_stat, t_p = ttest_ind(churned, retained, equal_var=False)
    results["tenure_t_test"] = {"t_statistic": float(t_stat), "p_value": float(t_p)}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    sns.countplot(data=df, x="contract_type", hue="churned", ax=axes[0])
    axes[0].set_title("Churn by Contract Type")
    axes[0].tick_params(axis="x", rotation=20)

    sns.histplot(data=df, x="tenure_months", hue="churned", bins=30, kde=True, ax=axes[1])
    axes[1].set_title("Tenure Distribution")

    sns.boxplot(data=df, x="churned", y="monthly_spend", ax=axes[2])
    axes[2].set_title("Monthly Spend by Outcome")

    payment_rate = df.groupby("payment_method")["churned"].mean().sort_values(ascending=False)
    payment_rate.plot(kind="bar", ax=axes[3], color="#d95f02")
    axes[3].set_title("Churn Rate by Payment Method")
    axes[3].set_ylabel("Churn rate")
    axes[3].tick_params(axis="x", rotation=25)

    sns.histplot(data=df, x="satisfaction_score", hue="churned", bins=20, ax=axes[4])
    axes[4].set_title("Satisfaction Distribution")

    numeric = df.select_dtypes(include="number").drop(columns=["churned"])
    corr = numeric.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", ax=axes[5], center=0)
    axes[5].set_title("Numeric Correlations")

    fig.tight_layout()
    fig.savefig(output / "eda_overview.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    segment_summary = df.groupby(["contract_type", "payment_method"]).agg(
        customers=("customer_id", "count"),
        churn_rate=("churned", "mean"),
        avg_tenure=("tenure_months", "mean"),
        avg_spend=("monthly_spend", "mean"),
    ).reset_index()
    segment_summary.to_csv(output / "segment_summary.csv", index=False)

    with open(output / "eda_statistics.json", "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results


if __name__ == "__main__":
    import pandas as pd
    frame = pd.read_csv("data/processed/customer_churn_clean.csv")
    print(run_eda(frame))

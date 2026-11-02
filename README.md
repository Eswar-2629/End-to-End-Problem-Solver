# Customer Churn End-to-End Problem Solver

## Problem
A subscription-commerce retention manager needs to decide which customers to contact and which contract/payment experiences to change. The stakeholder cannot intervene after churn has occurred, so the decision is an early-risk action list and a focused retention experiment.

**Hypothesis:** Customers on month-to-month contracts churn at materially higher rates than customers on annual contracts, especially when payment or satisfaction friction is present.

**Primary success metric:** Held-out ROC-AUC, while precision and recall determine the practical cost of retention outreach. A deployed threshold would also be tested against incremental churn reduction, revenue saved, and false-contact cost rather than accuracy alone.

## Data
This project uses a **synthetic dataset**, generated locally by `src/data_generation.py`. It contains 6,000 cleaned customer records after duplicate removal, with contract, plan, payment, tenure, spend, support, satisfaction, and churn fields. Synthetic data is explicitly labeled in the dashboard and README. It is useful for demonstrating the workflow, but it must not be presented as evidence from real customers or as a production model.

The generator deliberately creates a realistic monotonic relationship between month-to-month contracts, late payments, support calls, tenure, satisfaction, and churn. It also adds duplicate rows, inconsistent category spelling, and missing satisfaction scores so cleaning decisions are exercised in the same run.

## Project Structure

```text
End-to-End-Problem-Solver/
├── app.py                         # Streamlit dashboard
├── run_pipeline.py                # One-command pipeline
├── requirements.txt
├── src/
│   ├── data_generation.py         # Synthetic data generator
│   ├── data_processing.py         # Cleaning and decision log
│   ├── eda.py                     # Tests and visualizations
│   └── modeling.py                # Baseline and random forest models
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
│   ├── eda/
│   └── models/
├── tests/
│   └── test_project.py
└── README.md
```

## Approach

### Cleaning decisions

- Removed exact duplicate records: 35 duplicated rows were present before cleaning, leaving 6,000 clean records.
- Standardized `month-to-month` and `bank transfer` category variants.
- Converted numeric fields to numeric types and enforced realistic domain bounds for age, satisfaction, and spend.
- Median-imputed satisfaction when 2.5% of values were missing. Median imputation was selected as a simple, interpretable method for a small amount of numeric missingness.
- Kept meaningful extremes in support calls, late payments, and tenure rather than discarding them automatically.
- Validated that churn has exactly binary values: 0 or 1.

### Leakage guard

The 6,000 records were split into 4,800 training rows and 1,200 holdout rows using stratification and seed 42. Scaling and one-hot encoding are contained inside each `sklearn.pipeline.Pipeline`, and five-fold cross-validation was run only on training data. The holdout test set was never used for fitting, category learning, or hyperparameter selection.

### EDA and statistical tests

A chi-square test compared churn across contract types: $\chi^2=257.35$, $p=1.31\times10^{-56}$. A Welch t-test compared tenure for churned and retained customers: $t=-11.00$, $p=7.34\times10^{-28}$. These results support meaningful synthetic differences in contract type and tenure, while they do not establish real-world causality.

## Findings

The observed synthetic churn rate is 42.9%. Month-to-month contracts have the highest segment rate, with Electronic Check customers at 55.0% in that contract/payment combination; One-Year Electronic Check is 37.6% and Two-Year Credit Card is 23.1%.

The logistic baseline performed best:

| Model | 5-fold CV ROC-AUC | Holdout ROC-AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Logistic baseline | 0.7042 ± 0.0112 | 0.7151 | 0.6011 | 0.6583 | 0.6284 |
| Random forest | 0.6879 ± 0.0093 | 0.6868 | 0.5863 | 0.5806 | 0.5834 |

The 1,200-row holdout confusion matrix for the logistic champion is `[[460, 225], [176, 339]]`: 339 churners and 460 retained customers correctly classified at a 0.50 probability threshold. The random forest importance ranking lists contract type, late payments, satisfaction, and tenure ahead of demographic or transaction-only features.

## Recommendations

1. **Prioritize month-to-month customers for renewal campaigns.** Evidence: strongest contract-group association and the highest segment churn in the synthetic data. Expected impact: avoid 5-15% of targeted churn if a retention offer causes a small uplift; final impact should be measured with an A/B test.
2. **Contact high-risk accounts proactively.** Use the model as a prioritized queue, beginning with the dashboard threshold. At a 60% threshold, outreach volume should be reviewed against support capacity; a simulation using predicted retention savings is required before setting the threshold.
3. **Test payment-reminder experiments for electronic-check users.** In the synthetic data, this method is associated with higher churn, but it may also proxy for other customer behaviors. Test reminders against a holdout control and measure incremental retention.
4. **Build a 30- and 60-day onboarding cohort program.** Churned synthetic customers have significantly lower tenure. Estimate impact only after validating a causal sequence and avoiding offers sent after the outcome is already likely.

Assumptions: average retained account value, offer cost, and incrementality were not supplied because this is synthetic data. Do not use revenue numbers without real customer economics.

## Setup and Run

Requires Python 3.11+.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_pipeline.py
pytest -q
streamlit run app.py
```

Dashboard URL:

```text
http://localhost:8501
```

A successful run creates cleaned CSV data, seven cleaning-log entries, `eda_overview.png`, segment CSV/JSON results, model metrics, feature importance, and the trained champion model under `outputs/`. The dashboard displays KPI cards, contract/payment charts, tenure and segment views, a customer-level table, and recommendations.

## Limitations

- All observations and relationships come from synthetic data and cannot establish real-world causal effects.
- The holdout threshold is fixed at 0.50; production threshold selection requires cost-benefit simulation.
- The target class is only moderately imbalanced, but costs are asymmetric and the model has no real retention revenue estimate.
- No demographic fairness analysis was available beyond age; real deployment should assess protected-group disparities and false-positive burden.
- Randomly generated customer behavior can contain modelable noise even though it is intentionally structured.

## Next Steps

1. Replace the synthetic source with a licensed/permissioned customer dataset and revalidate each cleaning decision.
2. Define offer cost, gross margin, retention-value horizon, and churn uplift target.
3. Run threshold and calibration analysis, then a randomized retention experiment.
4. Monitor drift in contract mix, payment method, satisfaction response, tenure, and calibration.
5. Package training and dashboard deployment in CI/CD with data-quality gates and model cards.

## Tech Used

Python, pandas, NumPy, matplotlib, seaborn, SciPy, scikit-learn, Plotly, Streamlit, joblib, and pytest.

## Interview Talking Points

- **Baseline-first comparison:** I used a leakage-safe train/test split and pipelines, then compared interpretable logistic regression with random forest because the simpler model won.
- **Business framing:** The model does not decide churn by itself; it ranks customers so retention capacity can be allocated and measured experimentally.
- **Cleaning transparency:** Every transformation is logged and questionable data decisions have domain-based explanations.
- **Model honesty:** The synthetic holdout ROC-AUC is 0.7151 for the logistic champion; I do not present this as real-world performance.
- **Recommendations:** All actions are tied to evidence and carry assumptions plus an experiment requirement rather than claimed guaranteed impact.

### Likely Follow-up: Why this model?
Logistic regression is more interpretable, calibrated well for this generated linear relationship, and outperformed random forest on cross-validation and holdout ROC-AUC. A deployed shortlist can start here while random forest remains a benchmark.

### Likely Follow-up: How would you deploy it?
I would batch-score a versioned feature table on a schedule, write risk scores and model version to a warehouse, expose a restricted dashboard/CRM queue, and monitor calibration, drift, latency, and segment-level performance.

### Likely Follow-up: What would you do with more data?
I would add customer history, product usage, complaint text, service interactions, price changes, acquisition channel, and treatment history; then run time-aware validation, calibration, uplift modeling, and fairness analysis.

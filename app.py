"""Interactive Streamlit dashboard for the churn portfolio case study."""
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(page_title="Customer Churn Decision Hub", layout="wide")
st.title("Customer Churn Decision Hub")
st.caption("Synthetic dataset for portfolio demonstration. Outcomes are scenario-generated, not observed customer behavior.")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/customer_churn_clean.csv")

@st.cache_resource
def load_model():
    return joblib.load("outputs/models/churn_model.joblib")

try:
    df = load_data()
except FileNotFoundError:
    st.error("Run `python run_pipeline.py` before starting the dashboard.")
    st.stop()

with st.sidebar:
    st.header("Filters")
    contract = st.multiselect("Contract type", sorted(df["contract_type"].unique()), default=None)
    plan = st.multiselect("Plan", sorted(df["plan"].unique()), default=None)
    risk_threshold = st.slider("High-risk probability threshold", 0.05, 0.95, 0.60, 0.05)

filtered = df.copy()
if contract:
    filtered = filtered[filtered["contract_type"].isin(contract)]
if plan:
    filtered = filtered[filtered["plan"].isin(plan)]

features = ["age", "plan", "contract_type", "payment_method", "tenure_months", "monthly_spend", "support_calls", "late_payments", "satisfaction_score"]
try:
    model = load_model()
    filtered = filtered.copy()
    filtered["risk_score"] = model.predict_proba(filtered[features])[:, 1]
except FileNotFoundError:
    st.warning("Model not found. Showing observed rates only.")
    filtered["risk_score"] = 0.0

k1, k2, k3, k4 = st.columns(4)
k1.metric("Customers", f"{len(filtered):,}")
k2.metric("Observed churn rate", f"{filtered['churned'].mean():.1%}")
k3.metric("High-risk customers", f"{(filtered['risk_score'] >= risk_threshold).sum():,}")
k4.metric("Average monthly spend", f"${filtered['monthly_spend'].mean():,.0f}")

left, right = st.columns(2)
contract_chart = filtered.groupby("contract_type", as_index=False)["churned"].mean()
left.plotly_chart(px.bar(contract_chart, x="contract_type", y="churned", title="Observed Churn by Contract", labels={"churned": "Churn rate"}), use_container_width=True)
payment_chart = filtered.groupby("payment_method", as_index=False)["churned"].mean().sort_values("churned", ascending=False)
right.plotly_chart(px.bar(payment_chart, x="payment_method", y="churned", title="Observed Churn by Payment Method", labels={"churned": "Churn rate"}), use_container_width=True)

left, right = st.columns(2)
left.plotly_chart(px.histogram(filtered, x="tenure_months", color="churned", nbins=30, barmode="overlay", title="Tenure vs. Outcome", opacity=0.75), use_container_width=True)
segment = filtered.groupby(["contract_type", "payment_method"], as_index=False).agg(customers=("customer_id", "count"), churn_rate=("churned", "mean"))
right.plotly_chart(px.scatter(segment, x="customers", y="churn_rate", size="customers", color="contract_type", hover_data=["payment_method"], title="Churn Segments"), use_container_width=True)

st.subheader("Recommendations")
m2m = contract_chart.loc[contract_chart["contract_type"] == "Month-to-month", "churned"]
m2m_rate = float(m2m.iloc[0]) if len(m2m) else float("nan")
base_rate = filtered["churned"].mean()
recommendations = [
    ("1", "Target month-to-month customers with a renewal incentive", f"Observed segment churn: {m2m_rate:.1%}. Validate with a controlled offer experiment."),
    ("2", "Prioritize high-risk accounts for proactive support", f"{(filtered['risk_score'] >= risk_threshold).sum():,} accounts exceed the {risk_threshold:.0%} risk threshold in the current view."),
    ("3", "Add payment reminders for electronic-check customers", "Use payment method as a retention signal and test reminder timing; the relationship may reflect customer or channel mix."),
    ("4", "Create an early-tenure onboarding outreach", "Compare tenures using observed cohort data before choosing an exact outreach day."),
]
for priority, action, evidence in recommendations:
    st.markdown(f"**{priority}. {action}**  \n{evidence}")

with st.expander("Customer-level view"):
    display_columns = ["customer_id", "contract_type", "plan", "tenure_months", "monthly_spend", "support_calls", "late_payments", "satisfaction_score", "churned", "risk_score"]
    st.dataframe(filtered[display_columns].sort_values("risk_score", ascending=False), use_container_width=True)

st.caption("Recommendations are decision hypotheses. Expected impact requires uplift measurement in an A/B test.")

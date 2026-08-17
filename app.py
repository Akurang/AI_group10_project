"""
app.py -- E-Commerce Churn AI

Dashboard, Customer Analysis, AI Retention Strategy, and About pages, built
around the ACTUAL pipeline from final_project.ipynb + driver.ipynb:

  Customer Data -> Preprocessing + Feature Engineering -> Random Forest
                -> Churn Probability -> Local Explanation -> LLM
                -> Retention Strategy

Model: uses models/randomforest_fair.pkl if present (the audited version
without Gender/MaritalStatus), otherwise falls back to models/churn_model.pkl
(your current baseline model). A banner tells you which one is active.

Scaler: requires models/scaler.pkl (see README -- your notebook didn't save
this yet). Until it's added, the app runs in DEMO MODE with synthetic data
and an approximate in-memory scaler so you can still click through the UI --
predictions in that state will NOT match your real trained model.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv
load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from preprocessing import (
    engineer_features,
    encode_categoricals,
    prepare_model_input,
    NUMERIC_COLUMNS,
    CATEGORICAL_COLUMNS,
    CATEGORY_VALUES,
)
from prediction import predict_churn, risk_bucket, load_model, load_scaler, DEPLOYMENT_THRESHOLD
from explanation import explain_prediction
from llm import generate_retention_strategy

# DATA_PATH_CSV = os.path.join(os.path.dirname(__file__), "data", "ecommerce.csv")
DATA_PATH_XLSX = os.path.join(os.path.dirname(__file__), "data", "Test Data.xlsx")

st.set_page_config(page_title="E-Commerce Churn AI", page_icon="🛒", layout="wide")


# ----------------------------------------------------------------------------
# DATA / MODEL LOADING
# ----------------------------------------------------------------------------
@st.cache_resource
def load_everything():
    real_model, is_fair_model, model_source = load_model()
    real_scaler = load_scaler()

    real_data = None
    
    real_data = pd.read_excel(DATA_PATH_XLSX, sheet_name="E Comm")



    model = real_model
    if real_data is not None:
        df = real_data.copy()
        if "Tenure" in df.columns:
            df["HourSpendOnApp"] = df["HourSpendOnApp"].fillna(df["HourSpendOnApp"].mean())
            for col in ["Tenure", "WarehouseToHome", "OrderAmountHikeFromlastYear",
                        "CouponUsed", "OrderCount", "DaySinceLastOrder"]:
                if col in df.columns:
                    df[col] = df[col].fillna(df[col].median())
        if "PreferredPaymentMode" in df.columns:
            df["PreferredPaymentMode"] = df["PreferredPaymentMode"].replace(
                {"CC": "Credit Card", "COD": "Cash on Delivery"}
            )
        if "CustomerID" not in df.columns:
            df.insert(0, "CustomerID", np.arange(1000, 1000 + len(df)))

    if real_scaler is not None:
        scaler = real_scaler


    df = engineer_features(df) if "OrderFrequency" not in df.columns else df
    model_feature_names = getattr(model, "feature_names_in_", None)

    ref_rows = [prepare_model_input(row, scaler, model_feature_names) for _, row in df.iterrows()]
    reference_df = pd.concat(ref_rows, ignore_index=True) if ref_rows else pd.DataFrame()

    return df, model, scaler, reference_df, model_feature_names, is_fair_model, model_source


# def _build_demo_data_and_model(n=800, seed=42):
#     """Synthetic fallback -- schema-matched to the real dataset's exact category values."""
#     rng = np.random.default_rng(seed)
#     cv = CATEGORY_VALUES

#     df = pd.DataFrame(
#         {
#             "CustomerID": np.arange(1000, 1000 + n),
#             "Tenure": rng.integers(0, 40, n),
#             "PreferredLoginDevice": rng.choice(cv["PreferredLoginDevice"], n),
#             "CityTier": rng.integers(1, 4, n),
#             "WarehouseToHome": rng.integers(5, 40, n),
#             "PreferredPaymentMode": rng.choice(cv["PreferredPaymentMode"], n),
#             "Gender": rng.choice(cv["Gender"], n),
#             "HourSpendOnApp": rng.integers(1, 6, n),
#             "NumberOfDeviceRegistered": rng.integers(1, 6, n),
#             "PreferedOrderCat": rng.choice(cv["PreferedOrderCat"], n),
#             "SatisfactionScore": rng.integers(1, 6, n),
#             "MaritalStatus": rng.choice(cv["MaritalStatus"], n),
#             "NumberOfAddress": rng.integers(1, 10, n),
#             "Complain": rng.integers(0, 2, n),
#             "OrderAmountHikeFromlastYear": rng.integers(10, 30, n),
#             "CouponUsed": rng.integers(0, 10, n),
#             "OrderCount": rng.integers(0, 15, n),
#             "DaySinceLastOrder": rng.integers(0, 30, n),
#             "CashbackAmount": rng.integers(0, 300, n),
#         }
#     )
#     df = engineer_features(df)

#     risk_score = (
#         (df["Complain"] * 2)
#         + (df["DaySinceLastOrder"] / 10)
#         - (df["OrderFrequency"] * 3)
#         - (df["SatisfactionScore"] * 0.5)
#         + rng.normal(0, 1.5, n)
#     )
#     df["Churn"] = (risk_score > risk_score.median()).astype(int)

#     train_df = df.drop(columns=["CustomerID"])
#     X = encode_categoricals(train_df.drop(columns=["Churn"]))
#     y = train_df["Churn"]

#     scaler = StandardScaler()
#     X[NUMERIC_COLUMNS] = scaler.fit_transform(X[NUMERIC_COLUMNS])

#     model = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=seed)
#     model.fit(X, y)
#     model.feature_names_in_ = np.array(X.columns)

#     return df, model, scaler


(df, model, scaler, reference_df, model_feature_names,
 is_fair_model, model_source) = load_everything()


@st.cache_data(show_spinner=False)
def score_all_customers(_model, _scaler, _df, _model_feature_names):
    probs = []
    for _, row in _df.iterrows():
        model_input = prepare_model_input(row, _scaler, _model_feature_names)
        p = float(_model.predict_proba(model_input)[0, 1])
        probs.append(p)
    out = _df.copy()
    out["ChurnProbability"] = probs
    out["RiskBucket"] = out["ChurnProbability"].apply(risk_bucket)
    return out


scored_df = score_all_customers(model, scaler, df, model_feature_names)


# ----------------------------------------------------------------------------
# SIDEBAR NAV
# ----------------------------------------------------------------------------
st.sidebar.title("🛒 E-Commerce Churn AI")
page = st.sidebar.radio("Navigate", ["Dashboard", "Customer Analysis", "AI Retention Strategy", "About"])

if not is_fair_model:
    st.sidebar.warning(
        "**Baseline model active**\n\nThis model still includes `Gender` and "
        "`MaritalStatus`. Your own audit (Section 8) found it misses far more "
        "married churners than single ones. Drop `randomforest_fair.pkl` into "
        "`models/` to switch to the audited version automatically."
    )



# ----------------------------------------------------------------------------
# PAGE: DASHBOARD
# ----------------------------------------------------------------------------
if page == "Dashboard":
    st.title("Customer Churn Dashboard")

    total = len(scored_df)
    high_risk = (scored_df["RiskBucket"] == "HIGH").sum()
    churn_rate = scored_df["RiskBucket"].eq("HIGH").mean()

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("High Risk Customers", f"{high_risk:,}")
    c3.metric("High-Risk Rate", f"{churn_rate:.1%}")

    st.subheader("Churn Risk Distribution")
    dist = scored_df["RiskBucket"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
    st.bar_chart(dist)

    st.subheader("Customers Requiring Attention")
    attention = (
        scored_df.sort_values("ChurnProbability", ascending=False)
        .head(10)[["CustomerID", "RiskBucket", "ChurnProbability"]]
        .rename(columns={"ChurnProbability": "Probability"})
    )
    attention["Probability"] = attention["Probability"].map(lambda p: f"{p:.0%}")
    st.dataframe(attention, use_container_width=True, hide_index=True)


# ----------------------------------------------------------------------------
# PAGE: CUSTOMER ANALYSIS
# ----------------------------------------------------------------------------
elif page == "Customer Analysis":
    st.title("Customer Analysis")

    customer_id = st.selectbox("Select Customer", scored_df["CustomerID"].tolist())
    analyze = st.button("Analyze Customer", type="primary")

    if analyze or st.session_state.selected_customer == customer_id:
        st.session_state.selected_customer = customer_id
        customer_row = df[df["CustomerID"] == customer_id].iloc[0]

        st.subheader("Customer Profile")
        p1, p2, p3, p4 = st.columns(4)
        p1.metric("Tenure", f"{customer_row['Tenure']} months")
        p2.metric("Order Count", int(customer_row["OrderCount"]))
        p3.metric("Days Since Last Order", int(customer_row["DaySinceLastOrder"]))
        p4.metric("Satisfaction Score", int(customer_row["SatisfactionScore"]))

        with st.expander("Show all model features"):
            st.dataframe(customer_row.to_frame().T, use_container_width=True, hide_index=True)

        model_input = prepare_model_input(customer_row, scaler, model_feature_names)
        probability, is_churn, label = predict_churn(model_input, model, DEPLOYMENT_THRESHOLD)
        bucket = risk_bucket(probability)

        st.subheader("AI Churn Assessment")
        m1, m2, m3 = st.columns(3)
        m1.metric("Churn Risk", f"{probability:.0%}")
        m2.metric("Prediction", label)
        m3.metric("Decision Threshold", f"{DEPLOYMENT_THRESHOLD:.0%}")

        risk_color = {"HIGH": "🔴", "MEDIUM": "🟠", "LOW": "🟢"}[bucket]
        st.markdown(f"### {risk_color} {bucket} RISK")
        st.progress(min(probability, 1.0))

        st.subheader("Why Is This Customer At Risk?")
        drivers = explain_prediction(model, model_input, reference_df, top_n=4)
        if not drivers:
            st.info("No strong individual drivers identified for this customer.")
        else:
            for d in drivers:
                icon = "🔴" if bucket == "HIGH" else "🟡"
                actionable_tag = " *(actionable)*" if d["actionable"] else " *(not directly actionable)*"
                st.markdown(f"**{icon} {d['friendly_name']}** -- raises risk by {d['contribution']:.0%}{actionable_tag}")

        st.session_state.customer_profile = {
            k: v for k, v in customer_row.to_dict().items() if k != "CustomerID"
        }
        st.session_state.probability = probability
        st.session_state.bucket = bucket
        st.session_state.drivers = drivers
        st.success("Analysis complete -- head to **AI Retention Strategy** to generate recommended actions.")


# ----------------------------------------------------------------------------
# PAGE: AI RETENTION STRATEGY
# ----------------------------------------------------------------------------
elif page == "AI Retention Strategy":
    st.title("AI Retention Strategy")

    if "probability" not in st.session_state:
        st.info("Analyze a customer on the **Customer Analysis** page first.")
    else:
        bucket = st.session_state.bucket
        probability = st.session_state.probability
        st.markdown(f"The customer has been classified as **{bucket} RISK** ({probability:.0%} churn probability).")

        if st.button("Generate AI Strategy", type="primary"):
            with st.spinner("Generating retention strategy..."):
                strategy = generate_retention_strategy(
                    customer_profile=st.session_state.customer_profile,
                    probability=probability,
                    risk_label=bucket,
                    drivers=st.session_state.drivers,
                )
            st.markdown("### AI-Generated Retention Strategy")
            st.markdown(strategy)


# ----------------------------------------------------------------------------
# PAGE: ABOUT
# ----------------------------------------------------------------------------
else:
    st.title("About / How It Works")

    fairness_note = (
        "This deployment uses the **fairness-audited model** (`randomforest_fair.pkl`), "
        "trained without `Gender` or `MaritalStatus`, per Section 8.3 of the project notebook."
        if is_fair_model else
        "⚠️ This deployment is currently using the **baseline model**, which still includes "
        "`Gender` and `MaritalStatus`. Section 8 of the project notebook found this version "
        "misses roughly 48% of churning married customers versus 17% of single customers "
        "-- a real fairness gap. The audit also found removing these two fields improved "
        "accuracy (Macro-F1 0.875 → 0.882) at the same time, so this isn't a trade-off "
        "against performance. Save `randomforest_fair.pkl` per the README to switch over."
    )

    st.markdown(
        f"""
This system supports (not replaces) retention decisions for an e-commerce business.

**Pipeline**

```
Customer Data
   -> Preprocessing + Feature Engineering
   -> Random Forest  (Is this customer likely to churn?)
   -> Local Explanation  (Why?)
   -> LLM  (What should the business do about it?)
   -> Retention Strategy
```

**Supervised Learning** — A Random Forest classifier (chosen after comparing
against KNN, Logistic Regression, and SVM -- ROC-AUC 0.969, Macro-F1 0.875)
estimates churn probability from 17 numeric and 5 one-hot-encoded behavioral
features. Predictions use a tuned decision threshold of **{DEPLOYMENT_THRESHOLD:.0%}**
rather than the default 50%, since a missed churner costs more than a false alarm.

**Explainability** — Each prediction is paired with a local explanation: for
every feature, the customer's value is swapped for the typical (median)
value and the resulting drop in predicted probability is measured. The
largest drops are the features actually driving that customer's score.

**LLM** — Llama 3.3 70B (via Groq) turns the churn analysis into a practical,
business-readable retention plan. It never makes or overrides the churn
prediction itself -- that decision belongs to the Random Forest alone.

**Human in the Loop** — The system informs and supports staff decisions; it
does not automatically take action against any customer.

**Fairness note** — {fairness_note}

**Privacy note** — The project's own k-anonymity check found 38% of customers
are uniquely identifiable from just 7 ordinary attributes, well below the
standard k ≥ 5 threshold. This is a known limitation of the underlying
dataset, documented in Section 8.4 of the notebook.
        """
    )

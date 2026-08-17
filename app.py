

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



# load the data

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




#  Dashboard
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



# CUSTOMER ANALYSIS

elif page == "Customer Analysis":
    st.title("Customer Analysis")

    def run_analysis(customer_row: pd.Series):
        """Shared prediction + explanation flow, used by both the dataset
        picker and the manual-entry form below."""
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

    tab_existing, tab_manual = st.tabs(["Select Existing Customer", "Enter Customer Manually"])

    with tab_existing:
        customer_id = st.selectbox("Select Customer", scored_df["CustomerID"].tolist())
        analyze = st.button("Analyze Customer", type="primary")

        if analyze or st.session_state.selected_customer == customer_id:
            st.session_state.selected_customer = customer_id
            customer_row = df[df["CustomerID"] == customer_id].iloc[0]
            run_analysis(customer_row)

    with tab_manual:
        st.caption("Enter a customer's details manually to score them without adding them to the dataset.")

        with st.form("manual_customer_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                tenure = st.number_input("Tenure (months)", min_value=0, max_value=120, value=6)
                city_tier = st.selectbox("City Tier", [1, 2, 3], index=0)
                warehouse_to_home = st.number_input("Warehouse to Home Distance", min_value=0, max_value=200, value=15)
                hour_spend_on_app = st.number_input("Hours Spent on App", min_value=0.0, max_value=24.0, value=3.0, step=0.5)
                number_of_devices = st.number_input("Number of Devices Registered", min_value=1, max_value=10, value=3)
                satisfaction_score = st.slider("Satisfaction Score", 1, 5, 3)
            with c2:
                number_of_address = st.number_input("Number of Addresses", min_value=1, max_value=20, value=2)
                complain = st.selectbox("Filed a Complaint?", ["No", "Yes"], index=0)
                order_amount_hike = st.number_input("Order Amount Hike From Last Year (%)", min_value=0, max_value=100, value=15)
                coupon_used = st.number_input("Coupons Used", min_value=0, max_value=50, value=1)
                order_count = st.number_input("Order Count", min_value=0, max_value=100, value=3)
                day_since_last_order = st.number_input("Days Since Last Order", min_value=0, max_value=365, value=10)
            with c3:
                cashback_amount = st.number_input("Cashback Amount", min_value=0.0, max_value=2000.0, value=120.0, step=10.0)
                preferred_login_device = st.selectbox("Preferred Login Device", CATEGORY_VALUES["PreferredLoginDevice"])
                preferred_payment_mode = st.selectbox("Preferred Payment Mode", CATEGORY_VALUES["PreferredPaymentMode"])
                gender = st.selectbox("Gender", CATEGORY_VALUES["Gender"])
                prefered_order_cat = st.selectbox("Preferred Order Category", CATEGORY_VALUES["PreferedOrderCat"])
                marital_status = st.selectbox("Marital Status", CATEGORY_VALUES["MaritalStatus"])

            manual_submit = st.form_submit_button("Check Churn Risk", type="primary")

        if manual_submit:
            manual_row = pd.Series(
                {
                    "CustomerID": "Manual Entry",
                    "Tenure": tenure,
                    "PreferredLoginDevice": preferred_login_device,
                    "CityTier": city_tier,
                    "WarehouseToHome": warehouse_to_home,
                    "PreferredPaymentMode": preferred_payment_mode,
                    "Gender": gender,
                    "HourSpendOnApp": hour_spend_on_app,
                    "NumberOfDeviceRegistered": number_of_devices,
                    "PreferedOrderCat": prefered_order_cat,
                    "SatisfactionScore": satisfaction_score,
                    "MaritalStatus": marital_status,
                    "NumberOfAddress": number_of_address,
                    "Complain": 1 if complain == "Yes" else 0,
                    "OrderAmountHikeFromlastYear": order_amount_hike,
                    "CouponUsed": coupon_used,
                    "OrderCount": order_count,
                    "DaySinceLastOrder": day_since_last_order,
                    "CashbackAmount": cashback_amount,
                }
            )
            st.session_state.selected_customer = None  # so switching tabs doesn't re-trigger the dataset picker
            run_analysis(manual_row)



# AI RETENTION STRATEGY

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



# ABOUT

else:
    st.title("About / How It Works")

    fairness_note = (
        "This deployment uses the **fairness-audited model** (`randomforest_fair.pkl`), "
        "trained without `Gender` or `MaritalStatus`, per Section 8.3 of the project notebook."
        if is_fair_model else
        "This deployment is currently using the **baseline model**, which still includes "
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
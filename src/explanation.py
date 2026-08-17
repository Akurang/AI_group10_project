"""
explanation.py

Local explanation for a single customer's prediction -- this is your own
`explain_prediction` from Section 8.5 of final_project.ipynb (cell 108),
reused as-is: for each feature, swap in the "typical" (median) value and
measure how much the churn probability drops. A large drop means that
feature was driving the prediction. This is the exact method your notebook
used, so results here match what you'd get re-running that cell.
"""

import pandas as pd

# Features the business can actually act on -- matches your notebook's
# own note in Section 8.5: "Tenure does not [suggest an action] -- telling
# a customer 'you are at risk because you are new' is not useful."
ACTIONABLE_FEATURES = {
    "Complain",
    "DaySinceLastOrder",
    "OrderFrequency",
    "CouponUsageRate",
    "CashbackAmount",
    "SatisfactionScore",
    "HourSpendOnApp",
    "OrderCount",
    "CouponUsed",
    "EngagementScore",
    "RecencyRatio",
    "OrderAmountHikeFromlastYear",
}

FRIENDLY_NAMES = {
    "Complain": "Complaint history",
    "DaySinceLastOrder": "Days Since Last Order",
    "OrderFrequency": "Order Frequency",
    "CouponUsageRate": "Coupon Usage Rate",
    "CashbackAmount": "Cashback Amount",
    "SatisfactionScore": "Satisfaction Score",
    "HourSpendOnApp": "Time Spent on App",
    "OrderCount": "Order Count",
    "CouponUsed": "Coupons Used",
    "EngagementScore": "Engagement Score",
    "RecencyRatio": "Recency Ratio",
    "Tenure": "Tenure",
    "WarehouseToHome": "Warehouse-to-Home Distance",
    "NumberOfDeviceRegistered": "Devices Registered",
    "NumberOfAddress": "Number of Addresses",
    "OrderAmountHikeFromlastYear": "Order Amount Hike YoY",
    "CityTier": "City Tier",
}


def explain_prediction(model, model_input_df: pd.DataFrame, reference_df: pd.DataFrame, top_n: int = 4):
    """
    Exact method from final_project.ipynb cell 108: for each feature, replace
    the customer's value with the "typical" value (median across the
    reference dataset) and measure how far the predicted probability drops.

    model_input_df: the single preprocessed/scaled row fed to the model
    reference_df: the same preprocessed dataset for ALL customers, used to
                  compute the typical/median value for each feature
    Returns a list of dicts: [{feature, friendly_name, contribution, actionable}, ...]
    """
    customer = model_input_df.iloc[0]
    typical_values = reference_df.reindex(columns=model_input_df.columns, fill_value=0).median()

    base_probability = model.predict_proba(customer.to_frame().T)[0, 1]

    contributions = {}
    for feature in customer.index:
        modified = customer.copy()
        modified[feature] = typical_values[feature]
        modified_probability = model.predict_proba(modified.to_frame().T)[0, 1]
        contributions[feature] = base_probability - modified_probability

    ranked = pd.Series(contributions).sort_values(ascending=False)

    results = []
    for feature, contribution in ranked.items():
        if contribution <= 0:
            continue
        if feature not in FRIENDLY_NAMES:
            continue  # skip one-hot dummy noise, keep explanations readable
        results.append(
            {
                "feature": feature,
                "friendly_name": FRIENDLY_NAMES.get(feature, feature),
                "contribution": float(contribution),
                "actionable": feature in ACTIONABLE_FEATURES,
            }
        )
        if len(results) >= top_n:
            break

    return results

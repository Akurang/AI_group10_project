

import pandas as pd

# Features the business can actually act on
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
            continue  
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

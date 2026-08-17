

import pandas as pd


CATEGORICAL_COLUMNS = [
    "PreferredLoginDevice",
    "PreferredPaymentMode",
    "Gender",
    "PreferedOrderCat",
    "MaritalStatus",
]


CATEGORY_VALUES = {
    "PreferredLoginDevice": ["Computer", "Mobile Phone", "Phone"],
    "PreferredPaymentMode": ["Cash on Delivery", "Credit Card", "Debit Card", "E wallet", "UPI"],
    "Gender": ["Female", "Male"],
    "PreferedOrderCat": ["Fashion", "Grocery", "Laptop & Accessory", "Mobile", "Mobile Phone", "Others"],
    "MaritalStatus": ["Divorced", "Married", "Single"],
}


NUMERIC_COLUMNS = [
    "Tenure",
    "CityTier",
    "WarehouseToHome",
    "HourSpendOnApp",
    "NumberOfDeviceRegistered",
    "SatisfactionScore",
    "NumberOfAddress",
    "Complain",
    "OrderAmountHikeFromlastYear",
    "CouponUsed",
    "OrderCount",
    "DaySinceLastOrder",
    "CashbackAmount",
    "OrderFrequency",
    "CouponUsageRate",
    "RecencyRatio",
    "EngagementScore",
]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Exact formulas from Section 4.2 of final_project.ipynb."""
    df = df.copy()
    df["OrderFrequency"] = df["OrderCount"] / (df["Tenure"] + 1)
    df["CouponUsageRate"] = df["CouponUsed"] / (df["OrderCount"] + 1)
    df["RecencyRatio"] = df["DaySinceLastOrder"] / (df["Tenure"] + 1)
    df["EngagementScore"] = df["HourSpendOnApp"] * df["NumberOfDeviceRegistered"]
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Exact call from Section 3.6: drop_first=True, dtype=int."""
    present = [c for c in CATEGORICAL_COLUMNS if c in df.columns]
    return pd.get_dummies(df, columns=present, drop_first=True, dtype=int)


def prepare_model_input(customer_row: pd.Series, scaler, model_feature_names) -> pd.DataFrame:

    row = customer_row.to_frame().T
    row = engineer_features(row)
    row = encode_categoricals(row)

    numeric_present = [c for c in NUMERIC_COLUMNS if c in row.columns]
    for col in numeric_present:
        row[col] = pd.to_numeric(row[col], errors="coerce").fillna(0)

    if scaler is not None and numeric_present:
        row[numeric_present] = scaler.transform(row[numeric_present])

    if model_feature_names is not None:
        row = row.reindex(columns=model_feature_names, fill_value=0)

    return row

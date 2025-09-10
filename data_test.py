import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

def test_dataset_loads():
    """Dataset should not be empty and must contain expected columns."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    assert not df.empty
    expected_cols = {"Customer_ID", "Purchase_Amount", "Age", "Time_to_Decision", "Customer_Satisfaction"}
    assert expected_cols.issubset(df.columns)

def test_purchase_amount_cleaning():
    """Purchase_Amount should be numeric after cleaning."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df["Purchase_Amount"] = (
        df["Purchase_Amount"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    assert pd.api.types.is_float_dtype(df["Purchase_Amount"])
    # No '$' should remain
    assert df["Purchase_Amount"].astype(str).str.contains(r"\$").sum() == 0

def test_groupby_summary():
    """Grouped summary should return correct columns and non-empty result."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df["Purchase_Amount"] = (
        df["Purchase_Amount"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    grouped = df.groupby(["Gender", "Income_Level", "Education_Level"]).agg(
        Time_to_Decision_mean=("Time_to_Decision", "mean"),
        Avg_Purchase_Amount=("Purchase_Amount", "mean"),
        Count=("Customer_ID", "count")
    ).reset_index()
    assert not grouped.empty
    assert {"Time_to_Decision_mean", "Avg_Purchase_Amount", "Count"}.issubset(grouped.columns)

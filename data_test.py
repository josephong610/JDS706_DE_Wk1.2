import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pytest

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

def test_model_training_and_prediction():
    """XGBRegressor should train and return predictions with correct shape."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df["Purchase_Amount"] = (
        df["Purchase_Amount"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    X = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    y = df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert not np.isnan(y_pred).any()

def test_model_performance():
    """Model should achieve finite, non-negative MSE and R² within [-1, 1]."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df["Purchase_Amount"] = (
        df["Purchase_Amount"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )

    X = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    y = df["Purchase_Amount"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    assert mse >= 0
    assert -1 <= r2 <= 1


def test_empty_dataset_behavior():
    """Empty dataframe should not break cleaning logic."""
    df = pd.DataFrame(columns=["Customer_ID", "Purchase_Amount", "Age", "Time_to_Decision", "Customer_Satisfaction"])
    
    # Cleaning should still work without error
    df["Purchase_Amount"] = df["Purchase_Amount"].astype(str)
    assert df.empty

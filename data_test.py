import pandas as pd
import numpy as np
import pytest
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Import helpers from main script
from data import clean_purchase_amount, summarize_predictions


def test_dataset_loads():
    """Dataset should not be empty and must contain expected columns."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    assert not df.empty
    expected_cols = {
        "Customer_ID",
        "Purchase_Amount",
        "Age",
        "Time_to_Decision",
        "Customer_Satisfaction",
    }
    assert expected_cols.issubset(df.columns)


def test_purchase_amount_cleaning():
    """Purchase_Amount should be numeric after cleaning."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    assert pd.api.types.is_float_dtype(df["Purchase_Amount"])
    # No '$' should remain
    assert df["Purchase_Amount"].astype(str).str.contains(r"\$").sum() == 0


def test_groupby_summary():
    """Grouped summary should return correct columns and non-empty result."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    grouped = (
        df.groupby(["Gender", "Income_Level", "Education_Level"])
        .agg(
            Time_to_Decision_mean=("Time_to_Decision", "mean"),
            Avg_Purchase_Amount=("Purchase_Amount", "mean"),
            Count=("Customer_ID", "count"),
        )
        .reset_index()
    )

    assert not grouped.empty
    assert {"Time_to_Decision_mean", "Avg_Purchase_Amount", "Count"}.issubset(
        grouped.columns
    )


def test_model_training_and_prediction():
    """XGBRegressor should train and return predictions with correct shape."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    features = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    target = df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    assert len(y_pred) == len(y_test)
    assert not np.isnan(y_pred).any()


def test_model_performance():
    """Model should achieve finite, non-negative MSE and R² within [-1, 1]."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    features = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    target = df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    assert mse >= 0
    assert -1 <= r2 <= 1


def test_summarize_predictions_runs():
    """summarize_predictions should return non-empty results with Actual/Predicted columns."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    features = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    target = df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    test_df = X_test.copy()
    test_df["Actual"] = y_test
    test_df["Predicted"] = y_pred
    test_df["Age_Group"] = pd.cut(test_df["Age"], bins=[0, 25, 35, 50, 65, 100])

    grouped = summarize_predictions(test_df, "Age_Group")

    assert not grouped.empty
    assert {"Actual", "Predicted"}.issubset(grouped.columns)


def test_empty_dataset_behavior():
    """Empty dataframe should not break cleaning logic."""
    df = pd.DataFrame(
        columns=[
            "Customer_ID",
            "Purchase_Amount",
            "Age",
            "Time_to_Decision",
            "Customer_Satisfaction",
        ]
    )
    df = clean_purchase_amount(df)

    assert df.empty
    assert "Purchase_Amount" in df.columns


# ===================== New Tests =====================


def test_missing_values_filled():
    """Categorical columns should have no NaNs after filling."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    df["Engagement_with_Ads"] = df["Engagement_with_Ads"].fillna("No engagement")
    df["Social_Media_Influence"] = df["Social_Media_Influence"].fillna("No engagement")

    assert df["Engagement_with_Ads"].isnull().sum() == 0
    assert df["Social_Media_Influence"].isnull().sum() == 0


def test_outlier_clipping():
    """Numeric columns should be clipped at the 1st and 99th percentiles."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    numeric_cols = [
        "Purchase_Amount",
        "Age",
        "Time_to_Decision",
        "Customer_Satisfaction",
        "Product_Rating",
        "Time_Spent_on_Product_Research(hours)",
        "Frequency_of_Purchase",
    ]

    for col in numeric_cols:
        lower, upper = df[col].quantile([0.01, 0.99])
        clipped = df[col].clip(lower, upper)
        assert clipped.min() >= lower
        assert clipped.max() <= upper


def test_error_distribution_computation():
    """Prediction errors should be finite values."""
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    df = clean_purchase_amount(df)

    features = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    target = df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    model = XGBRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    errors = y_test - y_pred
    assert np.isfinite(errors).all()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor


# ===================== Utility Functions =====================

def clean_purchase_amount(df: pd.DataFrame) -> pd.DataFrame:
    """Cleans the Purchase_Amount column by removing $ and commas, converting to float."""
    df["Purchase_Amount"] = (
        df["Purchase_Amount"].astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    return df


def summarize_predictions(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarize Actual vs Predicted values by a given column."""
    return (
        df.groupby(group_col)[["Actual", "Predicted"]]
        .mean()
        .reset_index()
    )


def plot_actual_vs_predicted(y_true, y_pred):
    """Scatter plot comparing actual vs predicted purchase amounts."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.plot([y_true.min(), y_true.max()],
             [y_true.min(), y_true.max()],
             "r--", lw=2, label="Perfect Prediction")

    plt.title("Actual vs Predicted Purchase Amounts")
    plt.xlabel("Actual Purchase Amount")
    plt.ylabel("Predicted Purchase Amount")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png", dpi=300)
    plt.show()


def plot_feature_importance(model, feature_names):
    """Horizontal bar plot of feature importances from the trained model."""
    importances = model.feature_importances_
    plt.figure(figsize=(8, 5))
    plt.barh(feature_names, importances, color="skyblue", edgecolor="k")
    plt.xlabel("Importance Score")
    plt.title("Feature Importance in Predicting Purchase Amount")
    plt.tight_layout()
    plt.savefig("feature_importance_scores.png", dpi=300)
    plt.show()


# ===================== Main Script =====================

def main():
    # ===== Inspecting the Data =====
    consumer_df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    print(consumer_df.head())
    print(consumer_df.info())
    print(consumer_df.describe())

    # Missing values + duplicates
    print("=== Missing values per column ===")
    print(consumer_df.isnull().sum())

    num_duplicates = consumer_df.duplicated().sum()
    print(f"\n=== Duplicate Rows ===\n{num_duplicates} duplicate rows")
    consumer_df = consumer_df.drop_duplicates()

    # ===== Cleaning =====
    consumer_df = clean_purchase_amount(consumer_df)

    # ===== Grouped Summary =====
    grouped = consumer_df.groupby(["Gender", "Income_Level", "Education_Level"]).agg(
        Time_to_Decision_mean=("Time_to_Decision", "mean"),
        Time_to_Decision_median=("Time_to_Decision", "median"),
        Avg_Purchase_Amount=("Purchase_Amount", "mean"),
        Engagement_with_Ads_counts=("Engagement_with_Ads", lambda x: x.value_counts().to_dict()),
        Count=("Customer_ID", "count")
    ).reset_index()
    print("\n=== Grouped Summary ===")
    print(grouped.head(20))

    # ===== Machine Learning with XGBoost =====
    features = consumer_df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
    target = consumer_df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)

    # ===== Group-Level Analysis =====
    X_test_with_preds = X_test.copy()
    X_test_with_preds["Actual"] = y_test
    X_test_with_preds["Predicted"] = y_pred

    grouped_by_age = summarize_predictions(X_test_with_preds, "Age").sort_values("Age")
    print("\n=== Average Purchase Amounts by Exact Age ===")
    print(grouped_by_age.head(20))

    X_test_with_preds["Age_Group"] = pd.cut(
        X_test_with_preds["Age"],
        bins=[0, 25, 35, 50, 65, 100],
        labels=["<25", "25-35", "35-50", "50-65", "65+"]
    )
    grouped_by_age_group = summarize_predictions(X_test_with_preds, "Age_Group")
    print("\n=== Average Purchase Amounts by Age Group ===")
    print(grouped_by_age_group)

    grouped_by_satisfaction = summarize_predictions(X_test_with_preds, "Customer_Satisfaction")\
        .sort_values("Customer_Satisfaction")
    print("\n=== Average Purchase Amounts by Customer Satisfaction ===")
    print(grouped_by_satisfaction)

    X_test_with_preds["Decision_Time_Group"] = pd.cut(
        X_test_with_preds["Time_to_Decision"],
        bins=[0, 2, 5, 10, 20, 50],
        labels=["0-2", "3-5", "6-10", "11-20", "20+"]
    )
    grouped_by_decision_time = summarize_predictions(X_test_with_preds, "Decision_Time_Group")
    print("\n=== Average Purchase Amounts by Time to Decision Group ===")
    print(grouped_by_decision_time)

    # ===== Visualizations =====
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model, features.columns)


if __name__ == "__main__":
    main()

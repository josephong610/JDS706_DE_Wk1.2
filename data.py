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
        df["Purchase_Amount"]
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .astype(float)
    )
    return df


def summarize_predictions(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Summarize Actual vs Predicted values by a given column."""
    return df.groupby(group_col)[["Actual", "Predicted"]].mean().reset_index()


def plot_actual_vs_predicted(y_true, y_pred):
    """Scatter plot comparing actual vs predicted purchase amounts."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, edgecolor="k")
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
        label="Perfect Prediction",
    )

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


def plot_error_distribution(y_true, y_pred):
    """Histogram of prediction errors (Actual - Predicted)."""
    errors = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.hist(errors, bins=30, color="steelblue", edgecolor="k", alpha=0.7)
    plt.axvline(0, color="red", linestyle="--", lw=2, label="Perfect Prediction")
    plt.title("Distribution of Prediction Errors")
    plt.xlabel("Error (Actual - Predicted)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig("error_distribution.png", dpi=300)
    plt.show()


def plot_avg_actual_vs_predicted(df, group_col):
    """Bar chart comparing average Actual vs Predicted grouped by a column."""
    grouped = df.groupby(group_col)[["Actual", "Predicted"]].mean().reset_index()
    grouped.plot(x=group_col, kind="bar", figsize=(8, 6))
    plt.title(f"Average Actual vs Predicted by {group_col}")
    plt.ylabel("Purchase Amount")
    plt.tight_layout()
    plt.savefig(f"avg_actual_vs_predicted_{group_col}.png", dpi=300)
    plt.show()


def plot_residuals(y_true, y_pred):
    """Residual plot of prediction errors vs predicted values."""
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, edgecolor="k")
    plt.axhline(0, color="red", linestyle="--", lw=2)
    plt.title("Residual Plot (Errors vs Predicted Values)")
    plt.xlabel("Predicted Purchase Amount")
    plt.ylabel("Residual (Actual - Predicted)")
    plt.tight_layout()
    plt.savefig("residual_plot.png", dpi=300)
    plt.show()


# ===================== Main Script =====================


def main():
    # ===== Inspecting the Data =====
    consumer_df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    print(consumer_df.head())
    print(consumer_df.info())
    print(consumer_df.describe())

    # Missing values + duplicates
    print("=== Missing values per column BEFORE cleaning ===")
    print(consumer_df.isnull().sum())

    num_duplicates = consumer_df.duplicated().sum()
    print(f"\n=== Duplicate Rows ===\n{num_duplicates} duplicate rows")
    consumer_df = consumer_df.drop_duplicates()

    # ===== Cleaning =====
    print("\n=== Data Cleaning Steps ===")
    consumer_df = clean_purchase_amount(consumer_df)
    print("1. Removed $ and commas from Purchase_Amount and converted to float.")

    # Fill missing categorical values with "No engagement"
    consumer_df["Engagement_with_Ads"] = consumer_df["Engagement_with_Ads"].fillna(
        "No engagement"
    )
    consumer_df["Social_Media_Influence"] = consumer_df[
        "Social_Media_Influence"
    ].fillna("No engagement")
    print(
        "2. Filled NA in Engagement_with_Ads and Social_Media_Influence with 'No engagement'."
    )

    # Handle outliers (winsorize numeric columns at 1st/99th percentiles)
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
        lower, upper = consumer_df[col].quantile([0.01, 0.99])
        consumer_df[col] = consumer_df[col].clip(lower, upper)
    print("3. Winsorized numeric features at 1st and 99th percentiles.")

    print("\n=== Missing values per column AFTER cleaning ===")
    print(consumer_df.isnull().sum())

    # ===== Grouped Summary =====
    grouped = (
        consumer_df.groupby(["Gender", "Income_Level", "Education_Level"])
        .agg(
            Time_to_Decision_mean=("Time_to_Decision", "mean"),
            Time_to_Decision_median=("Time_to_Decision", "median"),
            Avg_Purchase_Amount=("Purchase_Amount", "mean"),
            Engagement_with_Ads_counts=(
                "Engagement_with_Ads",
                lambda x: x.value_counts().to_dict(),
            ),
            Count=("Customer_ID", "count"),
        )
        .reset_index()
    )
    print("\n=== Grouped Summary ===")
    print(grouped.head(20))

    # ===== Machine Learning with XGBoost =====
    features = consumer_df[
        [
            "Age",
            "Time_to_Decision",
            "Customer_Satisfaction",
            "Brand_Loyalty",
            "Product_Rating",
            "Time_Spent_on_Product_Research(hours)",
            "Frequency_of_Purchase",
        ]
    ]
    target = consumer_df["Purchase_Amount"]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=200, learning_rate=0.1, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print("\n=== Model Performance ===")
    print("MSE:", mse)
    print("RMSE:", rmse)
    print("R²:", r2)

    # ===== Group-Level Analysis =====
    X_test_with_preds = X_test.copy()
    X_test_with_preds["Actual"] = y_test
    X_test_with_preds["Predicted"] = y_pred

    grouped_by_loyalty = summarize_predictions(
        X_test_with_preds, "Brand_Loyalty"
    ).sort_values("Brand_Loyalty")
    print("\n=== Average Purchase Amounts by Brand Loyalty ===")
    print(grouped_by_loyalty)

    grouped_by_rating = summarize_predictions(
        X_test_with_preds, "Product_Rating"
    ).sort_values("Product_Rating")
    print("\n=== Average Purchase Amounts by Product Rating ===")
    print(grouped_by_rating)

    grouped_by_frequency = summarize_predictions(
        X_test_with_preds, "Frequency_of_Purchase"
    ).sort_values("Frequency_of_Purchase")
    print("\n=== Average Purchase Amounts by Purchase Frequency ===")
    print(grouped_by_frequency)

    grouped_by_research = summarize_predictions(
        X_test_with_preds, "Time_Spent_on_Product_Research(hours)"
    ).sort_values("Time_Spent_on_Product_Research(hours)")
    print("\n=== Average Purchase Amounts by Time Spent on Product Research ===")
    print(grouped_by_research)

    # ===== Visualizations =====
    plot_actual_vs_predicted(y_test, y_pred)
    plot_feature_importance(model, features.columns)
    plot_error_distribution(y_test, y_pred)

    X_test_with_preds["Age_Group"] = pd.cut(
        X_test_with_preds["Age"],
        bins=[0, 25, 35, 50, 65],
        labels=["<25", "25-35", "35-50", "50-65"],
    )
    plot_avg_actual_vs_predicted(X_test_with_preds, "Age_Group")

    plot_residuals(y_test, y_pred)


if __name__ == "__main__":
    main()

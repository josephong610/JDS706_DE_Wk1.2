import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Made with help of ChatGPT

# ===== Inspecting the Data =====
df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
# Observing the top of the dataset
print(df.head())

# Inspecting the data
print(df.info())
print(df.describe())

# Check for missing values per column
print("=== Missing values per column ===")
print(df.isnull().sum())

# Check if there are any duplicated rows and then drops duplicate rows
num_duplicates = df.duplicated().sum()
print(f"\n=== Duplicate Rows ===\n{num_duplicates} duplicate rows")
df = df.drop_duplicates()

# ===== Basic Filtering and Grouping =====

# Clean Purchase_Amount column: removing any $ and commas, convert to float
df["Purchase_Amount"] = (
    df["Purchase_Amount"]
    .astype(str)                              # ensure it's string
    .str.replace(r"[\$,]", "", regex=True)    # remove $ and commas
    .astype(float)                            # convert to float
)

# Finding things like mean, median, and counts for certain columns
grouped = df.groupby(["Gender", "Income_Level", "Education_Level"]).agg(
    Time_to_Decision_mean=("Time_to_Decision", "mean"),
    Time_to_Decision_median=("Time_to_Decision", "median"),
    Avg_Purchase_Amount=("Purchase_Amount", "mean"),
    Engagement_with_Ads_counts=("Engagement_with_Ads", lambda x: x.value_counts().to_dict()),
    Count=("Customer_ID", "count")
).reset_index()

print("\n=== Grouped Summary ===")
print(grouped.head(20))

# ===== Exploring Machine Learning Algorithms with XGBoost =====
X = df[["Age", "Time_to_Decision", "Customer_Satisfaction"]]
y = df["Purchase_Amount"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Training the model
model = XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Calculate MSE and RMSE
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("R²:", r2)

# ===== Group-Level Analysis =====
# Add predictions + actuals to the test set
X_test_with_preds = X_test.copy()
X_test_with_preds["Actual"] = y_test
X_test_with_preds["Predicted"] = y_pred

# Group by exact Age
grouped_by_age = (
    X_test_with_preds.groupby("Age")[["Actual", "Predicted"]]
    .mean()
    .reset_index()
    .sort_values("Age")
)

print("\n=== Average Purchase Amounts by Exact Age ===")
print(grouped_by_age.head(20))

# Group by Age ranges
X_test_with_preds["Age_Group"] = pd.cut(
    X_test_with_preds["Age"],
    bins=[0, 25, 35, 50, 65, 100],
    labels=["<25", "25-35", "35-50", "50-65", "65+"]
)

grouped_by_age_group = (
    X_test_with_preds.groupby("Age_Group")[["Actual", "Predicted"]]
    .mean()
    .reset_index()
)

print("\n=== Average Purchase Amounts by Age Group ===")
print(grouped_by_age_group)

# Group by Customer Satisfaction levels
grouped_by_satisfaction = (
    X_test_with_preds.groupby("Customer_Satisfaction")[["Actual", "Predicted"]]
    .mean()
    .reset_index()
    .sort_values("Customer_Satisfaction")
)

print("\n=== Average Purchase Amounts by Customer Satisfaction ===")
print(grouped_by_satisfaction)

# Group by Time to Decision ranges
X_test_with_preds["Decision_Time_Group"] = pd.cut(
    X_test_with_preds["Time_to_Decision"],
    bins=[0, 2, 5, 10, 20, 50],
    labels=["0-2", "3-5", "6-10", "11-20", "20+"]
)

grouped_by_decision_time = (
    X_test_with_preds.groupby("Decision_Time_Group")[["Actual", "Predicted"]]
    .mean()
    .reset_index()
)

print("\n=== Average Purchase Amounts by Time to Decision Group ===")
print(grouped_by_decision_time)


# ===== Visualization =====
# Scatter plot: Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolor="k")
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         "r--", lw=2, label="Perfect Prediction")

plt.title("Actual vs Predicted Purchase Amounts")
plt.xlabel("Actual Purchase Amount")
plt.ylabel("Predicted Purchase Amount")
plt.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=300)
plt.show()

# Feature importance from the trained model
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,5))
plt.barh(features, importances, color="skyblue", edgecolor="k")
plt.xlabel("Importance Score")
plt.title("Feature Importance in Predicting Purchase Amount")
plt.tight_layout()
plt.savefig("feature_importance_scores.png", dpi=300)
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

CSV_PATH = "Ecommerce_Consumer_Behavior_Analysis_Data.csv"

# =========================
# PANDAS SECTION
# =========================

# 1) IMPORT
def load_data(path: str = CSV_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "Purchase_Amount" in df.columns:
        df["Purchase_Amount"] = (
            df["Purchase_Amount"].astype(str)
            .str.replace(r"[\$,]", "", regex=True)
            .astype(float)
        )
    return df

# 2) INSPECT
def inspect_data(df: pd.DataFrame) -> dict:
    head = df.head(5).to_dict(orient="records")
    info_buf = []
    df.info(buf=info_buf.append)
    describe = df.describe(include="all", datetime_is_numeric=True).to_dict()
    missing = df.isna().sum().to_dict()
    dups = int(df.duplicated().sum())
    return {
        "head": head,
        "info": "\n".join(info_buf),
        "describe": describe,
        "missing": missing,
        "duplicates": dups,
    }

# 3) BASIC FILTER & GROUP
def filter_and_group(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    filt = df[df.get("Income_Level", pd.Series(index=df.index, dtype=object)) == "High"] if "Income_Level" in df else df.iloc[0:0]
    group = (
        df.groupby("Purchase_Channel", dropna=False)["Purchase_Amount"].mean().reset_index()
        if {"Purchase_Channel", "Purchase_Amount"} <= set(df.columns)
        else pd.DataFrame(columns=["Purchase_Channel", "Purchase_Amount"])
    )
    return filt, group

# 4) SIMPLE ML
def train_model(df: pd.DataFrame) -> tuple[LogisticRegression, str]:
    needed = {"Age", "Purchase_Amount", "Frequency_of_Purchase", "Customer_Satisfaction", "Discount_Used"}
    if not needed <= set(df.columns):
        raise ValueError(f"Missing columns for ML: {sorted(list(needed - set(df.columns)))}")
    X = df[["Age", "Purchase_Amount", "Frequency_of_Purchase", "Customer_Satisfaction"]].dropna()
    y = df.loc[X.index, "Discount_Used"].astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=300)
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    report = classification_report(y_te, y_pred)
    return model, report

# 5) ONE PLOT
def make_plot(df: pd.DataFrame, out_path: str = "purchase_amount_hist.png") -> str:
    if "Purchase_Amount" not in df.columns:
        raise ValueError("Column 'Purchase_Amount' not found.")
    plt.figure()
    df["Purchase_Amount"].dropna().hist(bins=30)
    plt.title("Distribution of Purchase Amounts")
    plt.xlabel("Purchase Amount")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path
[![Python Template for IDS706](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/main.yml/badge.svg)](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/main.yml)

# JDS706_DE_Wk1.2

## Project Goal and Real-World Relevance
The goal of this assignment is to analyze **Ecommerce Consumer Behavior** data and predict purchase amounts using customer features such as **age, gender, income level, education, satisfaction, loyalty, and time to decision**.  

Predicting purchase amounts has direct real-world relevance:
- **E-commerce platforms** can use it to optimize personalized recommendations.  
- **Marketing teams** can better allocate resources to high-value customer segments.  
- **Customer analytics** can identify trends in impulsive vs. planned purchases.  

By modeling consumer behavior, businesses can uncover insights into what drives higher spending and which factors matter most for purchase decisions.

---

## Data Source and Structure
The dataset (`Ecommerce_Consumer_Behavior_Analysis_Data.csv`) contains a collection of consumer behavior features, like demographics, purchase behavior, satisfaction ratings, loyalty indicators, and decision-making metrics. The table below shows an example of what the columns look like. (Although there are more columns in the actual file)

The website link to the Kaggle dataset is here: https://www.kaggle.com/datasets/salahuddinahmedshuvo/ecommerce-consumer-behavior-analysis-data

### Example of Dataset Structure

| Customer_ID | Age | Gender | Income_Level | Education_Level | Purchase_Amount | Purchase_Intent | Time_to_Decision | Customer_Satisfaction |
|-------------|-----|--------|--------------|----------------|-----------------|-----------------|------------------|-----------------------|
| 1001        | 25  | Male   | Low          | High School    | 120.50          | Impulsive       | 2                | 8                     |
| 1002        | 34  | Female | Middle       | Bachelor's     | 340.00          | Planned         | 5                | 9                     |
| 1003        | 42  | Male   | High         | Master's       | 580.75          | Needs-based     | 7                | 7                     |

---

## Data Cleaning and Preprocessing
Before modeling, several steps were taken to ensure **data quality and robustness**:

1. **Cleaning purchase amounts**  
   - Removed `$` symbols and commas.  
   - Converted values to numeric floats for analysis.  

2. **Handling missing values**  
   - Filled `Engagement_with_Ads` and `Social_Media_Influence` missing values with `"No engagement"`.  

3. **Removing duplicates**  
   - All duplicate rows were dropped from the dataset.  

4. **Outlier handling**  
   - Winsorized numeric columns (`Purchase_Amount`, `Age`, `Time_to_Decision`, `Customer_Satisfaction`, `Product_Rating`, `Time_Spent_on_Product_Research(hours)`, `Frequency_of_Purchase`) at the **1st and 99th percentiles** to reduce the influence of extreme values.  

These steps ensured that the machine learning model was trained on **clean, representative, and stable data**.

---

## Grouped Summary Statistics
To better understand the dataset, grouped summaries were computed by demographic and behavioral categories (Gender, Income Level, Education Level):

- **Mean and Median Time to Decision**  
- **Average Purchase Amount**  
- **Counts of Engagement with Ads**  
- **Number of Customers per Group**

This gave us insight into **how spending and decision-making vary across different demographic groups**.

---

## Machine Learning Model
We trained an **XGBoost Regressor** to predict purchase amounts. Features used included:

- `Age`  
- `Time_to_Decision`  
- `Customer_Satisfaction`  
- `Brand_Loyalty`  
- `Product_Rating`  
- `Time_Spent_on_Product_Research(hours)`  
- `Frequency_of_Purchase`  

### Model Metrics
After splitting the data (80% train, 20% test), we evaluated performance:

- **MSE:** `23649.48`  
- **RMSE:** `153.78`  
- **R²:** `-0.36`  

🔎 **Interpretation:**  
- The **RMSE of ~154** means that predictions are, on average, off by about $154 from the true purchase amounts.  
- The **negative R²** indicates that the model currently performs worse than a simple baseline (predicting the mean purchase amount for everyone).  

This highlights the **challenge of predicting spending behavior** — additional feature engineering or alternative models are needed.

---

## Visualizations
Several visualizations were generated to evaluate the model and interpret results:

1. **Scatter Plot of Actual vs. Predicted Values**  
   ![Actual vs Predicted](actual_vs_predicted.png)  
   - Shows where predictions deviate from the 1:1 line.  
   - Clear evidence that predictions often miss actual values by a wide margin.  

2. **Feature Importance Bar Chart**  
   ![Feature Importance](feature_importance_scores.png)  
   - Indicates which features had the most influence.  
   - `Age` and `Time_to_Decision` were most important, followed by satisfaction and loyalty.  

3. **Error Distribution Histogram**  
   ![Error Distribution](error_distribution.png)  
   - Plots the distribution of `(Actual – Predicted)` errors.  
   - Errors center near 0 but are wide, confirming poor model accuracy.  

4. **Residual Plot**  
   ![Residual Plot](residual_plot.png)  
   - Residuals vs. Predicted values.  
   - No clear linear trend, but very high variance across the prediction range.  

5. **Average Actual vs Predicted by Age Group**  
   ![Average Actual vs Predicted by Age Group](avg_actual_vs_predicted_Age_Group.png)  
   - Groups customers into age bands (`<25`, `25-35`, `35-50`, `50-65`).  
   - Shows that the model consistently **underestimates purchase amounts** across most age groups.  
   - The discrepancy is largest in the **25–35 group**, where predicted averages are well below actual averages.  

These plots show that while some signal exists in the data, the model still struggles to generalize.

---

## Environment Setup with Docker
This project is **fully containerized** using Docker to ensure reproducibility.

### Step 1: Clone the repository
```bash
git clone https://github.com/josephong610/JDS706_DE_Wk1.2.git
cd JDS706_DE_Wk1.2
```

### Step 2: Build the Docker image
```bash
docker build -t jds706_project .
```

### Step 3: Run the container
```bash
docker run --rm jds706_project
```

---

## Dockerfile
The `Dockerfile` ensures that dependencies and scripts are reproducible across systems.

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
```

---

## Running Tests
Tests in `data_test.py` validate:
- Dataset loads correctly and required columns exist.  
- Purchase amounts are cleaned and numeric.  
- Grouped summaries produce correct aggregates.  
- XGBoost trains and returns predictions of correct shape.  
- Performance metrics (MSE, R²) are valid.  
- Edge cases (empty dataset) don’t break the pipeline.  

Run tests locally:
```bash
make test
```

Or in Docker:
```bash
make docker-test
```

---

## Project Structure
```
.
├── data.py                 # Main analysis (cleaning, modeling, visualization)
├── data_test.py            # Test suite for validation
├── requirements.txt        # Dependencies
├── Dockerfile              # Containerized environment
├── Makefile                # Local + Docker workflows
├── README.md               # Project documentation
└── Ecommerce_Consumer_Behavior_Analysis_Data.csv
```

---

## Makefile Usage
For convenience, the project includes a Makefile:

- **Local workflow**:
  ```bash
  make all
  ```
  Runs install, format, lint, test, clean.

- **Docker workflow**:
  ```bash
  make docker-all
  ```
  Builds the image, lints, formats, tests, and cleans inside Docker.

---

## Results and Discussion
- **Feature importance** suggests that **age and decision time** are the strongest predictors of spending.  
- **Customer satisfaction** mattered less than expected, which might indicate non-linear relationships.  
- **Model accuracy was poor**, showing the difficulty of predicting purchase amounts with only demographic and decision-related features.  
- Future improvements could include:
  - Adding more features (engagement, loyalty, ad influence).  
  - Trying alternative models (Random Forest, Neural Nets).  
  - Hyperparameter tuning of XGBoost.  

---

## Conclusion
This project demonstrated the full pipeline of:
- Cleaning and preprocessing messy real-world data.  
- Handling missing values and outliers.  
- Training and evaluating a machine learning model.  
- Visualizing results to diagnose weaknesses.  
- Ensuring reproducibility via **Docker + Makefile**.  

While the model currently underperforms, the infrastructure (cleaning, testing, reproducibility) provides a strong foundation for **iterative improvement and experimentation**.

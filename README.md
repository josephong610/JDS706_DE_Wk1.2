[![Python CI](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/python-ci.yml/badge.svg)](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/python-ci.yml)

# JDS706_DE_Wk1.2

## Repository Overview
This repository is designed to explore consumer behavior data using Python. It contains scripts for data inspection, cleaning, grouping, machine learning experiments, and visualization. The repository also includes automated testing and linting setup to ensure reproducibility and maintain code quality.

- `data.py` – main analysis script (data inspection, cleaning, grouping, ML, visualization)  
- `data_test.py` – test cases for validating data processing and model pipeline  
- `requirements.txt` – list of dependencies for reproducibility  
- `Makefile` – automation for installing, formatting, linting, and testing  

## Goal of the Project
The goal of this project is to analyze factors such as **education level, income level, age, and time to decision** to uncover trends in consumer purchasing behavior. The focus is on identifying which groups of customers are more likely to spend on necessities, make impulsive purchases, or plan their purchases ahead of time.

## Data Source and Structure
The dataset (`Ecommerce_Consumer_Behavior_Analysis_Data.csv`) contains a comprehensive collection of consumer behavior features, including demographics, purchase behavior, satisfaction ratings, loyalty indicators, and decision-making metrics. This makes it well-suited for market segmentation, predictive modeling, and understanding customer decision-making.

### Example of Dataset Structure

| Customer_ID | Age | Gender | Income_Level | Education_Level | Purchase_Amount | Purchase_Intent | Time_to_Decision | Customer_Satisfaction |
|-------------|-----|--------|--------------|----------------|-----------------|-----------------|------------------|-----------------------|
| 1001        | 25  | Male   | Low          | High School    | 120.50          | Impulsive       | 2                | 8                     |
| 1002        | 34  | Female | Middle       | Bachelor's     | 340.00          | Planned         | 5                | 9                     |
| 1003        | 42  | Male   | High         | Master's       | 580.75          | Needs-based     | 7                | 7                     |

---

## Setup Process

1. Clone the repository:
   ```bash
   git clone https://github.com/josephong610/JDS706_DE_Wk1.2.git
   cd JDS706_DE_Wk1.2
   ```

2. Install dependencies:
   ```bash
   make install
   ```

3. Format and lint code:
   ```bash
   make format
   make lint
   ```

4. Run tests:
   ```bash
   make test
   ```

---

## Usage

To run the main analysis script:

```bash
python data.py
```

This will:
- Inspect and clean the dataset  
- Perform grouping and summary statistics  
- Train an XGBoost model to predict purchase amount  
- Generate and save visualizations (`actual_vs_predicted.png`, `feature_importance_scores.png`)  

---

## Visualizations
The project generates:
- **Scatter plots** of actual vs. predicted purchase amounts  
- **Feature importance bar charts** showing which features (e.g., age, satisfaction, time to decision) most influence purchase amount  

Plots are saved in the project directory as `.png` files for reporting.

---

## Target Audience
- **Data scientists & analysts** exploring consumer behavior  
- **Marketers** aiming to segment customers and improve targeting  
- **Researchers** studying factors that influence consumer decision-making  

---

## License
This project is for educational purposes as part of **JDS706 Data Engineering** coursework.

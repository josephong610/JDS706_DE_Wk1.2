[![Python Template for IDS706](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/main.yml/badge.svg)](https://github.com/josephong610/JDS706_DE_Wk1.2/actions/workflows/main.yml)

# JDS706_DE_Wk1.2

## Project Goal
The goal of this assignment this week is to make sure that our code is reproducible using Docker. We want to analyze the **Ecommerce Consumer Behavior** data, focusing on how demographics (age, gender, income, education), customer satisfaction, and decision time influence purchase behavior.

So far, I have
- Cleaned and preprocessed the dataset.
- Performed grouping and summary statistics by demographic segments.
- Trained an **XGBoost regression model** to predict purchase amounts.
- Evaluated model performance with MSE, RMSE, and R².
- Produced visualizations of predictions vs actuals and feature importance.

It seems that the visualizations showed that age and time to decision had the highest feature importance score compared to customer satisfaction. Our model still definitely needs work as shown by the predicted purchase amount vs actual purchase amount graph though.
---

## Environment Setup with Docker

This project is fully containerized using Docker to ensure a reproducible environment.  
The Dockerfile installs Python, project dependencies, and runs tests by default.

### Step 1: Clone the repository
```bash
git clone https://github.com/josephong610/JDS706_DE_Wk1.2.git
cd JDS706_DE_Wk1.2
```

### Step 2: Build the Docker image
Run the following command from the repo root:
```bash
docker build -t jds706_project .
```

This command:
1. Uses the official Python 3.12 slim image as a base.
2. Copies the `requirements.txt` file into the container.
3. Installs all dependencies (`pandas`, `scikit-learn`, `xgboost`, etc.).
4. Copies the rest of the project files into the container.
5. Sets the default command to run `pytest` so tests execute automatically.

### Step 3: Run the container
```bash
docker run --rm jds706_project
```
- `--rm` removes the container after it exits, keeping things clean.
- If all tests pass, you should see output similar to:
  ```
  ........                                                                 [100%]
  8 passed in 1.76s
  ```

---

## Dockerfile Contents

Below is the exact `Dockerfile` used in this project:

```dockerfile
# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy dependency file first for caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files
COPY . .

# Default command: run tests
CMD ["pytest", "--maxfail=1", "--disable-warnings", "-q"]
```

---

## Running Tests with Docker

Tests in `data_test.py` validate:
- Dataset loads correctly and required columns exist.
- Purchase amounts are cleaned and numeric.
- Grouped summaries return correct columns.
- XGBoost model trains and predicts without errors.
- Performance metrics (MSE, R²) are finite and within valid ranges.
- Edge cases (empty dataset) are handled gracefully.

By default, tests run automatically when the container starts.  

You can also override the default CMD to run a shell inside the container:
```bash
docker run -it --rm jds706_project bash
```

Once inside:
```bash
pytest -vv
```

---

## Project Structure

The most relevant files for this week's assignment are below:
```
.
├── data.py                 # Main analysis script (data cleaning, modeling, plots)
├── data_test.py            # Unit and system tests
├── requirements.txt        # Dependencies
├── Dockerfile              # Docker setup (reproducible environment + test runner)
├── Makefile                # Local + Docker workflows
├── README.md               # Documentation
└── Ecommerce_Consumer_Behavior_Analysis_Data.csv
```

---

## Using the Makefile with Docker

For convenience, this project includes a `Makefile` with common Docker commands:

- **Build the Docker image**
  ```bash
  make docker-build
  ```

- **Run tests inside Docker**
  ```bash
  make docker-test
  ```

- **Open a shell inside the container (debugging)**
  ```bash
  make docker-shell
  ```

- **Build and test in one step**
  ```bash
  make docker-all
  ```

This makes it easy to reproduce results using make.

---

## Example Test Output

When running the Docker container, all tests pass:

![Results in Docker](Tests_being_passed.png)


## README from the previous week's assignment
I kept this part in just in case I needed to keep in the context for this assignment.
--------------------------------------------------------------------------------------------------------------------------------------------
## Goal of the Assignment
The goal of this assignment is to analyze various factors like **education level, income level, age, and time to decision** to uncover trends in consumer purchasing behavior. The focus is to identify which groups of customers are more likely to spend on necessities, make impulsive purchases, or plan their purchases ahead of time.

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

## Repository Overview
This repository is made for homework 2 and also to explore consumer behavior data. It contains scripts for data inspection, cleaning, grouping, machine learning with XGBoost, and some visualization. It also includes automated testing and linting setup to ensure reproducibility and maintain code quality. The files below are the important ones for this assignment.

- `data.py` – main analysis script (data inspection, cleaning, grouping, ML, visualization)  
- `data_test.py` – test cases for validating data processing
- `requirements.txt` – list of dependencies for reproducibility  
- `Makefile` – automation for installing, formatting, linting, and testing  
- `Ecommerce_Consumer_Behavior_Analysis_Data` - dataset

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
python3 data.py
```

This will:
- Inspect and clean the dataset  
- Perform grouping and summary statistics  
- Train an XGBoost model to predict purchase amount  
- Generate and save visualizations (`actual_vs_predicted.png`, `feature_importance_scores.png`)  

---

## Preliminary Experiment Setup and Findings
I first did the basic steps like inspecting the data and doing some basic grouping to see some summary statistics. You'll find these results when running the data.py file. I also added some test cases in data_test.py just so I know that data.py should run without any troubles.

For the ML part of this assignment, I have set up an XGBoost model that looks through the "Age", "Time_to_Decision", and "Customer_Satisfaction" columns to see the average purchase amount (from the "Purchase_Amount" column) that these groups fall into.

So far from the XGBoost model, I predicted that people aged 30 would have the highest average purchase amount. It turns out that actually people aged 31 have the highest average purchase amount. My predicted numbers aren't as close as I want them to be as you'll see when running data.py, but the age is kind of close.

I also found that the customer satisfaction doesn't really seem to scale linearly with average purchase amount. Even for customer satisfaction scores of 1, which is the lowest it goes, the average purchase amount was still quite high. The actual average purchase amount for a score of 1 was $302 while I predicted it would be $272. 

Finally, I looked at how many days people decided to take beofre buying a product. I found that the decision time didn't really matter for the average purchase amount. It was all around $275, no matter how many days they took to decide.

These results were very interesting, but the MSE is extremely high right now and the R-squared value is not close to 1 at all. Therefore, there will need to be some tweaks to my process because the results do not seem to be very accurate right now. I also want to look at other variables as well in the future.

---

## Visualizations
I generated:
- **Scatter plots** of actual vs. predicted purchase amounts  
- **Feature importance bar charts** showing which features (e.g., age, satisfaction, time to decision) most influence purchase amount  

Plots are saved in the directory as well for reporting.

![Actual vs Predicted](actual_vs_predicted.png)
![Feature Importance](feature_importance_scores.png)

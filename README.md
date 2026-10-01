# Freight Rate Prediction - Machine Learning Assessment

This repository contains my solution for the Machine Learning Engineer freight-rate prediction assessment.

The objective is to train a regression model using historical freight-load data and predict the posted rate for 12,000 unseen validation loads.

## Project Overview

The labeled development dataset contains 48,000 freight loads with information including:

- Pickup and delivery locations
- Geographic coordinates
- Distance
- Equipment type
- Weight
- Date
- Market index
- Quote signal
- Posted rate

The target variable is:

`posted_rate`

The final validation dataset contains 12,000 unseen loads requiring predictions.

## Validation Strategy

Because the final evaluation data occurs chronologically after the development data, I used a time-based validation strategy rather than a random train/test split.

Development split:

- Training: January 1, 2025 through August 31, 2025
- Validation: September 1, 2025 through October 31, 2025

This produced approximately:

- 80% training data
- 20% validation data

Hyperparameter selection was performed using expanding monthly validation folds within the training period.

For example:

- January-May -> June
- January-June -> July
- January-July -> August

This prevents future observations from leaking into model training.

## Data Quality

Several data-quality issues were identified during exploratory analysis.

### Negative weights

292 negative weight values were detected.

The absolute values of these observations closely matched the distribution of valid positive weights, indicating that they were likely sign errors.

The negative values were therefore converted to their absolute values.

### Missing values

Missing values were found in:

- `weight`: 300 rows
- `market_index`: 374 rows

The missing values were sparse and appeared randomly distributed.

Median imputation was therefore performed inside the machine-learning pipelines so that imputation statistics were learned only from the training data.

## Feature Engineering

The following date features were created:

- Month
- Day of month
- Day of week
- Week of year
- Weekend indicator
- Days since January 1, 2025

A directional route feature was also created:

`pickup -> delivery`

For example:

`Lexington -> Fort Wayne`

Categorical variables were encoded using One-Hot Encoding with unknown-category handling.

## Models

Two primary regression approaches were evaluated:

### Ridge Regression

Ridge Regression was used as a regularized linear baseline.

The regularization strength was selected using chronological cross-validation.

Final parameter:

`alpha = 100`

### XGBoost

XGBoost was evaluated to capture nonlinear relationships and feature interactions.

Selected configuration:

- n_estimators: 300
- max_depth: 4
- learning_rate: 0.05
- subsample: 0.9
- colsample_bytree: 0.9

## Final Main Model

The final model uses an ensemble of:

- 75% Ridge Regression
- 25% XGBoost

On the September-October holdout set, the ensemble achieved:

| Metric | Result |
|---|---:|
| MAE | $130.43 |
| RMSE | $637.72 |
| R² | 0.8254 |

These metrics refer to the internal chronological validation set. The official validation metrics are calculated by Spotter after submission.

## December Prediction Model

The supplied December chart dataset does not contain all of the features available in the main validation dataset.

A separate compatible model was therefore trained using only features available in the December input:

- Pickup
- Delivery
- Route
- Distance
- Equipment
- Weight
- Date-derived features

The final December model is a 50/50 ensemble of Ridge Regression and XGBoost.

Its September-October holdout performance was:

| Metric | Result |
|---|---:|
| MAE | $147.54 |
| RMSE | $646.66 |
| R² | 0.8204 |

## Repository Structure

```text
freight-rate-ml-assessment/
│
├── README.md
├── requirements.txt
├── .gitignore
├── freight_rate_ml_assessment.ipynb
├── score.py
├── validation_predictions.csv
├── december_predictions.csv
│
├── data/
│   └── README.md
│
├── report/
│   └── Freight_Rate_ML_Assessment_Report.pdf
│
└── scorer_results/
    └── candidate_december.png

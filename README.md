# Supply Chain Order Classification
*Classification Model built for STAT 303-2 at Northwestern University - March 2025*

## Overview
This project focuses on predicting whether an order will be delivered on time and complete using historical supply chain data. By leveraging features related to orders, vendors, product classifications, and transit times, the goal is to develop a robust classification model that performs well on unseen data while minimizing overfitting.

## Methodology

### Data Preprocessing
- Cleaned and merged training and test datasets  
- Imputed missing values using median imputation  
- Created time-based features, such as week of the year and days between order and due date bins  
- Checked distributions, skewness, and correlations for potential feature transformations  

### Feature Engineering
- Log-transformed highly skewed numeric variables to stabilize variance  
- Created categorical bins for key date and duration features  
- One-hot encoded categorical variables for modeling  

### Model Development
- Primary model: logistic regression with L2 regularization  
- Tried a variety of techniques to evaluate accuracy and prevent overfitting, including:  
  - Multiple LASSO models  
  - Elastic Net regularization  
  - Randomized search for hyperparameter tuning  
  - Cross-validation to ensure generalizability  
- Optimized the classification threshold using ROC curve analysis  
- Evaluated performance using training accuracy, cross-validation accuracy, and Kaggle leaderboard results  

## Key Findings
- Optimal logistic regression model achieved ~79% training accuracy  
- Cross-validation accuracy confirmed model generalizability (~77%)  

## Data Sources
- `train_X.csv` and `train_y.csv` – historical order data with features and target labels  
- `public_private_X.csv` – test data used for Kaggle leaderboard submission  

## Files
- `order_prediction_model.ipynb` – Jupyter notebook containing full preprocessing, feature engineering, model development, and evaluation  
- `predictions_ridge_3.csv` – final predictions for submission to Kaggle leaderboard  

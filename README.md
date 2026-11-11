
# House Price Prediction using Ridge and Lasso Regression

## Project Overview

This project predicts house prices using Machine Learning regression algorithms. It compares Ordinary Least Squares (Linear Regression), Ridge Regression, and Lasso Regression to identify the model with the lowest validation RMSE.

## Objectives

- Load the Boston Housing dataset using Scikit-learn.
- Split the dataset into training and validation sets.
- Train Linear Regression, Ridge Regression, and Lasso Regression models.
- Compare models using Root Mean Squared Error (RMSE).
- Test different regularization strengths (Alpha values).
- Select and save the best-performing model.

## Technologies Used

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Joblib

## Machine Learning Algorithms

### 1. Linear Regression
Used as the baseline regression model to predict house prices.

### 2. Ridge Regression
Uses L2 regularization to reduce the impact of large model coefficients.

### 3. Lasso Regression
Uses L1 regularization and can reduce less useful feature coefficients towards zero.

## Dataset

The Boston Housing dataset is loaded through Scikit-learn's `fetch_openml()` function.

The dataset contains housing-related features and a target value representing house prices.

## Data Preprocessing

- Missing values are handled using SimpleImputer.
- Features are standardized using StandardScaler.
- Data is divided into training and validation sets.

## Model Evaluation

The models are evaluated using Root Mean Squared Error (RMSE).

A lower RMSE indicates better prediction performance on the validation set.

The following Alpha values are tested:

`0.001, 0.01, 0.1, 1, 10, 100`

## Project Files

| File | Description |
|---|---|
| train.py | Main machine learning program |
| requirements.txt | Required Python libraries |
| best_house_price_model.pkl | Saved best model |
| validation_rmse_comparison.png | RMSE comparison graph |

## Installation and Execution

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

### 2. Open the project folder

```bash
cd ridge-lasso-regression
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the project

```bash
python train.py
```

## Output

The project generates:

- Validation RMSE results for all trained models.
- Ridge and Lasso comparison graph.
- Best-performing trained model saved as a `.pkl` file.
- Sample house price predictions.

## Conclusion

This project demonstrates the application of Linear Regression, Ridge Regression, and Lasso Regression for house price prediction. It compares different regularization strengths using validation RMSE and saves the best-performing model for future predictions.

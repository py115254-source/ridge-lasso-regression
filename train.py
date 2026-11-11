
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# Project Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2

ALPHAS = [0.001, 0.01, 0.1, 1, 10, 100]

MODEL_FILE = "best_house_price_model.pkl"
PLOT_FILE = "validation_rmse_comparison.png"


def load_data():
    """Load the Boston Housing dataset from OpenML."""

    print("Loading Boston Housing dataset...")

    data = fetch_openml(
        name="boston",
        version=1,
        as_frame=True,
        parser="liac-arff"
    )

    X = data.data.apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(data.target, errors="coerce")

    print("Dataset loaded successfully!")
    print("Dataset shape:", X.shape)

    return X, y


def split_data(X, y):
    """Split data into training and validation sets."""

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    print("\nTraining samples:", len(X_train))
    print("Validation samples:", len(X_val))

    return X_train, X_val, y_train, y_val


def create_pipeline(model):
    """Create preprocessing and regression pipeline."""

    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("regressor", model)
    ])


def calculate_rmse(y_true, y_pred):
    """Calculate Root Mean Squared Error."""

    return np.sqrt(mean_squared_error(y_true, y_pred))


def train_models(X_train, y_train, X_val, y_val):
    """Train Linear, Ridge and Lasso regression models."""

    results = []
    trained_models = {}

    # Linear Regression
    print("\nTraining Linear Regression...")

    linear_model = create_pipeline(LinearRegression())
    linear_model.fit(X_train, y_train)

    predictions = linear_model.predict(X_val)
    rmse = calculate_rmse(y_val, predictions)

    results.append({
        "Model": "Linear Regression",
        "Alpha": 0,
        "Validation RMSE": rmse
    })

    trained_models["Linear Regression"] = linear_model

    # Ridge Regression
    print("\nTraining Ridge Regression...")

    for alpha in ALPHAS:

        ridge_model = create_pipeline(Ridge(alpha=alpha))
        ridge_model.fit(X_train, y_train)

        predictions = ridge_model.predict(X_val)
        rmse = calculate_rmse(y_val, predictions)

        results.append({
            "Model": "Ridge",
            "Alpha": alpha,
            "Validation RMSE": rmse
        })

        trained_models[f"Ridge_{alpha}"] = ridge_model

    # Lasso Regression
    print("\nTraining Lasso Regression...")

    for alpha in ALPHAS:

        lasso_model = create_pipeline(
            Lasso(alpha=alpha, max_iter=100000)
        )

        lasso_model.fit(X_train, y_train)

        predictions = lasso_model.predict(X_val)
        rmse = calculate_rmse(y_val, predictions)

        results.append({
            "Model": "Lasso",
            "Alpha": alpha,
            "Validation RMSE": rmse
        })

        trained_models[f"Lasso_{alpha}"] = lasso_model

    return pd.DataFrame(results), trained_models


def plot_results(results):
    """Plot validation RMSE for Ridge and Lasso models."""

    plt.figure(figsize=(10, 6))

    for model_name in ["Ridge", "Lasso"]:

        model_results = results[
            results["Model"] == model_name
        ]

        plt.plot(
            model_results["Alpha"],
            model_results["Validation RMSE"],
            marker="o",
            label=model_name
        )

    plt.xscale("log")
    plt.xlabel("Regularization Strength (Alpha)")
    plt.ylabel("Validation RMSE")
    plt.title("Ridge and Lasso Validation RMSE Comparison")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(PLOT_FILE, dpi=300)

    print("\nRMSE graph saved:", PLOT_FILE)

    plt.show()


def select_best_model(results, trained_models):
    """Select and save the model with the lowest validation RMSE."""

    best_result = results.loc[
        results["Validation RMSE"].idxmin()
    ]

    model_name = best_result["Model"]
    alpha = best_result["Alpha"]

    if model_name == "Linear Regression":
        model_key = "Linear Regression"
    else:
        model_key = f"{model_name}_{alpha}"

    best_model = trained_models[model_key]

    joblib.dump(best_model, MODEL_FILE)

    print("\nBest Model Selected")
    print("-------------------")
    print("Model:", model_name)
    print("Alpha:", alpha)
    print(
        "Validation RMSE:",
        round(best_result["Validation RMSE"], 4)
    )

    print("\nBest model saved:", MODEL_FILE)

    return best_model


def main():
    """Execute the complete house price prediction workflow."""

    X, y = load_data()

    X_train, X_val, y_train, y_val = split_data(X, y)

    results, trained_models = train_models(
        X_train,
        y_train,
        X_val,
        y_val
    )

    print("\nAll Model Results")
    print("-----------------")
    print(results.to_string(index=False))

    plot_results(results)

    select_best_model(results, trained_models)

    # Verify saved model
    loaded_model = joblib.load(MODEL_FILE)

    print("\nSaved model loaded successfully!")

    sample_predictions = loaded_model.predict(
        X_val.iloc[:5]
    )

    print("\nSample House Price Predictions:")
    print(sample_predictions)


if __name__ == "__main__":
    main()

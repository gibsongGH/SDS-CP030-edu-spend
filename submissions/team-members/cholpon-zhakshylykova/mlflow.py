# Full MLflow script with plot logging and automatic UI launch


import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import subprocess
import time
import webbrowser
import sys
import sklearn
import xgboost

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.preprocessing import LabelEncoder

# Load preprocessed data
data = pd.read_csv("preprocessed.csv")  # Change to your path

# Drop targets for feature matrix
X = data.drop(columns=["Total_cost", "Affordability_Tier"])
y_reg = data["Total_cost"]
y_cls = data["Affordability_Tier"]

# Encode class labels
le = LabelEncoder()
y_cls_encoded = le.fit_transform(y_cls)

# Split data
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_cls, y_test_cls = train_test_split(X, y_cls_encoded, test_size=0.2, random_state=42)

# Set MLflow experiment
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Preprocessed_Education_Cost_Pipeline")

# Define models and hyperparameter grids
regressors = {
    "RandomForest": (RandomForestRegressor(), {"n_estimators": [50, 100], "max_depth": [5, 10]}),
    "GradientBoosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}),
    "XGBoost": (XGBRegressor(objective="reg:squarederror", verbosity=0), {"n_estimators": [50, 100], "max_depth": [3, 5]})
}

# Store predicted costs
predicted_costs = {}

for name, (model, params) in regressors.items():
    with mlflow.start_run(run_name=f"{name}_Regressor"):
        # Log versions
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)

        grid = GridSearchCV(model, params, cv=3, scoring="neg_mean_squared_error")
        grid.fit(X_train, y_train_reg)
        best_model = grid.best_estimator_
        mlflow.sklearn.log_model(
            sk_model=best_model,
            name="model",
            input_example=X_test[:1],
            signature=None)
        
        preds = best_model.predict(X_test)

        # Save predicted costs
        predicted_costs[name] = preds

        # Regression metrics
        mae = mean_absolute_error(y_test_reg, preds)
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Plot: Predicted vs Actual
        plt.figure(figsize=(6, 4))
        plt.scatter(y_test_reg, preds, alpha=0.6)
        plt.xlabel("Actual Total Cost")
        plt.ylabel("Predicted Total Cost")
        plt.title(f"{name} - Predicted vs Actual")

        # Add best-fit line
        z = np.polyfit(y_test_reg, preds, 1)
        p = np.poly1d(z)
        plt.plot(y_test_reg, p(y_test_reg), color="red", linewidth=2, label="Best fit line")
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("pred_vs_actual.png")
        mlflow.log_artifact("pred_vs_actual.png")
        plt.close()

# CLASSIFICATION PHASE
for name, preds in predicted_costs.items():
    with mlflow.start_run(run_name=f"{name}_Classifier"):
        # Log versions
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)
        pred_tiers = pd.qcut(preds, q=3, labels=["Low", "Medium", "High"])
        y_pred_cls = le.transform(pred_tiers)

        # Metrics
        acc = accuracy_score(y_test_cls, y_pred_cls)
        prec = precision_score(y_test_cls, y_pred_cls, average="macro")
        rec = recall_score(y_test_cls, y_pred_cls, average="macro")
        f1 = f1_score(y_test_cls, y_pred_cls, average="macro")
        logloss = log_loss(y_test_cls, pd.get_dummies(y_pred_cls))

        mlflow.log_metric("Accuracy", acc)
        mlflow.log_metric("Precision", prec)
        mlflow.log_metric("Recall", rec)
        mlflow.log_metric("F1_score", f1)
        mlflow.log_metric("LogLoss", logloss)

        try:
            auc = roc_auc_score(y_test_cls, pd.get_dummies(y_pred_cls), multi_class="ovo", average="macro")
            mlflow.log_metric("AUC_ROC", auc)
        except:
            pass

        # Confusion matrix plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test_cls, y_pred_cls, ax=ax)
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        plt.savefig("conf_matrix.png")
        mlflow.log_artifact("conf_matrix.png")
        plt.close()


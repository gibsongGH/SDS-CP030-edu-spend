import shutil
import os
if os.path.exists("mlruns"):
    shutil.rmtree("mlruns")

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import sys
import sklearn
import xgboost
import joblib
from datetime import datetime
import os
import glob

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

# --- Clean old plots ---
for f in glob.glob("*.png"):
    os.remove(f)

# --- Load data ---
data = pd.read_csv("data_tca_clusters_raw.csv")

# --- Feature config ---
categorical_features = ["Country", "City", "University", "Program", "Level"]
numeric_features = [
    "Tuition_USD", "Living_Cost_Index", "Rent_USD",
    "Visa_Fee_USD", "Insurance_USD", "Duration_Years"
]

target_column = "Total_cost"
X = data[categorical_features + numeric_features]
y_reg = data[target_column]
y_cls = pd.qcut(y_reg, q=3, labels=["Low", "Medium", "High"])

# --- MLflow setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Education_Cost_Pipeline")

# --- Preprocessing pipeline ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features)
])

# --- Regressors and param grids ---
regressors = {
    "RandomForest": (
        RandomForestRegressor(),
        {"regressor__n_estimators": [50, 100], "regressor__max_depth": [5, 10]}
    ),
    "GradientBoosting": (
        GradientBoostingRegressor(),
        {"regressor__n_estimators": [50, 100], "regressor__learning_rate": [0.05, 0.1]}
    ),
    "XGBoost": (
        XGBRegressor(objective="reg:squarederror", verbosity=0),
        {"regressor__n_estimators": [50, 100], "regressor__max_depth": [3, 5]}
    )
}

# --- Train/test split (for final evaluation only) ---
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

best_overall_model = None
best_overall_r2 = -np.inf
best_overall_name = ""
predicted_costs = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# --- Loop over regressors ---
for name, (regressor, param_grid) in regressors.items():
    with mlflow.start_run(run_name=f"{name}_Regressor_{timestamp}"):
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)

        # Build pipeline
        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        # Cross-validation + grid search
        grid = GridSearchCV(pipe, param_grid, cv=5, scoring="r2")
        grid.fit(X_train, y_train_reg)
        best_model = grid.best_estimator_

        mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:1])

        # Predict on test set
        preds = best_model.predict(X_test)
        predicted_costs[name] = preds

        # Metrics
        mae = mean_absolute_error(y_test_reg, preds)
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        # Plot if best
        if r2 > best_overall_r2:
            best_overall_r2 = r2
            best_overall_model = best_model
            best_overall_name = name

            plt.figure(figsize=(6, 4))
            plt.scatter(y_test_reg, preds, alpha=0.6)
            plt.xlabel("Actual Total Cost")
            plt.ylabel("Predicted Total Cost")
            plt.title(f"Best Model: {name} - Predicted vs Actual")

            z = np.polyfit(y_test_reg, preds, 1)
            p = np.poly1d(z)
            plt.plot(y_test_reg, p(y_test_reg), color="red", linewidth=2, label="Best fit line")
            plt.legend()

            best_plot_filename = f"best_model_plot_{name}.png"
            plt.savefig(best_plot_filename)
            mlflow.log_artifact(best_plot_filename)
            plt.close()

mlflow.end_run()

# --- Save best pipeline ---
joblib.dump(best_overall_model, "model_pipeline.pkl")

print(f"📦 Saved best model ({best_overall_name}) with R² = {best_overall_r2:.3f} to model_pipeline.pkl")

# --- Classification from predicted TCA ---
final_preds = predicted_costs[best_overall_name]
pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])

with mlflow.start_run(run_name=f"{best_overall_name}_Classifier_{timestamp}"):
    acc = accuracy_score(y_test_cls, pred_tiers)
    prec = precision_score(y_test_cls, pred_tiers, average="macro")
    rec = recall_score(y_test_cls, pred_tiers, average="macro")
    f1 = f1_score(y_test_cls, pred_tiers, average="macro")
    logloss = log_loss(y_test_cls, pd.get_dummies(pred_tiers))

    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", rec)
    mlflow.log_metric("F1_score", f1)
    mlflow.log_metric("LogLoss", logloss)

    try:
        auc = roc_auc_score(y_test_cls, pd.get_dummies(pred_tiers), multi_class="ovo", average="macro")
        mlflow.log_metric("AUC_ROC", auc)
    except:
        pass

    fig, ax = plt.subplots(figsize=(6, 4))
    ConfusionMatrixDisplay.from_predictions(y_test_cls, pred_tiers, ax=ax)
    plt.title(f"{best_overall_name} - Confusion Matrix")
    file_name = f"confusion_matrix_{best_overall_name}.png"
    plt.savefig(file_name)
    mlflow.log_artifact(file_name)
    plt.close()

# --- Save final results ---
result_df = pd.DataFrame({
    "Actual_TCA": y_test_reg.values,
    "Predicted_TCA": final_preds,
    "Actual_Affordability": y_test_cls.values,
    "Predicted_Affordability": pred_tiers
})
result_df.to_csv("model_predictions.csv", index=False)
print("📄 Saved predictions to model_predictions.csv")
print("\n📊 Classification Report:")
print(classification_report(y_test_cls, pred_tiers))

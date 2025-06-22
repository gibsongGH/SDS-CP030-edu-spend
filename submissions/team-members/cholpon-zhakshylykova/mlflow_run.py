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

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# --- Clean plots ---
for f in glob.glob("*.png"):
    os.remove(f)

# --- Load data ---
data = pd.read_csv("data_full_filtered.csv")

# --- Feature configuration ---
categorical_features = ["Country", "City", "University", "Program", "Level"]
numeric_features = [
    "Tuition_USD", "Living_Cost_Index", "Rent_USD",
    "Visa_Fee_USD", "Insurance_USD", "Duration_Years"
]

# --- Targets ---
target_column = "Total_cost"
y_reg = data[target_column]
y_cls = pd.qcut(data[target_column], q=3, labels=["Low", "Medium", "High"])

# --- Fit encoder on categorical features ---
encoder = OneHotEncoder(sparse_output=False, drop="first", handle_unknown="ignore")
encoded_features = encoder.fit_transform(data[categorical_features])
encoded_columns = encoder.get_feature_names_out(categorical_features)

# --- Fit scaler on numeric features ---
scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[numeric_features])


# ✅ Debug check
print("Scaler trained on:", scaler.feature_names_in_)

# --- Combine encoded + scaled into feature matrix ---
# --- Combine encoded + scaled into feature matrix ---
X_full = pd.DataFrame(
    np.hstack([encoded_features, scaled_features]),
    columns=list(encoded_columns) + numeric_features
)

# Ensure target column is not accidentally included
if "Total_cost" in X_full.columns:
    X_full = X_full.drop(columns=["Total_cost"])

X = X_full.copy()

# --- Train/test split ---
X_train, X_test, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)
_, _, y_train_cls, y_test_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# --- MLflow setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Education_Cost_Pipeline")

# --- Define models ---
regressors = {
    "RandomForest": (RandomForestRegressor(), {"n_estimators": [50, 100], "max_depth": [5, 10]}),
    "GradientBoosting": (GradientBoostingRegressor(), {"n_estimators": [50, 100], "learning_rate": [0.05, 0.1]}),
    "XGBoost": (XGBRegressor(objective="reg:squarederror", verbosity=0), {"n_estimators": [50, 100], "max_depth": [3, 5]})
}

# --- Model selection ---
predicted_costs = {}
best_overall_model = None
best_overall_r2 = -np.inf
best_overall_name = ""
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

for name, (model, params) in regressors.items():
    with mlflow.start_run(run_name=f"{name}_Regressor_{timestamp}"):
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)

        grid = GridSearchCV(model, params, cv=3, scoring="neg_mean_squared_error")
        grid.fit(X_train, y_train_reg)
        best_model = grid.best_estimator_

        mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:1])

        preds = best_model.predict(X_test)
        predicted_costs[name] = preds

        mae = mean_absolute_error(y_test_reg, preds)
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

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

# --- Save the best model and preprocessors ---
joblib.dump({
    "regressor": best_overall_model,
    "encoder": encoder,
    "scaler": scaler
}, "model_pipeline.pkl")

print(f"Final training feature columns:", list(X_full.columns))
print(f"\U0001F4E6 Saved best model ({best_overall_name}) with R² = {best_overall_r2:.3f} to model_pipeline.pkl")

# --- Classification Phase (optional logging) ---
for name, preds in predicted_costs.items():
    with mlflow.start_run(run_name=f"{name}_Classifier_{timestamp}"):
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)

        pred_tiers = pd.qcut(preds, q=3, labels=["Low", "Medium", "High"])
        y_pred_cls = pred_tiers

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

        fig, ax = plt.subplots(figsize=(6, 4))
        ConfusionMatrixDisplay.from_predictions(y_test_cls, y_pred_cls, ax=ax)
        plt.title(f"{name} - Confusion Matrix")
        plt.tight_layout()
        file_name = f"confusion_matrix_{name}.png"
        plt.savefig(file_name)
        mlflow.log_artifact(file_name)
        plt.close()

# --- Save evaluation results ---
final_preds = predicted_costs[best_overall_name]
pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])
true_tiers = y_test_cls.values

result_df = pd.DataFrame({
    "Actual_TCA": y_test_reg.values,
    "Predicted_TCA": final_preds,
    "Actual_Affordability": true_tiers,
    "Predicted_Affordability": pred_tiers
})

result_df.to_csv("model_predictions.csv", index=False)
print("📄 Saved actual vs predicted results to model_predictions.csv")

print("\n📊 Classification Report (Predicted vs Actual Affordability):")
print(classification_report(true_tiers, pred_tiers))

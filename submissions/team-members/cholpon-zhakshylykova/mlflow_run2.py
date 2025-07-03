import shutil
import os
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
import glob

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

print("=" * 50)
print("CATEGORICAL-ONLY MODEL PIPELINE")
print("=" * 50)
print(f"XGBoost version: {xgboost.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print("=" * 50)

# --- Clean old plots ---
for f in glob.glob("*.png"):
    os.remove(f)

# --- Load data ---
data = pd.read_csv("data_tca_clusters_raw.csv")

# --- Feature config - ONLY CATEGORICAL FEATURES ---
categorical_features = ["Country", "City", "University", "Program", "Level"]
# Remove numeric features entirely
# numeric_features = ["Tuition_USD", "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Duration_Years"]

target_column = "Total_cost"
X = data[categorical_features]  # Only categorical features
y_reg = data[target_column]
y_cls = pd.qcut(y_reg, q=3, labels=["Low", "Medium", "High"])

print(f"Using only categorical features: {categorical_features}")
print(f"Feature matrix shape: {X.shape}")

# --- MLflow setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Education_Cost_Categorical_Only")

# --- Preprocessing - Only OneHotEncoder needed ---
preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
])

# --- Regressors (tree-based models typically work better with categorical data) ---
regressors = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", verbosity=0, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42)
}

# --- Param grids - Enhanced for categorical-only models ---
param_grids = {
    "RandomForest": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [10, 15, None],
        "regressor__min_samples_split": [2, 5]
    },
    "GradientBoosting": {
        "regressor__n_estimators": [100, 200],
        "regressor__learning_rate": [0.05, 0.1, 0.2],
        "regressor__max_depth": [5, 10]
    },
    "XGBoost": {
        "regressor__n_estimators": [100, 200],
        "regressor__max_depth": [5, 10],
        "regressor__learning_rate": [0.05, 0.1, 0.2]
    },
    "LinearRegression": {},
    "Ridge": {"regressor__alpha": [0.1, 1.0, 10.0]},
    "Lasso": {"regressor__alpha": [0.1, 1.0, 10.0]}
}

# --- Train/test split ---
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)

best_overall_model = None
best_overall_r2 = -np.inf
best_overall_name = ""
predicted_costs = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

performance_log = []

print("\nTraining categorical-only models...")
print("=" * 50)

# --- Loop over all models (no scaling needed for categorical-only) ---
for name, regressor in regressors.items():
    run_name = f"{name}_CategoricalOnly_{timestamp}"
    print(f"Training {name}...")
    
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("python_version", sys.version)
        mlflow.log_param("sklearn_version", sklearn.__version__)
        mlflow.log_param("xgboost_version", xgboost.__version__)
        mlflow.log_param("feature_type", "categorical_only")
        mlflow.log_param("features", categorical_features)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", regressor)
        ])

        grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="r2", n_jobs=-1)
        grid.fit(X_train, y_train_reg)
        best_model = grid.best_estimator_

        mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:1])

        preds = best_model.predict(X_test)
        predicted_costs[run_name] = preds

        mae = mean_absolute_error(y_test_reg, preds)
        mse = mean_squared_error(y_test_reg, preds)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test_reg, preds)

        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("MSE", mse)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)

        performance_log.append({
            "Model": name,
            "Feature_Type": "categorical_only",
            "MAE": mae,
            "MSE": mse,
            "RMSE": rmse,
            "R2": r2,
            "Best_Params": grid.best_params_
        })

        print(f"  {name}: R² = {r2:.3f}, RMSE = {rmse:.2f}")

        if r2 > best_overall_r2:
            best_overall_r2 = r2
            best_overall_model = best_model
            best_overall_name = run_name

            plt.figure(figsize=(8, 6))
            plt.scatter(y_test_reg, preds, alpha=0.6, color='blue')
            plt.xlabel("Actual Total Cost")
            plt.ylabel("Predicted Total Cost")
            plt.title(f"Best Categorical Model: {name}\nR² = {r2:.3f}")
            
            # Add perfect prediction line
            min_val = min(y_test_reg.min(), preds.min())
            max_val = max(y_test_reg.max(), preds.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
            
            # Add best fit line
            z = np.polyfit(y_test_reg, preds, 1)
            p = np.poly1d(z)
            plt.plot(y_test_reg, p(y_test_reg), color="orange", linewidth=2, label="Best fit line")
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("best_categorical_model_plot.png", dpi=300)
            mlflow.log_artifact("best_categorical_model_plot.png")
            plt.close()

mlflow.end_run()

# --- Save best model ---
joblib.dump(best_overall_model, "categorical_model_pipeline.pkl")
print(f"\n🎯 Best categorical-only model: {best_overall_name}")
print(f"📊 R² Score: {best_overall_r2:.3f}")
print(f"📦 Saved to: categorical_model_pipeline.pkl")

# --- Feature importance analysis for tree-based models ---
if hasattr(best_overall_model.named_steps['regressor'], 'feature_importances_'):
    feature_names = best_overall_model.named_steps['preprocessor'].get_feature_names_out()
    importances = best_overall_model.named_steps['regressor'].feature_importances_
    
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    print("\n📈 Top 10 Most Important Features:")
    print(feature_importance_df.head(10))
    
    # Plot feature importance
    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(15)
    plt.barh(range(len(top_features)), top_features['importance'])
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Top 15 Feature Importances (Categorical-Only Model)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig("categorical_feature_importance.png", dpi=300)
    plt.close()
    
    feature_importance_df.to_csv("categorical_feature_importance.csv", index=False)

# --- Classification evaluation ---
final_preds = predicted_costs[best_overall_name]
pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])

with mlflow.start_run(run_name=f"{best_overall_name}_Classifier_{timestamp}"):
    acc = accuracy_score(y_test_cls, pred_tiers)
    prec = precision_score(y_test_cls, pred_tiers, average="macro")
    rec = recall_score(y_test_cls, pred_tiers, average="macro")
    f1 = f1_score(y_test_cls, pred_tiers, average="macro")
    
    mlflow.log_metric("Accuracy", acc)
    mlflow.log_metric("Precision", prec)
    mlflow.log_metric("Recall", rec)
    mlflow.log_metric("F1_score", f1)

    try:
        auc = roc_auc_score(y_test_cls, pd.get_dummies(pred_tiers), multi_class="ovo", average="macro")
        mlflow.log_metric("AUC_ROC", auc)
    except:
        pass

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test_cls, pred_tiers, ax=ax)
    plt.title(f"Categorical Model - Confusion Matrix\n{best_overall_name}")
    file_name = f"confusion_matrix_categorical_{timestamp}.png"
    plt.savefig(file_name, dpi=300)
    mlflow.log_artifact(file_name)
    plt.close()

# --- Save results ---
result_df = pd.DataFrame({
    "Actual_TCA": y_test_reg.values,
    "Predicted_TCA": final_preds,
    "Actual_Affordability": y_test_cls.values,
    "Predicted_Affordability": pred_tiers
})
result_df.to_csv("categorical_model_predictions.csv", index=False)

performance_df = pd.DataFrame(performance_log)
performance_df.to_csv("categorical_model_performance.csv", index=False)

print("\n" + "=" * 50)
print("CATEGORICAL-ONLY MODEL RESULTS")
print("=" * 50)
print(f"📊 Model Performance Summary:")
print(performance_df.sort_values('R2', ascending=False))
print(f"\n🎯 Classification Report:")
print(classification_report(y_test_cls, pred_tiers))
print(f"\n✅ Files saved:")
print(f"  • categorical_model_pipeline.pkl")
print(f"  • categorical_model_predictions.csv")
print(f"  • categorical_model_performance.csv")
print(f"  • categorical_feature_importance.csv")
print("=" * 50)
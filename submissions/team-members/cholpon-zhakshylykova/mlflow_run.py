import shutil
import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import sklearn
import xgboost
import joblib
from datetime import datetime
import glob


from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, OrdinalEncoder
from sklearn.metrics import (
   mean_absolute_error, mean_squared_error, r2_score,
   accuracy_score, precision_score, recall_score, f1_score,
   roc_auc_score, log_loss, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR


print("=" * 60)
print("CATEGORICAL-ONLY ML PIPELINE - ENHANCED VERSION")
print("=" * 60)
print(f"XGBoost version: {xgboost.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print("=" * 60)


# --- Clean old plots ---
for f in glob.glob("*.png"):
   os.remove(f)


# --- Load data ---
data = pd.read_csv("data_full_filtered.csv")


# --- Enhanced Feature config - ONLY CATEGORICAL ---
categorical_features = ["Country", "City", "University", "Program", "Level"]


# Display categorical feature information
print("\n📊 CATEGORICAL FEATURES ANALYSIS")
print("=" * 40)
for feature in categorical_features:
   unique_count = data[feature].nunique()
   print(f"{feature}: {unique_count} unique values")
   if unique_count <= 10:
       print(f"  Values: {list(data[feature].unique())}")
   else:
       print(f"  Sample values: {list(data[feature].unique()[:5])}...")
   print(f"  Missing values: {data[feature].isnull().sum()}")
print("=" * 40)


# --- Target variable preparation ---
target_column = "Total_cost"
X = data[categorical_features]
y_reg = data[target_column]


# Handle missing values in categorical features
print("\n🔧 HANDLING MISSING VALUES")
missing_before = X.isnull().sum().sum()
X = X.fillna('Unknown')  # Fill missing categorical values
missing_after = X.isnull().sum().sum()
print(f"Missing values before: {missing_before}")
print(f"Missing values after: {missing_after}")


# Create cost tiers for classification
y_cls = pd.qcut(y_reg, q=3, labels=["Low", "Medium", "High"])


print(f"\n📈 TARGET VARIABLE STATISTICS")
print(f"Total Cost - Mean: ${y_reg.mean():.2f}")
print(f"Total Cost - Median: ${y_reg.median():.2f}")
print(f"Total Cost - Std: ${y_reg.std():.2f}")
print(f"Total Cost - Range: ${y_reg.min():.2f} - ${y_reg.max():.2f}")
print(f"\nCost Distribution:")
print(y_cls.value_counts())


# --- MLflow setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Education_Cost_Categorical_Pipeline")


# --- Enhanced Preprocessing Options ---
# 1. One-Hot Encoding (handles high cardinality)
onehot_preprocessor = ColumnTransformer([
   ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), categorical_features)
])


# 2. One-Hot Encoding with frequency threshold (for very high cardinality)
onehot_freq_preprocessor = ColumnTransformer([
   ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False, max_categories=20), categorical_features)
])


# 3. Label Encoding (for tree-based models)
label_preprocessor = ColumnTransformer([
   ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), categorical_features)
])


# --- Enhanced Model Selection ---
regressors = {
   "RandomForest": RandomForestRegressor(random_state=42),
   "GradientBoosting": GradientBoostingRegressor(random_state=42),
   "XGBoost": XGBRegressor(objective="reg:squarederror", verbosity=0, random_state=42),
   "DecisionTree": DecisionTreeRegressor(random_state=42),
   "LinearRegression": LinearRegression(),
   "Ridge": Ridge(),
   "Lasso": Lasso(),
   "SVR": SVR(kernel='rbf')
}


# --- Enhanced Parameter Grids ---
param_grids = {
   "RandomForest": {
       "regressor__n_estimators": [100, 200],
       "regressor__max_depth": [10, 15, None],
       "regressor__min_samples_split": [2, 5],
       "regressor__min_samples_leaf": [1, 2]
   },
   "GradientBoosting": {
       "regressor__n_estimators": [100, 150],
       "regressor__learning_rate": [0.1, 0.15],
       "regressor__max_depth": [5, 7],
       "regressor__min_samples_split": [2, 4]
   },
   "XGBoost": {
       "regressor__n_estimators": [100, 150],
       "regressor__max_depth": [5, 7],
       "regressor__learning_rate": [0.1, 0.15],
       "regressor__min_child_weight": [1, 3]
   },
   "DecisionTree": {
       "regressor__max_depth": [10, 15, 20],
       "regressor__min_samples_split": [2, 5, 10],
       "regressor__min_samples_leaf": [1, 2, 4]
   },
   "LinearRegression": {},
   "Ridge": {"regressor__alpha": [0.1, 1.0, 10.0]},
   "Lasso": {"regressor__alpha": [0.1, 1.0, 10.0]},
   "SVR": {"regressor__C": [0.1, 1.0, 10.0], "regressor__gamma": ['scale', 'auto']}
}


# --- Train/test split ---
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
   X, y_reg, y_cls, test_size=0.2, random_state=42, stratify=y_cls
)


print(f"\n🎯 TRAIN/TEST SPLIT")
print(f"Training set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# --- Initialize tracking variables ---
best_overall_model = None
best_overall_r2 = -np.inf
best_overall_name = ""
predicted_costs = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
performance_log = []


# --- Enhanced Model Training Loop ---
preprocessors = [
   ("OneHot", onehot_preprocessor),
   ("OneHot_Freq", onehot_freq_preprocessor),
   ("Label", label_preprocessor)
]


print(f"\n🚀 TRAINING MODELS")
print("=" * 60)


for name, regressor in regressors.items():
   for prep_name, preprocessor in preprocessors:
      
       # Skip certain combinations that don't work well
       if prep_name == "Label" and name in ["LinearRegression", "Ridge", "Lasso", "SVR"]:
           continue
      
       run_name = f"{name}_{prep_name}_Regressor_{timestamp}"
       print(f"Training: {run_name}")
      
       with mlflow.start_run(run_name=run_name):
           # Log system information
           mlflow.log_param("python_version", sys.version)
           mlflow.log_param("sklearn_version", sklearn.__version__)
           mlflow.log_param("xgboost_version", xgboost.__version__)
           mlflow.log_param("preprocessing", prep_name)
          
           # Create pipeline
           pipe = Pipeline([
               ("preprocessor", preprocessor),
               ("regressor", regressor)
           ])
          
           # Grid search with cross-validation
           try:
               grid = GridSearchCV(
                   pipe,
                   param_grids[name],
                   cv=5,
                   scoring="r2",
                   n_jobs=-1,
                   verbose=0
               )
               grid.fit(X_train, y_train_reg)
               best_model = grid.best_estimator_
              
               # Log model
               mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:1])
              
               # Predictions
               preds = best_model.predict(X_test)
               predicted_costs[run_name] = preds
              
               # Calculate metrics
               mae = mean_absolute_error(y_test_reg, preds)
               mse = mean_squared_error(y_test_reg, preds)
               rmse = np.sqrt(mse)
               r2 = r2_score(y_test_reg, preds)
              
               # Cross-validation score
               cv_scores = cross_val_score(best_model, X_train, y_train_reg, cv=5, scoring='r2')
               cv_mean = cv_scores.mean()
               cv_std = cv_scores.std()
              
               # Log parameters and metrics
               mlflow.log_params(grid.best_params_)
               mlflow.log_metric("MAE", mae)
               mlflow.log_metric("MSE", mse)
               mlflow.log_metric("RMSE", rmse)
               mlflow.log_metric("R2", r2)
               mlflow.log_metric("CV_R2_Mean", cv_mean)
               mlflow.log_metric("CV_R2_Std", cv_std)
              
               # Log performance
               performance_log.append({
                   "Model": name,
                   "Preprocessing": prep_name,
                   "MAE": mae,
                   "MSE": mse,
                   "RMSE": rmse,
                   "R2": r2,
                   "CV_R2_Mean": cv_mean,
                   "CV_R2_Std": cv_std,
                   "Best_Params": grid.best_params_
               })
              
               print(f"  R² Score: {r2:.4f} | CV R² Mean: {cv_mean:.4f} ± {cv_std:.4f}")
              
               # Track best model
               if r2 > best_overall_r2:
                   best_overall_r2 = r2
                   best_overall_model = best_model
                   best_overall_name = run_name
                  
                   # Create enhanced visualization for best model
                   fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
                  
                   # 1. Predicted vs Actual scatter plot
                   ax1.scatter(y_test_reg, preds, alpha=0.6, color='blue')
                   ax1.set_xlabel("Actual Total Cost")
                   ax1.set_ylabel("Predicted Total Cost")
                   ax1.set_title(f"Predicted vs Actual - {run_name}")
                  
                   # Best fit line
                   z = np.polyfit(y_test_reg, preds, 1)
                   p = np.poly1d(z)
                   ax1.plot(y_test_reg, p(y_test_reg), color="red", linewidth=2, label="Best fit line")
                  
                   # Perfect prediction line
                   min_val = min(y_test_reg.min(), preds.min())
                   max_val = max(y_test_reg.max(), preds.max())
                   ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label="Perfect prediction")
                   ax1.legend()
                   ax1.grid(True, alpha=0.3)
                  
                   # 2. Residuals plot
                   residuals = y_test_reg - preds
                   ax2.scatter(preds, residuals, alpha=0.6, color='green')
                   ax2.axhline(y=0, color='red', linestyle='--')
                   ax2.set_xlabel("Predicted Total Cost")
                   ax2.set_ylabel("Residuals")
                   ax2.set_title("Residuals Plot")
                   ax2.grid(True, alpha=0.3)
                  
                   # 3. Feature importance (if available)
                   if hasattr(best_model.named_steps['regressor'], 'feature_importances_'):
                       # Get feature names after preprocessing
                       feature_names = best_model.named_steps['preprocessor'].get_feature_names_out()
                       importances = best_model.named_steps['regressor'].feature_importances_
                      
                       # Sort and get top 10
                       indices = np.argsort(importances)[::-1][:10]
                       top_features = [feature_names[i] for i in indices]
                       top_importances = importances[indices]
                      
                       ax3.barh(range(len(top_features)), top_importances)
                       ax3.set_yticks(range(len(top_features)))
                       ax3.set_yticklabels(top_features)
                       ax3.set_xlabel("Importance")
                       ax3.set_title("Top 10 Feature Importances")
                       ax3.grid(True, alpha=0.3)
                   else:
                       ax3.text(0.5, 0.5, 'Feature importance\nnot available\nfor this model',
                               ha='center', va='center', transform=ax3.transAxes, fontsize=12)
                       ax3.set_title("Feature Importance")
                  
                   # 4. Distribution of predictions vs actual
                   ax4.hist(y_test_reg, bins=20, alpha=0.7, label='Actual', color='blue')
                   ax4.hist(preds, bins=20, alpha=0.7, label='Predicted', color='red')
                   ax4.set_xlabel("Total Cost")
                   ax4.set_ylabel("Frequency")
                   ax4.set_title("Distribution Comparison")
                   ax4.legend()
                   ax4.grid(True, alpha=0.3)
                  
                   plt.tight_layout()
                   plt.savefig("best_model_analysis.png", dpi=300, bbox_inches='tight')
                   mlflow.log_artifact("best_model_analysis.png")
                   plt.close()
                  
           except Exception as e:
               print(f"  Error training {run_name}: {str(e)}")
               continue


print("=" * 60)
print(f"🏆 BEST MODEL: {best_overall_name}")
print(f"🎯 Best R² Score: {best_overall_r2:.4f}")
print("=" * 60)


# --- Save best model ---
joblib.dump(best_overall_model, "model_pipeline.pkl")
print(f"\n💾 Saved best model to model_pipeline.pkl")


# --- Enhanced Classification Analysis ---
if best_overall_name in predicted_costs:
   final_preds = predicted_costs[best_overall_name]
   pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])
  
   with mlflow.start_run(run_name=f"{best_overall_name}_Classification_{timestamp}"):
       # Classification metrics
       acc = accuracy_score(y_test_cls, pred_tiers)
       prec = precision_score(y_test_cls, pred_tiers, average="macro")
       rec = recall_score(y_test_cls, pred_tiers, average="macro")
       f1 = f1_score(y_test_cls, pred_tiers, average="macro")
      
       # Handle log loss calculation
       try:
           pred_proba = pd.get_dummies(pred_tiers)
           actual_proba = pd.get_dummies(y_test_cls)
           # Align columns
           for col in actual_proba.columns:
               if col not in pred_proba.columns:
                   pred_proba[col] = 0
           pred_proba = pred_proba[actual_proba.columns]
           # Add small epsilon to avoid log(0)
           pred_proba = pred_proba + 1e-15
           pred_proba = pred_proba.div(pred_proba.sum(axis=1), axis=0)
           logloss = log_loss(actual_proba, pred_proba)
       except:
           logloss = float('inf')
      
       # Log classification metrics
       mlflow.log_metric("Accuracy", acc)
       mlflow.log_metric("Precision", prec)
       mlflow.log_metric("Recall", rec)
       mlflow.log_metric("F1_score", f1)
       if logloss != float('inf'):
           mlflow.log_metric("LogLoss", logloss)
      
       # AUC ROC for multiclass
       try:
           pred_proba_numeric = pd.get_dummies(pred_tiers).values
           actual_proba_numeric = pd.get_dummies(y_test_cls).values
           auc = roc_auc_score(actual_proba_numeric, pred_proba_numeric, multi_class="ovo", average="macro")
           mlflow.log_metric("AUC_ROC", auc)
       except:
           pass
      
       # Enhanced confusion matrix visualization
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      
       # Confusion matrix
       ConfusionMatrixDisplay.from_predictions(y_test_cls, pred_tiers, ax=ax1, cmap='Blues')
       ax1.set_title(f"Confusion Matrix - {best_overall_name}")
      
       # Classification distribution
       classification_df = pd.DataFrame({
           'Actual': y_test_cls,
           'Predicted': pred_tiers
       })
      
       actual_counts = classification_df['Actual'].value_counts()
       predicted_counts = classification_df['Predicted'].value_counts()
      
       x = np.arange(len(actual_counts))
       width = 0.35
      
       ax2.bar(x - width/2, actual_counts.values, width, label='Actual', alpha=0.8)
       ax2.bar(x + width/2, predicted_counts.values, width, label='Predicted', alpha=0.8)
       ax2.set_xlabel('Cost Category')
       ax2.set_ylabel('Count')
       ax2.set_title('Actual vs Predicted Distribution')
       ax2.set_xticks(x)
       ax2.set_xticklabels(actual_counts.index)
       ax2.legend()
       ax2.grid(True, alpha=0.3)
      
       plt.tight_layout()
       file_name = f"classification_analysis_{best_overall_name}.png"
       plt.savefig(file_name, dpi=300, bbox_inches='tight')
       mlflow.log_artifact(file_name)
       plt.close()


# --- Enhanced Results Export ---
if best_overall_name in predicted_costs:
   final_preds = predicted_costs[best_overall_name]
   pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])
  
   result_df = pd.DataFrame({
       "Actual_TCA": y_test_reg.values,
       "Predicted_TCA": final_preds,
       "Prediction_Error": y_test_reg.values - final_preds,
       "Actual_Affordability": y_test_cls.values,
       "Predicted_Affordability": pred_tiers
   })
  
   # Add categorical features for analysis
   test_indices = X_test.index
   for feature in categorical_features:
       result_df[feature] = data.loc[test_indices, feature].values
  
   result_df.to_csv("enhanced_model_predictions.csv", index=False)
   print("📊 Saved enhanced predictions to enhanced_model_predictions.csv")


# --- Enhanced Performance Summary ---
performance_df = pd.DataFrame(performance_log)
performance_df = performance_df.sort_values('R2', ascending=False)
performance_df.to_csv("enhanced_model_performance_log.csv", index=False)


print("\n📈 TOP 5 MODELS BY R² SCORE:")
print("=" * 80)
top_5 = performance_df.head()
for idx, row in top_5.iterrows():
   print(f"{row['Model']} ({row['Preprocessing']}): R² = {row['R2']:.4f}, RMSE = {row['RMSE']:.2f}")


print("\n🎯 CLASSIFICATION RESULTS:")
print("=" * 40)
if best_overall_name in predicted_costs:
   print(classification_report(y_test_cls, pred_tiers, target_names=["Low", "Medium", "High"]))


print("\n✅ PIPELINE COMPLETE!")
print(f"🏆 Best Model: {best_overall_name}")
print(f"📊 R² Score: {best_overall_r2:.4f}")
print(f"💾 Model saved as: model_pipeline.pkl")
print(f"📈 Performance log: enhanced_model_performance_log.csv")
print(f"🎯 Predictions: enhanced_model_predictions.csv")






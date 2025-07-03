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

from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression, RFE, SelectFromModel
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, ConfusionMatrixDisplay, classification_report
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso

print("=" * 50)
print("PACKAGE VERSIONS")
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

# --- Feature config ---
categorical_features = ["Country", "City", "University", "Program", "Level"]
numeric_features = ["Tuition_USD", "Living_Cost_Index", "Rent_USD", "Visa_Fee_USD", "Insurance_USD", "Duration_Years"]

target_column = "Total_cost"
X = data[categorical_features + numeric_features]
y_reg = data[target_column]
y_cls = pd.qcut(y_reg, q=3, labels=["Low", "Medium", "High"])

# --- MLflow setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Education_Cost_Pipeline_with_Feature_Selection")

# --- Feature Analysis Functions ---
def analyze_feature_correlation(X, y, numeric_features):
    """Analyze correlation between numeric features and target"""
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*60)
    
    # Calculate correlations for numeric features
    correlations = {}
    for feature in numeric_features:
        corr = X[feature].corr(y)
        correlations[feature] = corr
        print(f"{feature}: {corr:.3f}")
    
    # Plot correlation heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = X[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('feature_correlation_matrix.png')
    plt.close()
    
    return correlations

def get_feature_names_after_preprocessing(preprocessor, categorical_features, numeric_features):
    """Get feature names after preprocessing"""
    # Fit preprocessor to get feature names
    cat_encoder = preprocessor.named_transformers_['cat']
    
    # Get categorical feature names after one-hot encoding
    cat_feature_names = []
    if hasattr(cat_encoder, 'get_feature_names_out'):
        cat_feature_names = cat_encoder.get_feature_names_out(categorical_features).tolist()
    else:
        # Fallback for older sklearn versions
        for i, feature in enumerate(categorical_features):
            categories = cat_encoder.categories_[i]
            # Skip first category due to drop='first'
            for cat in categories[1:]:
                cat_feature_names.append(f"{feature}_{cat}")
    
    # Combine with numeric features
    all_feature_names = cat_feature_names + numeric_features
    return all_feature_names

def analyze_feature_importance(model, feature_names, model_name):
    """Analyze feature importance for tree-based models"""
    if hasattr(model.named_steps['regressor'], 'feature_importances_'):
        importances = model.named_steps['regressor'].feature_importances_
        
        # Create feature importance dataframe
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        # Map back to original features
        original_feature_mapping = map_to_original_features(feature_importance_df, categorical_features, numeric_features)
        
        print(f"\n{model_name} - Top 10 Most Important Features:")
        print("-" * 50)
        for idx, row in feature_importance_df.head(10).iterrows():
            original_feat = original_feature_mapping.get(row['feature'], row['feature'])
            print(f"{row['feature']:<30} | {original_feat:<15} | {row['importance']:.4f}")
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title(f'{model_name} - Feature Importance')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(f'feature_importance_{model_name.replace(" ", "_")}.png')
        plt.close()
        
        return feature_importance_df
    
    return None

def map_to_original_features(feature_df, categorical_features, numeric_features):
    """Map processed feature names back to original features"""
    mapping = {}
    
    for _, row in feature_df.iterrows():
        feature_name = row['feature']
        
        # Check if it's a numeric feature
        if feature_name in numeric_features:
            mapping[feature_name] = feature_name
        else:
            # Check if it's a categorical feature (one-hot encoded)
            for cat_feat in categorical_features:
                if feature_name.startswith(f"{cat_feat}_"):
                    mapping[feature_name] = cat_feat
                    break
    
    return mapping

def perform_feature_selection(X_train_processed, y_train, feature_names, method='univariate', k=10):
    """Perform feature selection and return selected features"""
    print(f"\n{'='*60}")
    print(f"FEATURE SELECTION - {method.upper()} METHOD")
    print("="*60)
    
    if method == 'univariate':
        selector = SelectKBest(score_func=f_regression, k=k)
    elif method == 'rfe':
        selector = RFE(estimator=RandomForestRegressor(n_estimators=50, random_state=42), n_features_to_select=k)
    elif method == 'tree_based':
        selector = SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=42), max_features=k)
    
    X_selected = selector.fit_transform(X_train_processed, y_train)
    selected_features = selector.get_support()
    selected_feature_names = [feature_names[i] for i in range(len(feature_names)) if selected_features[i]]
    
    print(f"Selected {len(selected_feature_names)} features out of {len(feature_names)}:")
    
    # Group by original feature
    original_feature_counts = {}
    for feat in selected_feature_names:
        for cat_feat in categorical_features:
            if feat.startswith(f"{cat_feat}_"):
                original_feature_counts[cat_feat] = original_feature_counts.get(cat_feat, 0) + 1
                break
        else:
            if feat in numeric_features:
                original_feature_counts[feat] = original_feature_counts.get(feat, 0) + 1
    
    print("\nOriginal features and their selected encoded versions:")
    for orig_feat, count in sorted(original_feature_counts.items()):
        if orig_feat in categorical_features:
            print(f"  {orig_feat}: {count} categories selected")
        else:
            print(f"  {orig_feat}: selected")
    
    return selector, selected_feature_names

# --- Initial Feature Analysis ---
print("\n" + "="*60)
print("INITIAL FEATURE ANALYSIS")
print("="*60)
print(f"Total features: {len(categorical_features + numeric_features)}")
print(f"Categorical features: {categorical_features}")
print(f"Numeric features: {numeric_features}")

# Analyze correlations
correlations = analyze_feature_correlation(X, y_reg, numeric_features)

# --- Preprocessing ---
scaling_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ("num", StandardScaler(), numeric_features)
])

non_scaling_preprocessor = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features),
    ("num", "passthrough", numeric_features)
])

# --- Train/test split ---
X_train, X_test, y_train_reg, y_test_reg, y_train_cls, y_test_cls = train_test_split(
    X, y_reg, y_cls, test_size=0.2, random_state=42
)

# Get feature names after preprocessing
scaling_preprocessor.fit(X_train)
feature_names_after_preprocessing = get_feature_names_after_preprocessing(
    scaling_preprocessor, categorical_features, numeric_features
)

print(f"\nFeatures after preprocessing: {len(feature_names_after_preprocessing)}")

# --- Feature Selection ---
X_train_processed = scaling_preprocessor.transform(X_train)

# Try different feature selection methods
selection_methods = ['univariate', 'rfe', 'tree_based']
selectors = {}

for method in selection_methods:
    selector, selected_features = perform_feature_selection(
        X_train_processed, y_train_reg, feature_names_after_preprocessing, 
        method=method, k=15
    )
    selectors[method] = selector

# --- Regressors ---
regressors = {
    "RandomForest": RandomForestRegressor(random_state=42),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost": XGBRegressor(objective="reg:squarederror", verbosity=0, random_state=42),
    "LinearRegression": LinearRegression(),
    "Ridge": Ridge(random_state=42),
    "Lasso": Lasso(random_state=42)
}

# --- Param grids ---
param_grids = {
    "RandomForest": {"regressor__n_estimators": [100], "regressor__max_depth": [10]},
    "GradientBoosting": {"regressor__n_estimators": [100], "regressor__learning_rate": [0.1]},
    "XGBoost": {"regressor__n_estimators": [100], "regressor__max_depth": [5]},
    "LinearRegression": {},
    "Ridge": {"regressor__alpha": [0.1, 1.0]},
    "Lasso": {"regressor__alpha": [0.1, 1.0]}
}

best_overall_model = None
best_overall_r2 = -np.inf
best_overall_name = ""
predicted_costs = {}
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

performance_log = []
feature_importance_results = {}

# --- Training loop ---
configurations = [
    ("all_features", "scaled", scaling_preprocessor, None),
    ("all_features", "unscaled", non_scaling_preprocessor, None),
    ("selected_univariate", "scaled", scaling_preprocessor, selectors['univariate']),
    ("selected_rfe", "scaled", scaling_preprocessor, selectors['rfe']),
    ("selected_tree", "scaled", scaling_preprocessor, selectors['tree_based'])
]

for name, regressor in regressors.items():
    for config_name, scaling, preprocessor, selector in configurations:
        run_name = f"{name}_{config_name}_{scaling}_{timestamp}"
        
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("python_version", sys.version)
            mlflow.log_param("sklearn_version", sklearn.__version__)
            mlflow.log_param("xgboost_version", xgboost.__version__)
            mlflow.log_param("feature_config", config_name)
            mlflow.log_param("scaling", scaling)
            
            # Create pipeline
            if selector is not None:
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("selector", selector),
                    ("regressor", regressor)
                ])
            else:
                pipe = Pipeline([
                    ("preprocessor", preprocessor),
                    ("regressor", regressor)
                ])
            
            # Grid search
            grid = GridSearchCV(pipe, param_grids[name], cv=5, scoring="r2", n_jobs=-1)
            grid.fit(X_train, y_train_reg)
            best_model = grid.best_estimator_
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model", input_example=X_test[:1])
            
            # Predictions
            preds = best_model.predict(X_test)
            predicted_costs[run_name] = preds
            
            # Metrics
            mae = mean_absolute_error(y_test_reg, preds)
            mse = mean_squared_error(y_test_reg, preds)
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test_reg, preds)
            
            # Log parameters and metrics
            mlflow.log_params(grid.best_params_)
            mlflow.log_metric("MAE", mae)
            mlflow.log_metric("MSE", mse)
            mlflow.log_metric("RMSE", rmse)
            mlflow.log_metric("R2", r2)
            
            # Feature importance analysis
            if selector is None:
                current_feature_names = feature_names_after_preprocessing
            else:
                # Get selected feature names
                if hasattr(selector, 'get_support'):
                    selected_mask = selector.get_support()
                    current_feature_names = [feature_names_after_preprocessing[i] 
                                           for i in range(len(feature_names_after_preprocessing)) 
                                           if selected_mask[i]]
                else:
                    current_feature_names = feature_names_after_preprocessing
            
            importance_df = analyze_feature_importance(best_model, current_feature_names, run_name)
            if importance_df is not None:
                feature_importance_results[run_name] = importance_df
            
            # Performance log
            performance_log.append({
                "Model": name,
                "Feature_Config": config_name,
                "Scaling": scaling,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "Best_Params": grid.best_params_,
                "Num_Features": len(current_feature_names)
            })
            
            # Track best model
            if r2 > best_overall_r2:
                best_overall_r2 = r2
                best_overall_model = best_model
                best_overall_name = run_name
                
                # Save best model plot
                plt.figure(figsize=(10, 6))
                plt.scatter(y_test_reg, preds, alpha=0.6, s=50)
                plt.xlabel("Actual Total Cost")
                plt.ylabel("Predicted Total Cost")
                plt.title(f"Best Model: {run_name}\nR² = {r2:.4f}")
                
                # Add perfect prediction line
                min_val = min(y_test_reg.min(), preds.min())
                max_val = max(y_test_reg.max(), preds.max())
                plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
                
                # Add best fit line
                z = np.polyfit(y_test_reg, preds, 1)
                p = np.poly1d(z)
                plt.plot(y_test_reg, p(y_test_reg), color="blue", linewidth=2, label="Best Fit Line")
                
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig("best_model_plot.png", dpi=300, bbox_inches='tight')
                mlflow.log_artifact("best_model_plot.png")
                plt.close()

mlflow.end_run()

# --- Save results ---
joblib.dump(best_overall_model, "model_pipeline.pkl")
print(f"\n🎯 BEST MODEL RESULTS")
print("="*60)
print(f"Best Model: {best_overall_name}")
print(f"R² Score: {best_overall_r2:.4f}")
print(f"Saved to: model_pipeline.pkl")

# Classification analysis
final_preds = predicted_costs[best_overall_name]
pred_tiers = pd.qcut(final_preds, q=3, labels=["Low", "Medium", "High"])

# Save detailed results
result_df = pd.DataFrame({
    "Actual_TCA": y_test_reg.values,
    "Predicted_TCA": final_preds,
    "Actual_Affordability": y_test_cls.values,
    "Predicted_Affordability": pred_tiers
})
result_df.to_csv("model_predictions.csv", index=False)

performance_df = pd.DataFrame(performance_log)
performance_df.to_csv("model_performance_log.csv", index=False)

# Save feature importance summary
if feature_importance_results:
    with open("feature_importance_summary.txt", "w") as f:
        f.write("FEATURE IMPORTANCE SUMMARY\n")
        f.write("="*50 + "\n\n")
        
        for model_name, importance_df in feature_importance_results.items():
            f.write(f"Model: {model_name}\n")
            f.write("-" * 30 + "\n")
            
            # Map to original features
            original_mapping = map_to_original_features(importance_df, categorical_features, numeric_features)
            
            for _, row in importance_df.head(10).iterrows():
                original_feat = original_mapping.get(row['feature'], row['feature'])
                f.write(f"{row['feature']:<30} | {original_feat:<15} | {row['importance']:.4f}\n")
            f.write("\n")

print("\n📊 FINAL RESULTS SUMMARY")
print("="*60)
print("Files created:")
print("- model_pipeline.pkl (Best trained model)")
print("- model_predictions.csv (Predictions on test set)")
print("- model_performance_log.csv (All model performances)")
print("- feature_importance_summary.txt (Feature importance analysis)")
print("- Various PNG plots for visualization")

print(f"\n📈 Performance Summary:")
print(f"Best performing configuration: {best_overall_name}")
print(f"R² Score: {best_overall_r2:.4f}")

# Show top performing models
print(f"\nTop 5 Models by R² Score:")
top_models = performance_df.nlargest(5, 'R2')[['Model', 'Feature_Config', 'Scaling', 'R2', 'Num_Features']]
print(top_models.to_string(index=False))

print(f"\n🎯 Original Feature Importance (from correlations):")
sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
for feature, corr in sorted_correlations:
    print(f"{feature:<20}: {corr:>7.4f}")

print("\n✅ Analysis complete! Check the generated files for detailed results.")
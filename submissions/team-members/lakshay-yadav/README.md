# 🎓 EduSpend — Budget Planner (Streamlit App)

**Author:** Lakshay Yadav  
**Date:** July 2025  
**Status:** ✅ Streamlit App & ML Models Deployed

---

## 🎯 PROJECT OVERVIEW

EduSpend is an ML-powered budget planner to estimate the **Total Cost of Attendance (TCA)** and classify the **Affordability Tier** (Low/Medium/High) for students planning to study abroad. This project uses trained machine learning models and a dynamic Streamlit interface to generate real-time, user-driven predictions.

---

## 🌟 FEATURES

- 🧮 **TCA Prediction** (Regression)
- 🏷️ **Affordability Tier Classification**
- 🌍 **Dynamic Country/City/Degree Dropdowns**
- 🧾 Manual input fields for tuition, rent, insurance
- 📈 Clean, single-page **Streamlit UI**
- 💾 Models and preprocessor loaded from disk
- 🚀 Fully deployable on Streamlit Cloud

---

## 📁 PROJECT STRUCTURE

```bash
EduSpend/
├── app.py                          # Streamlit App
├── requirements.txt               # Deployment dependencies
├── deployment_models/
│   ├── best_regressor.pkl         # Trained regression model
│   └── best_classifier.pkl        # Trained classification model
├── preprocessing/
│   └── preprocessor.pkl           # Saved ColumnTransformer
├── data/
│   └── International_Education_Costs.csv
|   └── cleaned_education_costs.csv
└── README.md                      # Project documentation
```

---

## 🧪 MACHINE LEARNING PIPELINE

### 1. **Phase 1: EDA & Preprocessing**
- Cleaned international education cost data
- Engineered `TCA` as target variable
- Processed missing values, encoded categorical columns
- Exported cleaned dataset

### 2. **Phase 2: Model Development**
- Trained regression models: RandomForest, XGBoost, GradientBoosting
- Trained classification models: RandomForest, XGBoost, GradientBoosting
- Used `RandomizedSearchCV` for tuning
- Logged results and metrics using **MLflow**
- Saved best models and preprocessor with `joblib`

### 3. **Phase 3: Deployment**
- Built `app.py` in Streamlit for real-time prediction
- Integrated dynamic dropdowns (country, city, level)
- Deployed models with correct preprocessing pipeline
- Tested app locally and on Streamlit Cloud

---

## ⚠️ CHALLENGES FACED

### ❌ NumPy DLL Import Error
- Encountered `numpy._core.multiarray failed to import`
- 🛠️ Fixed by downgrading NumPy to `1.24.4` and rebuilding environment

### ❌ OneDrive Path Errors
- Models failed to load due to locked `.pkl` files inside OneDrive
- 🛠️ Moved entire project to a local directory

### ❌ Version Incompatibility in Deployment
- Errors due to mismatched versions of `scikit-learn`, `xgboost`, and `joblib`
- 🛠️ Fixed with a fresh environment (`eduspend3`) and strict version control

### ❌ Static UI Inputs
- Country, city, and level fields were initially hardcoded
- 🛠️ Replaced with dynamic dropdowns based on the original dataset

---

## 💻 HOW TO RUN LOCALLY

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/eduspend-app.git
cd eduspend-app

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run the Streamlit app
streamlit run app.py
```

---

## 🌐 STREAMLIT CLOUD DEPLOYMENT

- ✅ App tested on Streamlit Cloud with working dropdowns and model predictions
- 🚀 Ready to share as a public link after uploading to GitHub

---

## 📷 UI PREVIEW

![App Screenshot](assets/UI.png)


---

## 📦 REQUIREMENTS (Core)

```txt
streamlit>=1.30.0
pandas==2.2.2
numpy==1.24.4
scikit-learn==1.3.2
xgboost==2.0.3
joblib>=1.2.0,<1.4.0
```

---

## 📄 LICENSE

This project is intended for academic and demonstration purposes under SDS-CP030. All models and code are created by Lakshay Yadav.
# EduSpend Project: Analysis & Modeling Guide

This document summarizes the workflow, findings, and setup for the EduSpend project, based on the work completed in the EDA and modeling notebooks.

---

## 1. Environment Setup

1. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # OR
   venv\Scripts\activate   # On Windows
   ```
2. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```
3. **Test your environment:**
   ```bash
   python test_environment.py
   ```

---

## 2. Data Analysis Workflow

### a. Data Loading & Profiling
- Data source: `International_Education_Costs.csv`
- Inspected shape, columns, and types (907 records, 12 columns)
- Explored distributions by country, city, level, and program

### b. Correlation Analysis
- Identified cost-related columns: Tuition, Rent, Living Cost Index, Visa Fee, Insurance, Exchange Rate
- Visualized correlation matrix for numeric variables

### c. Total Cost of Attendance (TCA)
- Calculated TCA as:
  > Tuition_USD + (Rent_USD × 12) + Visa_Fee_USD + Insurance_USD
- Visualized TCA distribution and average TCA by country
- TCA statistics:
  - Mean: ~$29,247
  - Median: ~$18,590
  - Min: $3,100
  - Max: $93,660

### d. Outlier Detection & Currency Stability
- Used boxplots to detect outliers in cost columns
- Analyzed exchange rate stability by country

### e. Summary Stats
- 71 countries, 556 cities, 92 programs, 3 education levels

---

## 3. Modeling Workflow

### a. Feature Engineering
- One-hot encoded categoricals: Country, City, Level, Program
- Scaled numeric features
- Created affordability tiers (Low/Medium/High) using TCA quantiles

### b. Regression: Predicting TCA
- Baseline: Linear Regression
  - MAE: ~0.24
  - RMSE: ~0.47
  - R²: 1.00
- XGBoost Regression
  - MAE: ~495
  - RMSE: ~706
  - R²: 1.00
- Hyperparameter tuning (GridSearchCV) for XGBoost
  - Best params: max_depth=7, n_estimators=100
  - Best MAE: ~607

### c. Classification: Affordability Tier
- Logistic Regression classifier
  - Accuracy: ~0.95
  - Macro F1: ~0.95

### d. Clustering
- KMeans clustering on scaled numeric features (n_clusters=3)
  - Silhouette Score: ~0.41
  - Visualized clusters using PCA

### e. Model Saving
- Saved best regression pipeline as `best_tca_model.joblib`

---

## 4. How to Run

- **EDA:** Open `EduSpend_EDA.ipynb` in Jupyter and run all cells to reproduce the exploratory analysis.
- **Modeling:** Open `EduSpend_Model.ipynb` to run feature engineering, regression, classification, clustering, and model saving steps.
- **Streamlit App:** (If available) Run `streamlit run streamlit_app.py` to launch the dashboard.

---

## 5. Resources
- Data: [Kaggle Dataset](https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education)
- Requirements: See `requirements.txt`

---

For questions or further details, refer to the notebooks or contact the project maintainer. 
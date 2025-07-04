# Cost of International Education Analysis

A comprehensive machine learning project for analyzing and predicting the costs of international education across different countries, universities, and programs. This project combines data science techniques with interactive web applications to provide insights into educational affordability and cost patterns.

## 🎯 Project Overview

This project analyzes international education costs using machine learning models and provides interactive tools for prospective students to explore affordability, compare programs, and predict educational expenses. The analysis includes both regression models for cost prediction and classification models for affordability categorization.

## 📊 Key Features

- **Cost Prediction Models**: Predict total cost of attendance for international education programs
- **Affordability Classification**: Categorize programs into Low, Medium, and High affordability tiers
- **Interactive Web Application**: User-friendly interface for exploring data and getting predictions
- **Cluster Analysis**: Identify patterns and segments in educational programs
- **Budget Planning Tools**: Help students plan their educational expenses
- **Comprehensive Visualizations**: Interactive charts and maps for data exploration

## 🏗️ Project Structure

```
cost-of-international-education-analysis/
├── mlflow_run.py                    # ML models and experiment tracking
├── webapp.py                        # Streamlit web application
├── data/
│   ├── preprocessed_df.csv          # Preprocessed dataset for ML models
│   └── data_with_clusters_and_reduced_features.csv  # Dataset with clustering results
├── models/                          # Trained model artifacts
├── requirements.txt                 # Project dependencies
└── README.md                        # Project documentation
```

## 🤖 Machine Learning Pipeline (`mlflow_run.py`)

The core ML pipeline implements both regression and classification tasks with comprehensive experiment tracking through MLflow.

### Regression Models
- **Models**: XGBoost, Random Forest, Gradient Boosting Machine (GBM)
- **Target Variable**: `Total_cost`
- **Evaluation Metrics**: 
  - Mean Absolute Error (MAE)
  - Mean Squared Error (MSE)
  - Root Mean Squared Error (RMSE)
  - R² Score

### Classification Models
- **Models**: XGBoost, GBM, Random Forest
- **Target Variable**: `Affordability` (Low, Medium, High)
- **Evaluation Metrics**:
  - Accuracy
  - Precision, Recall, F1 Score
  - AUC-ROC
  - Log Loss

### Advanced Features
- **Hyperparameter Tuning**: Grid search and random search optimization
- **MLflow Integration**: Complete experiment tracking and model versioning
- **Reproducibility**: Logged parameters, metrics, and artifacts
- **Model Comparison**: Side-by-side performance evaluation

## 🌐 Web Application (`webapp.py`)

An interactive Streamlit application that makes the analysis accessible to end users.

### Core Features

#### 1. Budget Planner
- Input target country, education level, and program duration
- Get personalized Total Cost of Attendance (TCA) forecasts
- Compare costs across different scenarios

#### 2. Affordability Map & Dashboards
- Visual representation of affordability tiers across countries
- Summary statistics and rankings
- Top universities and programs by affordability

#### 3. Cluster Explorer
- Interactive visualization of K-Means and HDBSCAN clustering results
- PCA and t-SNE dimensionality reduction plots
- Cluster insights and segment descriptions

#### 4. Predictive Models Integration
- Real-time cost predictions using trained ML models
- Affordability tier classification
- User-friendly input forms and result displays

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/cost-of-international-education-analysis.git
cd cost-of-international-education-analysis
```

2. Install required dependencies:
```bash
pip install -r requirements.txt
```

### Required Packages
```bash
pip install mlflow streamlit scikit-learn xgboost pandas numpy matplotlib seaborn plotly
```

## 📈 Usage

### Running the ML Pipeline

1. **Train Models and Track Experiments**:
```bash
python mlflow_run.py
```

2. **View MLflow UI**:
```bash
mlflow ui
```
Access the MLflow dashboard at `http://localhost:5000`

### Running the Web Application

1. **Launch Streamlit App**:
```bash
streamlit run webapp.py
```

2. **Access the Application**:
Open your browser and navigate to `http://localhost:8501`

## 📊 Data Requirements

The project expects two main datasets:

1. **`preprocessed_df.csv`**: Clean, preprocessed dataset for ML model training
2. **`data_with_clusters_and_reduced_features.csv`**: Dataset with clustering results and dimensionality reduction features

### Expected Data Schema
- `Total_cost`: Target variable for regression (numeric)
- `Affordability`: Target variable for classification (categorical: Low, Medium, High)
- Country, university, program-level features
- Preprocessed numerical and categorical features

## 🔍 Model Performance

The project tracks comprehensive metrics for both regression and classification tasks:

### Regression Performance
- **MAE**: Mean Absolute Error in cost prediction
- **RMSE**: Root Mean Squared Error
- **R²**: Coefficient of determination

### Classification Performance
- **Accuracy**: Overall classification accuracy
- **F1 Score**: Harmonic mean of precision and recall
- **AUC-ROC**: Area under the ROC curve

## 📁 Outputs

### MLflow Artifacts
- Trained model files
- Performance metrics
- Hyperparameter configurations
- Feature importance plots
- Confusion matrices

### Web App Outputs
- Interactive visualizations
- Cost predictions
- Affordability assessments
- Cluster analysis results
- Budget planning recommendations



## 🙏 Acknowledgments

- MLflow for experiment tracking capabilities
- Streamlit for the interactive web framework
- scikit-learn and XGBoost for machine learning algorithms
- The open-source community for supporting libraries



**Note**: This project is designed for educational and research purposes. Always verify predictions with official university sources before making financial decisions.
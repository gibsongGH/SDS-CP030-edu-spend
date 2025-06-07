# EduSpend Project: yan-cotta's Submission

**Author:** yan-cotta  
**Phase:** 2 - Model Development  
**Date:** June 7, 2025  

## Phase 1: Setup & Exploratory Data Analysis (Completed)

This phase involved setting up the project structure and performing a deep dive into the international education costs dataset to understand patterns, relationships, and data quality.

### Key Achievements from 01_EDA_EduSpend.ipynb:

- Loaded and inspected the dataset to understand its structure and identify missing values
- Analyzed cost distributions for tuition, rent, and other fees using histograms and box plots
- Visualized correlations between key financial metrics to uncover relationships
- Engineered a baseline 'Total Cost of Attendance' (TCA) feature for a holistic cost view
- Identified and analyzed outliers to ensure data quality for future modeling

## Project Overview
This project analyzes international education costs data to provide insights for students planning to study abroad. The goal is to develop a comprehensive cost analytics and planning tool.

## Project Structure
```
submissions/team-members/yan-cotta/
├── 01_EDA_EduSpend.ipynb      # Phase 1: Exploratory Data Analysis
├── 02_Model_Development.ipynb # Phase 2: Model Development
├── data/                      # Dataset folder
│   └── International_Education_Costs.csv
├── requirements.txt           # Project dependencies
├── activate_env.sh            # Environment activation script
├── venv/                      # Virtual environment
└── README.md                  # This file
```

## Setup Instructions

### 1. Dataset Setup
- Download the `International_Education_Costs.csv` dataset from Kaggle
- Place it in the `data/` folder

### 2. Environment Activation
You have several options to activate the environment:

**Option A: Using the activation script**
```bash
cd /home/yan/Documents/Git/SDS-CP030-edu-spend/submissions/team-members/yan-cotta
./activate_env.sh
```

**Option B: Manual activation**
```bash
cd /home/yan/Documents/Git/SDS-CP030-edu-spend/submissions/team-members/yan-cotta
source venv/bin/activate
```

### 3. Running the Notebook
After activating the environment:

**Start Jupyter Notebook:**
```bash
jupyter notebook
```

**Start JupyterLab:**
```bash
jupyter lab
```

Then open `01_EDA_EduSpend.ipynb` and select the "EduSpend Project" kernel.

## Installed Packages
- **Data Analysis:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Jupyter:** jupyter, ipykernel
- **Utilities:** openpyxl

## Project Phases

### Phase 1: Setup & EDA (Completed)
- [x] Project setup and virtual environment
- [x] Data loading and initial inspection
- [x] Cost distribution analysis
- [x] Correlation analysis
- [x] Total Cost of Attendance (TCA) calculation
- [x] Outlier detection

### Phase 2: Model Development (Completed)
- [x] Data preparation & feature engineering
- [x] Baseline regression model development
- [x] Model evaluation and refinement
- [x] Feature importance analysis
- [x] Cross-validation and model testing

## Key Findings from EDA

### Dataset Overview
- Total records: 907 education programs across multiple countries
- Features include country, city, university, program, tuition fees, living costs, and more
- No missing values identified in the dataset

### Cost Distribution Analysis
- Tuition fees vary significantly across countries, with the US, UK, and Australia showing higher average costs
- Higher degree levels (PhD, Masters) generally have higher tuition fees than Bachelors programs
- Living cost indices correlate strongly with rent prices across different cities

### Correlation Highlights
- Moderate positive correlation between tuition fees and living costs (r ≈ 0.45)
- Strong correlation between rent costs and total living expenses (r ≈ 0.82)
- Exchange rates show inverse relationship with some cost indicators

### Total Cost of Attendance (TCA)
- Successfully engineered TCA metric combining tuition, rent, insurance and visa fees
- Countries with highest TCA: USA, UK, Australia, Canada
- Programs with highest TCA: Medical, Business, Engineering

### Outliers
- Identified several high-cost outlier programs, primarily in medical and business fields
- Approximately 8% of tuition fees and 6% of rent costs classified as outliers
- Outliers were retained for modeling to maintain real-world cost variability

## Key Findings from Model Development

### Model Performance

- **Exceptional Accuracy**: R² score of 0.963, indicating the model explains 96.3% of variance in Total Cost of Attendance
- **Low Prediction Error**: Mean Absolute Error of $2,631.72 (8.89% of mean TCA)
- **Strong RMSE**: Root Mean Squared Error of $4,029.74, showing good handling of outliers
- **Robust Cross-Validation**: 5-fold cross-validation confirms consistent performance (R² = 0.963 ±0.0066)
- **Well-Balanced Errors**: Error distribution is centered around zero with no systematic bias

### Feature Importance

- **Rent Dominance**: Monthly rent cost is the most influential feature by far (approximately 60% importance)
- **Geographic Impact**: USA location significantly increases predicted costs (around 10% importance)
- **Living Cost Index**: General living costs provide additional predictive power (about 6% importance)
- **City Effects**: Simplified city grouping shows moderate importance in predictions
- **Program Factors**: Degree level and program duration have smaller but notable effects

### Model Visualization Insights

- **Excellent Fit**: Actual vs. predicted values show strong linear relationship along the ideal prediction line
- **Normal Error Distribution**: Prediction errors follow approximately normal distribution around zero
- **Consistent Performance**: Cross-validation metrics show minimal variation across different data subsets
- **Visual Feature Importance**: Clear visualization of the dominant effect of rent costs on TCA predictions

### Prediction Capabilities

- Successfully predicts TCA across different countries, cities, and program types
- Ability to estimate costs for new education scenarios not in the original dataset
- Works best for countries well-represented in the training data (some limitations with uncommon countries)
- Demo predictions show practical application for education planning scenarios

## Next Steps

1. Phase 3: Advanced Model Development & Interactive Tool
2. Hyperparameter tuning and alternative algorithm testing
3. Development of user-friendly cost prediction interface
4. Expanded analysis with more detailed program-specific factors

## Notes

- The notebook includes comprehensive error handling for missing data files
- All visualizations are designed to work with the expected dataset structure
- The TCA calculation adapts to available data columns

## Troubleshooting

If you encounter issues:

1. Ensure the virtual environment is activated
2. Verify the dataset is placed in the correct location
3. Check that all packages are installed: `pip list`
4. Restart the Jupyter kernel if needed

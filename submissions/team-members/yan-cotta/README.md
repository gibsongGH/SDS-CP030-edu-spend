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

- [x] Data preparation & feature engineering (including TCA calculation, handling categorical features, and scaling numerical features)
- [x] Baseline regression model development (Random Forest Regressor)
- [x] Model evaluation and refinement (achieved R² of ~0.82, MAE of ~$6,420)
- [x] Cross-validation performed to ensure model robustness (average R² ~0.81)
- [x] Feature importance analysis (identified `Rent_USD`, `Country_United States`, `Living_Cost_Index` as key predictors)
- [x] Created a sample prediction function for practical application.

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

## Key Findings from Model Development (Phase 2)

- **Model Performance**: Developed a Random Forest Regressor capable of predicting Total Cost of Attendance (TCA) with an R² score of approximately 0.82 on the test set. The Mean Absolute Error (MAE) was around $6,420.
- **Cross-Validation**: 5-fold cross-validation confirmed the model's stability, yielding an average R² of ~0.81, MAE of ~$6,550, and RMSE of ~$9,990.
- **Key Predictors**: The most influential features in predicting TCA include `Rent_USD`, `Country_United States`, `Living_Cost_Index`, `Country_Australia`, `Country_United Kingdom`, and degree levels (e.g., `Level_PhD`, `Level_Masters`).
- **Practical Application**: A function was developed to allow for sample TCA predictions based on user inputs, demonstrating the model's utility.
- **Data Handling**: Successfully loaded and prepared data, including a strategy for handling the high-cardinality 'City' feature by simplifying it.

## Next Steps

1. **Phase 3: Advanced Modeling & Deployment**
   - Explore hyperparameter tuning for the Random Forest model.
   - Experiment with other regression algorithms (e.g., Gradient Boosting, Neural Networks).
   - Develop an interactive web application or dashboard for users to get personalized cost estimates.
   - Consider incorporating more data sources or features to further enhance prediction accuracy.

## Notes

- The notebook includes comprehensive error handling for missing data files
- All visualizations are designed to work with the expected dataset structure
- The TCA calculation adapts to available data columns

## Troubleshooting

1. Ensure the virtual environment is activated
2. Verify the dataset is placed in the correct location
3. Check that all packages are installed: `pip list`
4. Restart the Jupyter kernel if needed

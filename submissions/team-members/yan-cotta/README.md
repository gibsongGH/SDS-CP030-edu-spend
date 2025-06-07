# EduSpend Project: yan-cotta's Submission

**Author:** yan-cotta  
**Phase:** 1 - Setup & EDA  
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
├── 01_EDA_EduSpend.ipynb      # Main EDA notebook
├── data/                       # Dataset folder
│   └── International_Education_Costs.csv  # [Place dataset here]
├── requirements.txt            # Project dependencies
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

## Phase 1 Goals
- [x] Project setup and virtual environment
- [x] Data loading and initial inspection
- [x] Cost distribution analysis
- [x] Correlation analysis
- [x] Total Cost of Attendance (TCA) calculation
- [x] Outlier detection

## Next Steps
1. Download and place the dataset in the `data/` folder
2. Activate the environment using one of the methods above
3. Open and run the EDA notebook
4. Analyze the results and proceed to Phase 2

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

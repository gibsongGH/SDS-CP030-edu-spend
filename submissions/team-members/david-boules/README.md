# International Education Costs Analysis

This project analyzes international education costs using data from Kaggle's 'cost_of_international_education.csv' dataset.

## Setup Instructions

1. Ensure Python 3.9+ is installed:
   ```bash
   python --version
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Analysis

1. Make sure your dataset (`data.csv`) is in the project root directory.

2. Run the analysis script:
   ```bash
   python education_cost_analysis.py
   ```

## Output

The script will generate:
- Three visualization plots in the `plots` directory:
  - `tuition_histogram.png`: Distribution of tuition fees
  - `rent_boxplot.png`: Box plot of rent by country
  - `correlation_heatmap.png`: Correlation heatmap of key metrics
- `cleaned_data.csv`: A cleaned version of the dataset

## Next Steps

The next iteration will focus on calculating the Total Cost of Attendance (TCA) by:
- Combining tuition and rent costs
- Incorporating living cost index
- Adding additional expense factors 
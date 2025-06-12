
# Cost of International Education Analysis

This project analyzes the cost of international education across various countries, cities, universities, and programs. The dataset includes tuition fees, living costs, rent, visa fees, and insurance costs, among other attributes.

## Key Features of the Analysis

1. **Exploratory Data Analysis (EDA):**
    - Profiled cost distributions by country, city, level, and program.
    - Visualized correlations among tuition fees, living cost index, rent, and other variables.
    - Computed baseline total cost of attendance (tuition + rent × months + visa + insurance × years).
    - Detected outliers and assessed currency-conversion stability.

2. **Derived Features:**
    - One-hot encoding for categorical variables.
    - Scaled and transformed numeric cost-related features.
    - Generated affordability-tier labels (Low/Medium/High) using quantile-based segmentation.

3. **Predictive Modeling:**
    - Developed a regression model to estimate the total cost of study based on program specifications and location attributes.
    - Built a classification model to categorize records into affordability tiers.

4. **Clustering:**
    - Applied K-Means and HDBSCAN clustering to group universities or destinations into cost archetypes.

5. **Evaluation & Tuning:**
    - Regression: Evaluated using MAE, RMSE, and R².
    - Classification: Evaluated using accuracy, macro F1, and ROC-AUC.
    - Clustering: Evaluated using silhouette score and qualitative review.
    - Performed hyperparameter tuning for tree-based and gradient-boosted models.

## Key Insights

- **Top Countries with Highest Costs:**
  - USA, Hong Kong, Singapore, Australia, and Canada lead in total costs.
- **Strong Correlations:**
  - Living cost index and rent are highly correlated.
  - Tuition fees moderately correlate with rent and insurance costs.
- **Affordability Tiers:**
  - Quantile-based segmentation provides insights into low, medium, and high-cost tiers.

## Files and Outputs

- **Plots:** Correlation matrix, cost distributions by categories.
- **Reports:** Summary statistics, top countries/universities/programs by cost metrics.
- **Models:** Regression and classification models for cost prediction and affordability classification.
- **Clustering:** Grouped universities and destinations into cost archetypes.

## Tools and Libraries

- Python: pandas, numpy, seaborn, matplotlib, scikit-learn, hdbscan.
- Data Source: Kaggle dataset on international education costs.

## How to Run

1. Clone the repository.
2. Install the required libraries using `pip install -r requirements.txt`.
3. Run the Jupyter Notebook to reproduce the analysis and results.

## Future Work

- Incorporate additional features like scholarships and financial aid.
- Explore advanced clustering techniques for better grouping.
- Develop a web-based dashboard for interactive exploration of the dataset.


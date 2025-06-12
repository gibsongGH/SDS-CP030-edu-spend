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
    - Regression model to estimate total cost of study based on program specifications and location attributes.
    - Classification model to categorize records into affordability tiers.

4. **Clustering:**
    - Applied K-Means and HDBSCAN clustering to group universities or destinations into cost archetypes.

5. **Evaluation & Tuning:**
    - Regression: Evaluated using MAE, RMSE, and R².
    - Classification: Evaluated using accuracy, macro F1, and ROC-AUC.
    - Clustering: Evaluated using silhouette score and qualitative review.
    - Hyperparameter tuning for tree-based and gradient-boosted models.

## Key Insights

- **Top Countries with Highest Costs:** USA, Hong Kong, Singapore, Australia, and Canada lead in total costs.
- **Strong Correlations:** Living cost index and rent are highly correlated. Tuition fees moderately correlate with rent and insurance costs.
- **Affordability Tiers:** Quantile-based segmentation provides insights into low, medium, and high-cost tiers.

## Tools and Libraries

- Python: pandas, numpy, seaborn, matplotlib, scikit-learn, hdbscan.
- Data Source: Kaggle dataset on international education costs.

## Future Work

- Incorporate additional features like scholarships and financial aid.
- Explore advanced clustering techniques for better grouping.
- Develop a web-based dashboard for interactive exploration of the dataset.

## Conclusion

This analysis provides a comprehensive overview of the costs associated with international education, highlighting key trends and insights that can help prospective students make informed decisions about their study abroad plans.

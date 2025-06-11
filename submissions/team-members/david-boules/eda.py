#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Create plots directory if it doesn't exist
Path("plots").mkdir(exist_ok=True)

# Read the dataset
print("Loading the dataset...")
df = pd.read_csv('data.csv')

# Fix city-university swaps
print("\nChecking and fixing city-university swaps...")
# Known swap: MIT and Massachusetts
mask_mit = df['City'] == 'MIT'
if any(mask_mit):
    df.loc[mask_mit, ['City', 'University']] = df.loc[mask_mit, ['University', 'City']].values

# Look for other potential swaps where university name appears in city column
print("Potential city-university inconsistencies:")
for idx, row in df.iterrows():
    city = str(row['City']).lower()
    univ = str(row['University']).lower()
    
    # Check if 'university' or 'college' appears in city name
    if 'university' in city or 'college' in city:
        print(f"Row {idx}: City: {row['City']}, University: {row['University']}")
    # Check if city name appears in university name but university field doesn't contain city name
    elif city in univ and city not in ['new', 'los', 'san', 'hong']:  # exclude common city name parts
        print(f"Row {idx}: City: {row['City']}, University: {row['University']}")

# Display first 5 rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Calculate Total Cost of Attendance (TCA)
print("\nCalculating Total Cost of Attendance...")
# Convert Duration_Years to numeric if not already
df['Duration_Years'] = pd.to_numeric(df['Duration_Years'])

# Calculate TCA: (tuition + rent × 12 months × years + visa + insurance × years)
df['TCA_USD'] = (
    df['Tuition_USD'] +
    (df['Rent_USD'] * 12 * df['Duration_Years']) +
    df['Visa_Fee_USD'] +
    (df['Insurance_USD'] * df['Duration_Years'])
)

# Basic statistics for key columns including TCA
key_columns = ['Tuition_USD', 'Rent_USD', 'Living_Cost_Index', 'Exchange_Rate']
print("\nBasic statistics for key columns:")
print(df[key_columns + ['TCA_USD']].describe())

# Create visualizations
print("\nGenerating plots...")

# 1. Correlation heatmap (excluding TCA)
plt.figure(figsize=(10, 8))
correlation_matrix = df[key_columns].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('plots/correlation_heatmap.png')
plt.close()

# 2. TCA Distribution by Country (all countries)
plt.figure(figsize=(20, 10))
sns.boxplot(x='Country', y='TCA_USD', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Total Cost of Attendance Distribution by Country')
plt.ylabel('Total Cost of Attendance (USD)')
plt.tight_layout()
plt.savefig('plots/tca_by_country.png')
plt.close()

# 3. TCA Distribution by Level
plt.figure(figsize=(12, 6))
sns.boxplot(x='Level', y='TCA_USD', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Total Cost of Attendance Distribution by Education Level')
plt.ylabel('Total Cost of Attendance (USD)')
plt.tight_layout()
plt.savefig('plots/tca_by_level.png')
plt.close()

# 4. TCA Distribution by Program (all programs)
plt.figure(figsize=(20, 10))
sns.boxplot(x='Program', y='TCA_USD', data=df)
plt.xticks(rotation=45, ha='right')
plt.title('Total Cost of Attendance Distribution by Program')
plt.ylabel('Total Cost of Attendance (USD)')
plt.tight_layout()
plt.savefig('plots/tca_by_program.png')
plt.close()

# 5. Top 15 Most Expensive Cities (by median TCA)
plt.figure(figsize=(15, 8))
top_15_cities = df.groupby('City')['TCA_USD'].agg(['median', 'count']).sort_values('median', ascending=False).head(15)
top_15_cities['median'].plot(kind='bar')
plt.title('Top 15 Most Expensive Cities by Total Cost of Attendance')
plt.xlabel('City')
plt.ylabel('Median Total Cost of Attendance (USD)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/top_15_expensive_cities.png')
plt.close()

# 6. TCA Distribution Histogram
plt.figure(figsize=(12, 6))
plt.hist(df['TCA_USD'], bins=50, edgecolor='black')
plt.title('Distribution of Total Cost of Attendance')
plt.xlabel('Total Cost of Attendance (USD)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('plots/tca_histogram.png')
plt.close()

# Save summary statistics to a file
print("\nGenerating summary statistics...")
with open('summary_statistics.txt', 'w') as f:
    f.write("=== Summary Statistics ===\n\n")
    
    # Overall TCA statistics
    f.write("Overall TCA Statistics:\n")
    f.write(df['TCA_USD'].describe().to_string())
    f.write("\n\n")
    
    # TCA by Level with counts
    level_stats = df.groupby('Level').agg({
        'TCA_USD': ['median', 'count']
    }).round(1)
    f.write("TCA by Education Level:\n")
    f.write("Level | Median TCA ($) | Number of Programs\n")
    f.write("-" * 45 + "\n")
    for idx, row in level_stats.iterrows():
        f.write(f"{idx:<20} {row[('TCA_USD', 'median')]:>10.1f} {row[('TCA_USD', 'count')]:>15.0f}\n")
    f.write("\n\n")
    
    # Top 10 most expensive countries with counts
    country_stats = df.groupby('Country').agg({
        'TCA_USD': ['median', 'count']
    }).round(1).sort_values(('TCA_USD', 'median'), ascending=False).head(10)
    f.write("Top 10 Most Expensive Countries:\n")
    f.write("Country | Median TCA ($) | Number of Programs\n")
    f.write("-" * 45 + "\n")
    for idx, row in country_stats.iterrows():
        f.write(f"{idx:<20} {row[('TCA_USD', 'median')]:>10.1f} {row[('TCA_USD', 'count')]:>15.0f}\n")
    f.write("\n\n")
    
    # Top 10 most expensive programs with counts
    program_stats = df.groupby('Program').agg({
        'TCA_USD': ['median', 'count']
    }).round(1).sort_values(('TCA_USD', 'median'), ascending=False).head(10)
    f.write("Top 10 Most Expensive Programs:\n")
    f.write("Program | Median TCA ($) | Number of Programs\n")
    f.write("-" * 45 + "\n")
    for idx, row in program_stats.iterrows():
        f.write(f"{idx:<30} {row[('TCA_USD', 'median')]:>10.1f} {row[('TCA_USD', 'count')]:>15.0f}\n")

# Save cleaned data with TCA
print("\nSaving cleaned data with TCA calculations...")
df.to_csv('cleaned_data_with_tca.csv', index=False)

print("\nAnalysis complete! Files saved:")
print("- Visualizations in the 'plots' directory")
print("- Summary statistics in 'summary_statistics.txt'")
print("- Enhanced dataset in 'cleaned_data_with_tca.csv'")

print("\nNext Steps for Analysis:")
print("1. Outlier Detection:")
print("   - Use IQR method or z-score to identify statistical outliers")
print("   - Investigate extreme TCA values by country/program")
print("2. Currency Conversion Stability:")
print("   - Analyze historical exchange rate variations")
print("   - Calculate TCA sensitivity to exchange rate fluctuations")
print("3. Cost-Benefit Analysis:")
print("   - Compare TCA with country-specific income potential")
print("   - Consider program duration vs. total cost trade-offs") 
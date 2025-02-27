import pandas as pd
import numpy as np

# Read vacancy rate data
df = pd.read_csv('PCA Raw Data/Vacancy_Rates_Raw(RHVRUSQ156N).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include 1995 through 2023 (need 1995 for calculating 1996Q1 changes)
mask = (df['DATE'] >= '1995-01-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Calculate quarter-over-quarter absolute changes
df['VACANCY_CHANGE'] = df['RHVRUSQ156N'].diff()

# Now filter to start from 1996Q1
df = df[df['DATE'] >= '1996-01-01']

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': df['DATE'],
    'VACANCY_CHANGE': df['VACANCY_CHANGE']
})

# Filter to Q1 1996 to Q4 2023
output = output[output['DATE'] >= pd.to_datetime('1996-01-01')]

# Scale by maximum absolute value to preserve signs
vacancy_maxabs = output['VACANCY_CHANGE'] / max(abs(output['VACANCY_CHANGE']))
output['VACANCY_NORMALIZED'] = vacancy_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['VACANCY_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/vacancy_normalized.csv', index=False)

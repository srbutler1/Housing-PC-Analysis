import pandas as pd
import numpy as np

# Read population data
df = pd.read_csv('PCA Raw Data/US_Pop(POPTHM).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include 1995 through 2023 (need 1995 for calculating 1996Q1 changes)
mask = (df['DATE'] >= '1995-01-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Calculate quarter-over-quarter percent changes
df['POP_CHANGE'] = df['POPTHM'].pct_change() * 100

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': df['DATE'][df['DATE'] >= '1996-01-01'],
    'POP_CHANGE': df['POP_CHANGE'][df['DATE'] >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
pop_maxabs = output['POP_CHANGE'] / max(abs(output['POP_CHANGE']))
output['POP_NORMALIZED'] = pop_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['POP_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/population_normalized.csv', index=False)

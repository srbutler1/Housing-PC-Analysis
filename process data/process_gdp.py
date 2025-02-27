import pandas as pd
import numpy as np

# Read GDP data
df = pd.read_csv('PCA Raw Data/GDP(GDP).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include Q4 1995 through Q4 2023
mask = (df['DATE'] >= '1995-10-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Calculate quarter-over-quarter percent changes
df['GDP_CHANGE'] = df['GDP'].pct_change() * 100

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': df['DATE'][df['DATE'] >= '1996-01-01'],
    'GDP_CHANGE': df['GDP_CHANGE'][df['DATE'] >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
gdp_maxabs = output['GDP_CHANGE'] / max(abs(output['GDP_CHANGE']))
output['GDP_NORMALIZED'] = gdp_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['GDP_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/gdp_normalized.csv', index=False)

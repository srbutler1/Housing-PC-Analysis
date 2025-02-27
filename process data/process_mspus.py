import pandas as pd
import numpy as np

# Read MSPUS data
df = pd.read_csv('PCA Raw Data/Median_sales_price(MSPUS).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include 1995 through 2023 (need 1995 for calculating 1996Q1 changes)
mask = (df['DATE'] >= '1995-01-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Calculate quarter-over-quarter percent changes
df['MSPUS_CHANGE'] = df['MSPUS'].pct_change() * 100

# Now filter to start from 1996Q1
df = df[df['DATE'] >= '1996-01-01']

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': df['DATE'],
    'MSPUS_CHANGE': df['MSPUS_CHANGE']
})

# Scale by maximum absolute value to preserve signs
mspus_maxabs = output['MSPUS_CHANGE'] / max(abs(output['MSPUS_CHANGE']))
output['MSPUS_NORMALIZED'] = mspus_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['MSPUS_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/mspus_normalized.csv', index=False)

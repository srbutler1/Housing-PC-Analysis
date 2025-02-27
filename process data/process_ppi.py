import pandas as pd
import numpy as np

# Read PPI data
df = pd.read_csv('PCA Raw Data/PPI_res_construction(WPUIP2311001).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include 1995 through 2023 (need 1995 for calculating 1996Q1 changes)
mask = (df['DATE'] >= '1995-01-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Calculate month-over-month percent changes
df['PPI_CHANGE'] = df['WPUIP2311001'].pct_change() * 100

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': df['DATE'][df['DATE'] >= '1996-01-01'],
    'PPI_CHANGE': df['PPI_CHANGE'][df['DATE'] >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
ppi_maxabs = output['PPI_CHANGE'] / max(abs(output['PPI_CHANGE']))
output['PPI_NORMALIZED'] = ppi_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['PPI_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/ppi_normalized.csv', index=False)

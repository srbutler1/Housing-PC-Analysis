import pandas as pd
import numpy as np

# Read PRFI data
df = pd.read_csv('PCA Raw Data/Residential_investment(PRFI).csv')

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['observation_date'])

# Filter to include Q4 1995 through Q4 2023
mask = (df['DATE'] >= '1995-10-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Set date as index
df.set_index('DATE', inplace=True)

# Calculate quarter-over-quarter percent changes
df['PRFI_CHANGE'] = df['PRFI'].pct_change() * 100

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': df.index[df.index >= '1996-01-01'],
    'PRFI_CHANGE': df['PRFI_CHANGE'][df.index >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
prfi_maxabs = output['PRFI_CHANGE'] / max(abs(output['PRFI_CHANGE']))
output['PRFI_NORMALIZED'] = prfi_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['PRFI_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/prfi_normalized.csv', index=False)

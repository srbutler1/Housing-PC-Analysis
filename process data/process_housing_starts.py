import pandas as pd
import numpy as np

# Read housing starts data
df = pd.read_csv('PCA Raw Data/Single_family_starts(HOUST1F).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include 1995 through 2023 (need 1995 for calculating 1996Q1 changes)
mask = (df['DATE'] >= '1995-01-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Create quarter start dates
df['Quarter'] = df['DATE'].dt.to_period('Q')
quarterly_df = df.groupby('Quarter')['HOUST1F'].mean().reset_index()

# Set date to start of quarter (e.g., 2023Q1 becomes 2023-01-01)
quarterly_df['DATE'] = quarterly_df['Quarter'].astype(str).apply(lambda x: pd.to_datetime(x.replace('Q1', '-01-01').replace('Q2', '-04-01').replace('Q3', '-07-01').replace('Q4', '-10-01')))

# Calculate quarter-over-quarter percent changes
quarterly_df['HOUSING_CHANGE'] = quarterly_df['HOUST1F'].pct_change() * 100

# Now filter to start from 1996Q1
quarterly_df = quarterly_df[quarterly_df['DATE'] >= '1996-01-01']

# Create output dataframe
output = pd.DataFrame({
    'DATE': quarterly_df['DATE'],
    'HOUSING_CHANGE': quarterly_df['HOUSING_CHANGE']
})

# Scale by maximum absolute value to preserve signs
housing_maxabs = output['HOUSING_CHANGE'] / max(abs(output['HOUSING_CHANGE']))
output['HOUSING_NORMALIZED'] = housing_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['HOUSING_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/housing_starts_normalized.csv', index=False)

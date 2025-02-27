import pandas as pd
import numpy as np

# Read PCE data
df = pd.read_csv('PCA Raw Data/Personal_consumption_exp(PCE).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include Q4 1995 through Q4 2023
mask = (df['DATE'] >= '1995-10-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Convert monthly to quarterly by summing the values for each quarter
df['Quarter'] = df['DATE'].dt.to_period('Q')
quarterly_df = df.groupby('Quarter')['PCE'].sum().reset_index()
quarterly_df['DATE'] = quarterly_df['Quarter'].astype(str).apply(lambda x: pd.to_datetime(x.replace('Q', '-')))
quarterly_df = quarterly_df.drop('Quarter', axis=1)

# Calculate quarter-over-quarter percent changes
quarterly_df['PCE_CHANGE'] = quarterly_df['PCE'].pct_change() * 100

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': quarterly_df['DATE'][quarterly_df['DATE'] >= '1996-01-01'],
    'PCE_CHANGE': quarterly_df['PCE_CHANGE'][quarterly_df['DATE'] >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
pce_maxabs = output['PCE_CHANGE'] / max(abs(output['PCE_CHANGE']))
output['PCE_NORMALIZED'] = pce_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['PCE_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/pce_normalized.csv', index=False)

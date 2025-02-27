import pandas as pd
import numpy as np

# Read CPI data (already in growth rate form)
df = pd.read_csv('PCA Raw Data/CPI_all_growth_rate(CPALTT01USM657N).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include Q1 1996 through Q4 2023
mask = (df['DATE'] >= '1996-01-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Create quarter start dates
df['Quarter'] = df['DATE'].dt.to_period('Q')
quarterly_df = df.groupby('Quarter')['CPALTT01USM657N'].mean().reset_index()

# Set date to start of quarter (e.g., 2023Q1 becomes 2023-01-01)
quarterly_df['DATE'] = quarterly_df['Quarter'].astype(str).apply(lambda x: pd.to_datetime(x.replace('Q1', '-01-01').replace('Q2', '-04-01').replace('Q3', '-07-01').replace('Q4', '-10-01')))

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': quarterly_df['DATE'][quarterly_df['DATE'] >= '1996-01-01'],
    'CPI_CHANGE': quarterly_df['CPALTT01USM657N'][quarterly_df['DATE'] >= '1996-01-01']  # Already growth rate
})

# Scale by maximum absolute value to preserve signs
cpi_maxabs = output['CPI_CHANGE'] / max(abs(output['CPI_CHANGE']))
output['CPI_NORMALIZED'] = cpi_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['CPI_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/cpi_normalized.csv', index=False)

import pandas as pd
import numpy as np

# Read unemployment rate data
df = pd.read_csv('PCA Raw Data/Unemployment_rate(UNRATE).csv')

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['observation_date'])

# Filter to include Q4 1995 through Q4 2023
mask = (df['DATE'] >= '1995-10-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Convert monthly to quarterly by taking the average for each quarter
df['Quarter'] = df['DATE'].dt.to_period('Q')
quarterly_df = df.groupby('Quarter')['UNRATE'].mean().reset_index()
# Convert Quarter periods to dates at start of quarter
quarterly_df['DATE'] = quarterly_df['Quarter'].astype(str).apply(lambda x: pd.to_datetime(x.replace('Q1', '-01-01').replace('Q2', '-04-01').replace('Q3', '-07-01').replace('Q4', '-10-01')))
quarterly_df = quarterly_df.drop('Quarter', axis=1)

# Calculate quarter-over-quarter changes
quarterly_df['UNEMPLOYMENT_CHANGE'] = quarterly_df['UNRATE'].diff()

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': quarterly_df['DATE'][quarterly_df['DATE'] >= '1996-01-01'],
    'UNEMPLOYMENT_CHANGE': quarterly_df['UNEMPLOYMENT_CHANGE'][quarterly_df['DATE'] >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
unemployment_maxabs = output['UNEMPLOYMENT_CHANGE'] / max(abs(output['UNEMPLOYMENT_CHANGE']))
output['UNEMPLOYMENT_NORMALIZED'] = unemployment_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['UNEMPLOYMENT_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/unemployment_normalized.csv', index=False)

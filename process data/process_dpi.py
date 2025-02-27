import pandas as pd
import numpy as np

# Read DPI data
df = pd.read_csv('PCA Raw Data/Disposable_personal_income(DPIC96).csv')

# Rename observation_date to DATE for consistency
df = df.rename(columns={'observation_date': 'DATE'})

# Convert date to datetime
df['DATE'] = pd.to_datetime(df['DATE'])

# Filter to include Q4 1995 through Q4 2023
mask = (df['DATE'] >= '1995-10-01') & (df['DATE'] <= '2023-12-31')
df = df[mask]

# Convert monthly to quarterly by summing the values for each quarter
df['Quarter'] = df['DATE'].dt.to_period('Q')
quarterly_df = df.groupby('Quarter')['DPIC96'].sum().reset_index()
# Convert Quarter periods to start-of-quarter dates
quarterly_df['DATE'] = quarterly_df['Quarter'].apply(lambda x: pd.Period(x).start_time.normalize())
quarterly_df = quarterly_df.drop('Quarter', axis=1)

# Calculate quarter-over-quarter percent changes
quarterly_df['DPI_CHANGE'] = quarterly_df['DPIC96'].pct_change() * 100

# Create output dataframe with date and percent change
output = pd.DataFrame({
    'DATE': quarterly_df['DATE'][quarterly_df['DATE'] >= '1996-01-01'],
    'DPI_CHANGE': quarterly_df['DPI_CHANGE'][quarterly_df['DATE'] >= '1996-01-01']
})

# Scale by maximum absolute value to preserve signs
dpi_maxabs = output['DPI_CHANGE'] / max(abs(output['DPI_CHANGE']))
output['DPI_NORMALIZED'] = dpi_maxabs

print("First few rows of processed data:")
print(output.head())
print("\nDescriptive statistics:")
print(output['DPI_NORMALIZED'].describe())
print("\nShape of data:", output.shape)

# Save to cleaned data folder
output.to_csv('PCA Cleaned Data/dpi_normalized.csv', index=False)

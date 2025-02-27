import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('MORTGAGE30US (7).csv')

# Convert observation_date to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Create a new dataframe with quarterly dates and calculate mean for each quarter
quarterly_df = df.set_index('observation_date').resample('Q')['MORTGAGE30US'].mean().reset_index()

# Calculate percent change between quarters
quarterly_df['pct_change'] = quarterly_df['MORTGAGE30US'].pct_change() * 100

# Round values to 2 decimal places
quarterly_df['MORTGAGE30US'] = quarterly_df['MORTGAGE30US'].round(2)
quarterly_df['pct_change'] = quarterly_df['pct_change'].round(2)

# Convert dates to quarter format (e.g., "1995Q2")
quarterly_df['observation_date'] = quarterly_df['observation_date'].dt.to_period('Q').astype(str)

# Drop the first row since it will have NaN percent change
quarterly_df = quarterly_df.dropna()

# Save to new CSV
quarterly_df.to_csv('MORTGAGE30US_quarterly.csv', index=False)

# Print validation information
print("Conversion complete. Data saved to MORTGAGE30US_quarterly.csv")
print(f"Original number of rows: {len(df)}")
print(f"Quarterly rows: {len(quarterly_df)}")
print("\nFirst few rows of quarterly data with percent changes:")
print(quarterly_df.head().to_string())
print("\nLast few rows of quarterly data with percent changes:")
print(quarterly_df.tail().to_string())
print("\nChecking for missing values:", quarterly_df.isna().sum().to_string())

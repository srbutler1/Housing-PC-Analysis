import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('MEDCPIM158SFRBCLE.csv')

# Convert observation_date to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Create a new dataframe with quarterly dates and calculate mean for each quarter
quarterly_df = df.set_index('observation_date').resample('Q')['MEDCPIM158SFRBCLE'].mean().reset_index()

# Round values to 2 decimal places
quarterly_df['MEDCPIM158SFRBCLE'] = quarterly_df['MEDCPIM158SFRBCLE'].round(2)

# Convert dates to quarter format (e.g., "1995Q2")
quarterly_df['observation_date'] = quarterly_df['observation_date'].dt.to_period('Q').astype(str)

# Save to new CSV
quarterly_df.to_csv('MEDCPIM158SFRBCLE_quarterly.csv', index=False)

# Print validation information
print("Conversion complete. Data saved to MEDCPIM158SFRBCLE_quarterly.csv")
print(f"Original number of rows: {len(df)}")
print(f"Quarterly rows: {len(quarterly_df)}")
print("\nFirst few rows of quarterly data:")
print(quarterly_df.head().to_string())
print("\nLast few rows of quarterly data:")
print(quarterly_df.tail().to_string())
print("\nChecking for missing values:", quarterly_df.isna().sum().to_string())

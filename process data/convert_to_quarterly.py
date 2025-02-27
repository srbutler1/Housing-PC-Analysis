import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('MEHOINUSA672N (2).csv')

# Convert observation_date to datetime
df['observation_date'] = pd.to_datetime(df['observation_date'])

# Map January 1st dates to Q1 dates (end of March)
df['observation_date'] = df['observation_date'].apply(
    lambda x: pd.Timestamp(year=x.year, month=3, day=31) if x.month == 1 else x
)

# Create quarterly dates from start to end
quarterly_dates = pd.date_range(
    start=df['observation_date'].min(),
    end=df['observation_date'].max(),
    freq='Q'
)

# Create a new dataframe with quarterly dates
quarterly_df = pd.DataFrame(index=quarterly_dates)

# Assign the annual values to Q1 of each year
for date, value in zip(df['observation_date'], df['MEHOINUSA672N']):
    quarterly_df.loc[date, 'MEHOINUSA672N'] = value

# Interpolate missing values
quarterly_df['MEHOINUSA672N'] = quarterly_df['MEHOINUSA672N'].interpolate(method='linear')

# Calculate percent change between quarters
quarterly_df['pct_change'] = quarterly_df['MEHOINUSA672N'].pct_change() * 100

# Round values to 2 decimal places
quarterly_df['MEHOINUSA672N'] = quarterly_df['MEHOINUSA672N'].round(2)
quarterly_df['pct_change'] = quarterly_df['pct_change'].round(2)

# Reset index to make date a column
quarterly_df.reset_index(inplace=True)
quarterly_df.rename(columns={'index': 'observation_date'}, inplace=True)

# Convert dates to quarter format (e.g., "1995Q2")
quarterly_df['observation_date'] = quarterly_df['observation_date'].dt.to_period('Q').astype(str)

# Drop the first row since it will have NaN percent change
quarterly_df = quarterly_df.dropna()

# Save to new CSV
quarterly_df.to_csv('MEHOINUSA672N_quarterly.csv', index=False)

# Print validation information
print("Conversion complete. Data saved to MEHOINUSA672N_quarterly.csv")
print(f"Original number of rows: {len(df)}")
print(f"Quarterly interpolated rows: {len(quarterly_df)}")
print("\nFirst few rows of interpolated data with percent changes:")
print(quarterly_df.head().to_string())
print("\nLast few rows of interpolated data with percent changes:")
print(quarterly_df.tail().to_string())
print("\nChecking for missing values:", quarterly_df.isna().sum().to_string())

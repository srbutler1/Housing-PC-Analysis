import pandas as pd

# Load the CSV file
file_path = '/Users/appleowner/Downloads/Thesis/Data/PC Analysis /UNRATE.csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Check column names to ensure observation_date exists
print(data.columns)

# Convert the observation_date column to datetime
data['observation_date'] = pd.to_datetime(data['observation_date'])

# Set the observation_date column as the index
data.set_index('observation_date', inplace=True)

# Resample the data to quarterly frequency and calculate the sum for each quarter
data_quarterly = data.resample('Q').mean()

# Add the quarterly format column (e.g., 'Q11996')
data_quarterly['Quarterly_Format'] = data_quarterly.index.to_period('Q').astype(str).str.replace('-', 'Q')


# Save the quarterly data to a new CSV file
output_path = '/Users/appleowner/Downloads/Thesis/Data/PC Analysis /UnemRrate_quarterly_average.csv'  
data_quarterly.to_csv(output_path, index=False)

print("Quarterly totals have been saved to:", output_path)


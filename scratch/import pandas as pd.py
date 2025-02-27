import pandas as pd

# Load the CSV file
file_path = '/Users/appleowner/Downloads/Thesis/Data/PC Analysis /GDP (2).csv'  # Replace with your file path
data = pd.read_csv(file_path)

# Convert the observation_date column to datetime
data['observation_date'] = pd.to_datetime(data['observation_date'])

# Convert to quarterly format (e.g., 'Q11996')
data['Quarterly_Format'] = data['observation_date'].dt.to_period('Q').astype(str).str.replace('-', 'Q')

# Drop the original observation_date column if not needed
data = data[['Quarterly_Format', 'GDP']]


# Save the modified data to a new CSV file
output_path = '/Users/appleowner/Downloads/Thesis/Data/PC Analysis /GDP_quarterly.csv'  # Replace with your desired output path
data.to_csv(output_path, index=False)

print(f"Quarterly formatted data saved to: {output_path}")

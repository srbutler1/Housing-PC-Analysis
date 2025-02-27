import pandas as pd
import os
from datetime import datetime

def process_hai_data():
    """
    Process HAI (Housing Affordability Index) data without normalization.
    The data will be aligned with the quarterly format used by independent variables.
    """
    # Read HAI data from the cleaned data directory
    hai_df = pd.read_csv('PCA Cleaned Data/HAI_Median_Income.csv')
    
    # Convert quarter notation to date format if not already in date format
    if 'quarter' in hai_df.columns:
        def quarter_to_date(quarter):
            year = int(quarter[:4])
            q = int(quarter[5])
            month = (q - 1) * 3 + 1
            return f"{year}-{month:02d}-01"
        
        hai_df['DATE'] = hai_df['quarter'].apply(quarter_to_date)
    
    # Ensure DATE is in datetime format
    if 'DATE' in hai_df.columns:
        hai_df['DATE'] = pd.to_datetime(hai_df['DATE'])
        hai_df = hai_df.set_index('DATE')
    
    # Rename column for consistency if needed
    if 'Home Price Affordability Median HHI' in hai_df.columns:
        hai_df = hai_df.rename(columns={'Home Price Affordability Median HHI': 'HAI'})
    
    # Sort by date to ensure proper ordering
    hai_df = hai_df.sort_index()
    
    # Save processed data with _processed suffix to distinguish from original
    output_path = os.path.join('PCA Cleaned Data', 'hai_processed.csv')
    hai_df.to_csv(output_path)
    
    print(f"Processed HAI data saved to {output_path}")
    return hai_df

if __name__ == "__main__":
    hai_df = process_hai_data()
    print("\nHAI Data Summary:")
    print(hai_df.describe())
    print("\nDate Range:")
    print(f"Start: {hai_df.index.min()}")
    print(f"End: {hai_df.index.max()}")
    print("\nFirst few rows:")
    print(hai_df.head())
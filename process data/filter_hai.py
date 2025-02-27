import pandas as pd

def filter_hai_data():
    """
    Filter HAI data to include only 1996-2023 and remove the quarter column.
    """
    # Read the processed HAI data
    hai_df = pd.read_csv('PCA Cleaned Data/hai_processed.csv')
    
    # Convert DATE to datetime
    hai_df['DATE'] = pd.to_datetime(hai_df['DATE'])
    
    # Filter for 1996-2023
    mask = (hai_df['DATE'] >= '1996-01-01') & (hai_df['DATE'] <= '2023-12-31')
    hai_df = hai_df[mask]
    
    # Set DATE as index and drop quarter column
    hai_df = hai_df.set_index('DATE')
    hai_df = hai_df.drop('quarter', axis=1)
    
    # Save filtered data
    output_path = 'PCA Cleaned Data/hai_1996_2023.csv'
    hai_df.to_csv(output_path)
    
    print(f"Filtered HAI data saved to {output_path}")
    return hai_df

if __name__ == "__main__":
    hai_df = filter_hai_data()
    print("\nHAI Data Summary:")
    print(hai_df.describe())
    print("\nDate Range:")
    print(f"Start: {hai_df.index.min()}")
    print(f"End: {hai_df.index.max()}")
    print("\nFirst few rows:")
    print(hai_df.head())

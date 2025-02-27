import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Group variables according to the paper's methodology
supply_factors = {
    'Housing Starts': 'PCA Cleaned Data/housing_starts_normalized.csv',
    'House Prices': 'PCA Cleaned Data/mspus_normalized.csv',
    'PRFI': 'PCA Cleaned Data/prfi_normalized.csv',  # Private Residential Fixed Investment
    'PPI': 'PCA Cleaned Data/ppi_normalized.csv',    # Producer Price Index
}

demand_factors = {
    'Population': 'PCA Cleaned Data/population_normalized.csv',
    'DPI': 'PCA Cleaned Data/dpi_normalized.csv',    # Disposable Personal Income
    'Mortgage Rate': 'PCA Cleaned Data/mortgage_normalized.csv',
    'GDP': 'PCA Cleaned Data/gdp_normalized.csv',
}

market_environment = {
    'Unemployment': 'PCA Cleaned Data/unemployment_normalized.csv',
    'CPI': 'PCA Cleaned Data/cpi_normalized.csv',    # Consumer Price Index
}

# Function to load and process data for a group of factors
def load_factor_group(factor_dict):
    dfs = []
    for name, file in factor_dict.items():
        df = pd.read_csv(file)
        df.set_index('DATE', inplace=True)
        normalized_col = [col for col in df.columns if '_NORMALIZED' in col][0]
        df = df[[normalized_col]]
        df.columns = [name]
        dfs.append(df)
    return pd.concat(dfs, axis=1)

# Load data for each factor group
supply_df = load_factor_group(supply_factors)
demand_df = load_factor_group(demand_factors)
market_df = load_factor_group(market_environment)

def get_components_for_threshold(explained_variance_ratio, threshold=0.90):
    """Determine number of components needed to explain threshold% of variance"""
    cumulative_variance = np.cumsum(explained_variance_ratio)
    n_components = np.argmax(cumulative_variance >= threshold) + 1
    return n_components

# Function to perform PCA on a factor group
def perform_group_pca(data, group_name, period_name, output_dir):
    # Perform PCA
    pca = PCA()
    pca_result = pca.fit_transform(data)
    
    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # Determine number of components needed for 90% variance
    n_components = get_components_for_threshold(explained_variance_ratio)
    
    # Save component composition
    component_composition = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(pca_result.shape[1])],
        index=data.columns
    )
    component_composition.to_csv(f'{output_dir}/component_composition_{group_name}_{period_name}.csv')
    
    # Create component composition heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(component_composition.iloc[:, :n_components], 
                annot=True, cmap='coolwarm', center=0, fmt='.3f')
    plt.title(f'Principal Components Composition - {group_name}\n{period_name}\n'
             f'(First {n_components} PCs explain {cumulative_variance_ratio[n_components-1]:.1%} of variance)')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/component_composition_{group_name}_{period_name}.png')
    plt.close()
    
    # Create scree plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(explained_variance_ratio) + 1), 
            cumulative_variance_ratio, 'bo-')
    plt.axhline(y=0.9, color='r', linestyle='--', label='90% Threshold')
    plt.axvline(x=n_components, color='g', linestyle='--', 
                label=f'{n_components} PCs Selected')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title(f'Scree Plot - {group_name}\n{period_name}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/explained_variance_{group_name}_{period_name}.png')
    plt.close()
    
    return pca_result, explained_variance_ratio, cumulative_variance_ratio, component_composition, n_components

# Ensure all DataFrames have the same index
common_index = supply_df.index.intersection(demand_df.index).intersection(market_df.index)
supply_df = supply_df.loc[common_index]
demand_df = demand_df.loc[common_index]
market_df = market_df.loc[common_index]

# Convert index to datetime
supply_df.index = pd.to_datetime(supply_df.index)
demand_df.index = pd.to_datetime(demand_df.index)
market_df.index = pd.to_datetime(market_df.index)

# Sort by date
supply_df = supply_df.sort_index()
demand_df = demand_df.sort_index()
market_df = market_df.sort_index()

# Drop any rows with missing values
supply_df = supply_df.dropna()
demand_df = demand_df.dropna()
market_df = market_df.dropna()

# Define time periods (4-year periods and 2020-2023)
start_year = supply_df.index.min().year
end_year = supply_df.index.max().year

# Create periods list
periods = []
current_year = start_year
while current_year < 2020:  # Up to 2020, use 4-year periods
    period_start = f"{current_year}-01-01"
    period_end = f"{current_year + 3}-12-31"
    periods.append((period_start, period_end))
    current_year += 4

# Add the 2020-2023 period
periods.append(("2020-01-01", "2023-12-31"))

# Function to perform PCA for each group in a period
def perform_period_analysis(supply_data, demand_data, market_data, period_name, output_dir):
    # Perform PCA for each group
    supply_results = perform_group_pca(supply_data, 'Supply', period_name, output_dir)
    demand_results = perform_group_pca(demand_data, 'Demand', period_name, output_dir)
    market_results = perform_group_pca(market_data, 'Market', period_name, output_dir)
    
    # Create summary for this period
    summary = f"\nPeriod: {period_name}\n"
    summary += "=" * (len(period_name) + 8) + "\n\n"
    
    for group_name, results in [
        ('Supply Factors', supply_results),
        ('Demand Factors', demand_results),
        ('Market Environment', market_results)
    ]:
        _, explained_variance_ratio, cumulative_variance_ratio, composition, n_components = results
        
        summary += f"{group_name}:\n"
        summary += "-" * len(group_name) + "\n"
        summary += f"Number of components needed for 90% variance: {n_components}\n"
        summary += "Explained Variance Ratio:\n"
        for i, var in enumerate(explained_variance_ratio):
            summary += f"PC{i+1}: {var:.4f} ({cumulative_variance_ratio[i]:.4f} cumulative)\n"
        
        summary += "\nSignificant Component Loadings (>= |0.3|):\n"
        for pc_idx in range(n_components):
            pc_loadings = []
            for var_idx, loading in enumerate(composition.iloc[:, pc_idx]):
                if abs(loading) >= 0.3:
                    var_name = composition.index[var_idx]
                    pc_loadings.append(f"{var_name} ({loading:.3f})")
            
            if pc_loadings:
                summary += f"\nPC{pc_idx + 1}:\n"
                for loading in pc_loadings:
                    summary += f"  {loading}\n"
        
        summary += "\n" + "="*50 + "\n\n"
    
    return summary

# Create output directory for results
output_dir = 'PCA Cleaned Data/Period Analysis'
import os
os.makedirs(output_dir, exist_ok=True)

# Perform PCA for each period
summary_text = "PCA Analysis Summary by Period and Factor Group\n"
summary_text += "==========================================\n"
summary_text += "\nFollowing E3S paper methodology:\n"
summary_text += "- Supply Factors (4 variables): Housing Starts, House Prices, PRFI, PPI\n"
summary_text += "- Demand Factors (4 variables): Population, DPI, Mortgage Rate, GDP\n"
summary_text += "- Market Environment (2 variables): Unemployment, CPI\n"
summary_text += "\nAnalysis shows components needed to explain 90% variance in each group.\n"
summary_text += "Significant loadings are those >= |0.3|\n\n"

for start_date, end_date in periods:
    period_data_supply = supply_df[start_date:end_date]
    period_data_demand = demand_df[start_date:end_date]
    period_data_market = market_df[start_date:end_date]
    
    period_name = f"{start_date[:4]}_{end_date[:4]}"
    
    if len(period_data_supply) == 0:
        continue
    
    # Perform analysis for this period
    period_summary = perform_period_analysis(
        period_data_supply,
        period_data_demand,
        period_data_market,
        period_name,
        output_dir
    )
    
    summary_text += period_summary

# Save summary to text file
with open(f'{output_dir}/pca_summary_by_period.txt', 'w') as f:
    f.write(summary_text)

print("PCA analysis by period completed. Results saved in 'PCA Cleaned Data/Period Analysis' directory.")

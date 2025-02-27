import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

def load_data():
    # Load your actual datasets
    gdp = pd.read_csv('GDP_quarterly.csv')
    population = pd.read_csv('population_quarterly_totals.csv')
    housing_starts = pd.read_csv('HOUST1F_quarterly_totals.csv')
    prices = pd.read_csv('MSPUS_quarterly.csv')
    income = pd.read_csv('MEHOINUSA672N_quarterly.csv')
    mortgage = pd.read_csv('MORTGAGE30US_quarterly.csv')
    construction = pd.read_csv('PPI_Residential_Construction_quarterly_average.csv')
    
    # Convert quarterly format to datetime
    def quarter_to_date(quarter_str):
        year = int(quarter_str[:4])
        quarter = int(quarter_str[-1])
        month = (quarter - 1) * 3 + 1
        return pd.to_datetime(f"{year}-{month:02d}-01")
    
    # Process each dataset with appropriate date column names
    gdp['Date'] = gdp['Quarterly_Format'].apply(quarter_to_date)
    population['Date'] = population['Quarterly_Format'].apply(quarter_to_date)
    housing_starts['Date'] = housing_starts['Quarterly_Format'].apply(quarter_to_date)
    prices['Date'] = prices['Quarterly_Format'].apply(quarter_to_date)
    income['Date'] = income['observation_date'].apply(quarter_to_date)
    mortgage['Date'] = mortgage['Quarterly_Format'].apply(quarter_to_date)
    construction['Date'] = construction['Quarterly_Format'].apply(quarter_to_date)
    
    # Set Date as index and select only the value columns
    gdp.set_index('Date', inplace=True)
    population.set_index('Date', inplace=True)
    housing_starts.set_index('Date', inplace=True)
    prices.set_index('Date', inplace=True)
    income.set_index('Date', inplace=True)
    mortgage.set_index('Date', inplace=True)
    construction.set_index('Date', inplace=True)
    
    # Select only the value columns (second column for each dataset)
    data = pd.concat([
        gdp['GDP'],
        population.iloc[:, 1],
        housing_starts.iloc[:, 1],
        prices.iloc[:, 1],
        income['MEHOINUSA672N'],
        mortgage.iloc[:, 1],
        construction.iloc[:, 1]
    ], axis=1)
    
    # Name the columns
    data.columns = ['GDP', 'Population', 'Housing_Starts', 'Prices', 'Income', 'Mortgage_Rate', 'Construction_PPI']
    
    return data

def perform_pca_analysis(data):
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=3)
    pca_result = pca.fit_transform(scaled_data)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Get factor loadings
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    
    return {
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'loadings': loadings,
        'feature_names': data.columns
    }

def validate_results(data):
    # Split data into periods
    periods = {
        '1996-1999': data.loc['1996':'1999'],
        '2000-2003': data.loc['2000':'2003'],
        '2004-2007': data.loc['2004':'2007'],
        '2008-2011': data.loc['2008':'2011'],
        '2012-2015': data.loc['2012':'2015'],
        '2016-2019': data.loc['2016':'2019'],
        '2020-2022': data.loc['2020':'2022']
    }
    
    results = {}
    for period_name, period_data in periods.items():
        results[period_name] = perform_pca_analysis(period_data)
    
    return results

def print_validation_results(results):
    print("PCA Validation Results\n")
    print("1. Explanatory Power Analysis:")
    print("─" * 60)
    for period, result in results.items():
        print(f"\nPeriod: {period}")
        print(f"PC1: {result['explained_variance'][0]*100:.2f}%")
        print(f"PC2: {result['explained_variance'][1]*100:.2f}%")
        print(f"PC3: {result['explained_variance'][2]*100:.2f}%")
        print(f"Cumulative: {result['cumulative_variance'][2]*100:.2f}%")
    
    print("\n2. Factor Loadings Analysis:")
    print("─" * 60)
    for period, result in results.items():
        print(f"\nPeriod: {period}")
        feature_names = result['feature_names']
        loadings = result['loadings']
        for pc in range(3):
            # Get indices of top 3 factors by absolute loading values
            top_indices = np.argsort(-abs(loadings[:, pc]))[:3]
            print(f"PC{pc+1} top factors:", [
                f"{feature_names[i]}: {loadings[i, pc]:.3f}"
                for i in top_indices
            ])

# Execute validation
try:
    print("Loading data...")
    data = load_data()
    print("Performing validation...")
    results = validate_results(data)
    print_validation_results(results)
except Exception as e:
    print(f"Error during validation: {str(e)}")
    import traceback
    print(traceback.format_exc())

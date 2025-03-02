import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression

# Load normalized data for each factor
def load_normalized_data(variable):
    """Load normalized data for a variable"""
    file_path = f'PCA Cleaned Data/{variable.lower()}_normalized.csv'
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        # Convert date to match quarterly format (first month of quarter)
        df['DATE'] = pd.to_datetime(df['DATE'])
        normalized_col = [col for col in df.columns if '_NORMALIZED' in col][0]
        df = df[['DATE', normalized_col]]
        df.set_index('DATE', inplace=True)
        return df
    return None

# Define variable groups based on the E3S paper methodology but using our available variables
VARIABLE_GROUPS = {
    'Supply': ['housing_starts', 'prfi', 'vacancy'],  # Housing starts, Private Residential Fixed Investment, Vacancy Rate
    'Demand': ['population', 'dpi', 'unemployment'],  # Population, Disposable Personal Income, Unemployment
    'Market': ['mortgage', 'mspus', 'cpi', 'ppi', 'gdp']  # Mortgage rate, Median Sales Price, CPI, PPI, GDP
}

def load_group_data(group):
    """Load normalized data for a group of factors"""
    if group not in VARIABLE_GROUPS:
        print(f"Unknown group: {group}")
        return None
        
    group_data = []
    for var in VARIABLE_GROUPS[group]:
        file_path = os.path.join('PCA Cleaned Data', f'{var}_normalized.csv')
        if not os.path.exists(file_path):
            print(f"Missing data file: {file_path}")
            continue
            
        df = pd.read_csv(file_path)
        df['DATE'] = pd.to_datetime(df['DATE'])
        df.set_index('DATE', inplace=True)
        
        # Get the normalized column
        normalized_col = [col for col in df.columns if '_NORMALIZED' in col][0]
        df = df[[normalized_col]]
        df.columns = [var]  # Use the variable name as the column name
        group_data.append(df)
    
    if not group_data:
        return None
        
    return pd.concat(group_data, axis=1)

def get_variable_data(group):
    """Get the normalized data for variables in a group"""
    data_dict = {}
    for name, var in VARIABLE_GROUPS[group].items():
        data = load_normalized_data(var)
        if data is not None:
            data.columns = [name]
            data_dict[name] = data
    
    if data_dict:
        return pd.concat(data_dict.values(), axis=1)
    return None

def load_hai_data():
    """Load and prepare HAI data"""
    # Read HAI data directly from the cleaned file
    hai_df = pd.read_csv('PCA Cleaned Data/hai_1996_2023.csv')
    hai_df['DATE'] = pd.to_datetime(hai_df['DATE'])
    hai_df.set_index('DATE', inplace=True)
    return hai_df['HAI']

def load_all_data():
    """Load data for the entire time period"""
    # Load HAI data
    hai_data = load_hai_data()
    if hai_data is None:
        return None, None, None
    
    # Load all variable groups
    group_data = []
    for group in VARIABLE_GROUPS:
        data = load_group_data(group)
        if data is not None:
            group_data.append(data)
    
    if not group_data:
        return None, None, None
    
    # Combine all variables
    X_combined = pd.concat(group_data, axis=1)
    
    # Align dates between X and y
    common_dates = X_combined.index.intersection(hai_data.index)
    X_combined = X_combined.loc[common_dates]
    hai_data = hai_data.loc[common_dates]
    
    return X_combined, hai_data.values, X_combined.index

def calculate_rvi(X, y, max_components=10):
    """
    Calculate Residual Variance Indicator (RVI) for component selection
    as described in the paper.
    """
    n = X.shape[0]
    rvi_values = []
    
    for a in range(1, max_components + 1):
        # Fit PLS with a components
        pls = PLSRegression(n_components=a)
        pls.fit(X, y)
        
        # Calculate residuals
        y_pred = pls.predict(X)
        residuals = y - y_pred
        
        # Calculate RVI
        rvi = np.sum(residuals**2) / (n - a - 1)
        rvi_values.append(rvi)
        
        print(f"RVI for {a} components: {rvi:.6f}")
        
        # Check if RVI has stabilized (less than 5% change)
        if a > 1:
            change = abs(rvi_values[-1] - rvi_values[-2]) / rvi_values[-2]
            if change < 0.05:
                print(f"RVI stabilized at {a} components (change: {change:.3%})")
                return a
    
    return len(rvi_values)

def enhanced_pls_regression(X, y, n_components=None):
    """
    Enhanced PLS regression following paper equations:
    Xa-1 = tapaT + E  (Eq. 6)
    pa = XTa-1ta / taTta  (Eq. 7)
    ya-1 = taqa + f  (Eq. 8)
    qa = yTa-1ta / taTta  (Eq. 9)
    """
    if n_components is None:
        n_components = calculate_rvi(X, y)
    
    n_samples, n_features = X.shape
    
    # Initialize matrices
    T = np.zeros((n_samples, n_components))  # X scores
    P = np.zeros((n_features, n_components))  # X loadings
    Q = np.zeros((1, n_components))          # Y loadings
    W = np.zeros((n_features, n_components))  # Weights
    
    # Copy X and y to avoid modifying original data
    Xa = X.copy()
    ya = y.copy()
    
    # For each component
    for a in range(n_components):
        # Calculate weights
        w = np.dot(Xa.T, ya)
        w = w / np.linalg.norm(w)
        W[:, a] = w.ravel()
        
        # Calculate scores
        t = np.dot(Xa, w)
        T[:, a] = t.ravel()
        
        # Calculate loadings
        p = np.dot(Xa.T, t) / np.dot(t.T, t)
        P[:, a] = p.ravel()
        
        q = np.dot(ya.T, t) / np.dot(t.T, t)
        Q[0, a] = q
        
        # Deflate X and y
        Xa = Xa - np.outer(t, p)
        ya = ya - t * q
    
    # Calculate regression coefficients
    W_star = W.dot(np.linalg.inv(P.T.dot(W)))
    coefficients = W_star.dot(Q.T)
    
    # Calculate intercept
    y_mean = np.mean(y)
    x_mean = np.mean(X, axis=0)
    intercept = y_mean - np.dot(x_mean, coefficients)
    
    # Make predictions
    y_pred = np.dot(X, coefficients) + intercept
    r2 = r2_score(y, y_pred)
    
    # Calculate variance explained
    x_scores = np.dot(X, W)
    x_loadings = P
    
    return {
        'coefficients': coefficients,
        'intercept': intercept[0],
        'r2': r2,
        'n_components': n_components,
        'x_scores': x_scores,
        'x_loadings': x_loadings,
        'y_loadings': Q
    }

def analyze_results(X, y, dates, feature_names):
    """Analyze dataset using enhanced PLS regression"""
    print("\nPerforming PLS regression analysis...")
    
    # Perform PLS regression
    results = enhanced_pls_regression(X, y)
    
    print(f"\nFinal model R-squared: {results['r2']:.4f}")
    
    # Print regression equation
    print("\nRegression equation:")
    print(f"HAI = {results['intercept']:.4f}", end='')
    for i, (coef, name) in enumerate(zip(results['coefficients'], feature_names)):
        sign = '+' if coef > 0 else '-'
        print(f" {sign} {abs(float(coef)):.4f}·{name}", end='')
    print()
    
    # Get component loadings with variable names
    loadings = []
    for i in range(results['n_components']):
        component_loadings = pd.Series(results['x_loadings'][:, i], index=feature_names)
        loadings.append(component_loadings)
        print(f"\nComponent {i+1} loadings:")
        for var, loading in component_loadings.items():
            print(f"{var}: {loading:.4f}")
    
    # Analyze changes over time periods
    scores = results['x_scores']
    scores_df = pd.DataFrame(scores, index=dates, columns=[f'Component_{i+1}' for i in range(results['n_components'])])
    y_series = pd.Series(y.flatten(), index=dates)
    
    # Define time periods
    periods = [
        ('1996-2000', '1996-01-01', '2000-12-31'),
        ('2001-2006', '2001-01-01', '2006-12-31'),
        ('2007-2012', '2007-01-01', '2012-12-31'),
        ('2013-2018', '2013-01-01', '2018-12-31'),
        ('2019-2023', '2019-01-01', '2023-12-31')
    ]
    
    print("\nComponent contributions by time period:")
    for period_name, start_date, end_date in periods:
        period_scores = scores_df[(scores_df.index >= start_date) & (scores_df.index <= end_date)]
        period_hai = y_series[(y_series.index >= start_date) & (y_series.index <= end_date)]
        print(f"\n{period_name}:")
        print(f"Average HAI: {period_hai.mean():.2f}")
        for comp_num in range(results['n_components']):
            comp_contribution = period_scores[f'Component_{comp_num+1}'].mean()
            print(f"Component {comp_num+1} average contribution: {comp_contribution:.4f}")
            # Show top contributing variables for this component
            comp_loadings = loadings[comp_num].abs().sort_values(ascending=False)
            print("Top contributing variables:")
            for var, loading in comp_loadings[:3].items():
                print(f"  {var}: {loading:.4f}")
    
    return results

def create_detailed_summary(results):
    """Create a comprehensive summary of findings"""
    summary = {
        'model_performance': {
            'r2': results['r2'],
            'n_components': results['n_components']
        },
        'component_loadings': {}
    }
    
    # Add loadings for each component
    for i in range(results['n_components']):
        loadings = results['x_loadings'][:, i]
        summary['component_loadings'][f'Component_{i+1}'] = loadings.tolist()
    
    return summary

def save_composite_loadings(loadings_dict, output_dir):
    """Save composite loadings to CSV"""
    # Create DataFrame from dictionary of Series
    all_loadings = pd.DataFrame(loadings_dict).T
    all_loadings.to_csv(os.path.join(output_dir, 'composite_component_loadings.csv'))

def create_math_documentation():
    """Create detailed mathematical documentation of the PLS regression process"""
    doc = """
Mathematical Documentation of PLS Regression Analysis
==================================================

1. Data Preprocessing
-------------------
For each independent variable x_i, we calculate the quarter-over-quarter percent change:
    Δx_i = (x_i,t - x_i,t-1) / x_i,t-1 * 100

We then normalize using max absolute value scaling to preserve the direction of change:
    x_i_normalized = Δx_i / max(|Δx_i|)

This scaling method:
- Preserves the sign (direction) of changes
- Bounds values between -1 and 1
- Maintains relative magnitude of changes
- Allows for comparison between variables on different scales

2. Component Selection
--------------------
The number of components is determined using the Residual Variance Indicator (RVI):
    RVI_a = trace(X_a^T X_a) / trace(X_{a-1}^T X_{a-1})

where:
- X_a is the residual matrix after extracting a components
- The algorithm stops when RVI stabilizes (change < 2.5%)

3. PLS Regression Model
---------------------
The PLS regression follows these steps:

a) For each component a:
   - Calculate scores: t_a = X_{a-1} w_a
   - Calculate X loadings: p_a = X_{a-1}^T t_a / (t_a^T t_a)
   - Calculate Y loadings: q_a = y_{a-1}^T t_a / (t_a^T t_a)
   - Update X: X_a = X_{a-1} - t_a p_a^T
   - Update y: y_a = y_{a-1} - t_a q_a

b) Final regression equation:
    HAI = β_0 + Σ(β_i * x_i)
    where β_i are the regression coefficients

4. Time Period Analysis
---------------------
To analyze how drivers of housing affordability change over time:

a) Split data into periods:
   - 1996-2000: Pre-housing boom
   - 2001-2006: Housing boom
   - 2007-2012: Financial crisis and recovery
   - 2013-2018: Post-crisis period
   - 2019-2023: Recent period

b) For each period:
   - Calculate mean HAI
   - Calculate mean component scores
   - Identify top contributing variables through loading analysis

5. Interpretation
---------------
- Component loadings (p_a) show how variables relate to each component
- Significant loadings are those with |loading| > 0.35
- Component scores (t_a) show how each observation relates to components
- Regression coefficients (β_i) show direct effects on HAI
- Time period analysis reveals how drivers change over different market conditions

6. Model Evaluation
-----------------
- R² measures total variance explained in HAI
- Individual component contributions show incremental variance explained
- Loading patterns reveal underlying market dynamics
- Temporal analysis shows evolution of housing affordability drivers
"""
    return doc

def plot_scores_over_time(scores, output_dir):
    """Plot how scores evolve over time"""
    plt.figure(figsize=(12, 6))
    for component in scores.columns:
        plt.plot(scores.index, scores[component], label=component, marker='o')
    
    plt.title('Component Scores Over Time')
    plt.xlabel('Year')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    plt.savefig(os.path.join(output_dir, 'scores_over_time.png'))
    plt.close()

def main():
    """Run PLS regression analysis"""
    print("Loading data...")
    
    # Load all data
    X_combined, y, dates = load_all_data()
    if X_combined is None or y is None:
        print("Error loading data")
        return
    
    # Create output directory
    output_dir = 'PCA Cleaned Data/PLS Single Analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Perform analysis using sklearn's PLSRegression for stability
    n_components = 2  # Fixed based on previous analysis
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_combined.values, y)
    y_pred = pls.predict(X_combined.values)
    r2 = r2_score(y, y_pred)
    
    # Create results dictionary matching our format
    results = {
        'coefficients': pls.coef_.flatten(),
        'intercept': pls.intercept_[0],
        'r2': r2,
        'n_components': n_components,
        'x_scores': pls.x_scores_,
        'x_loadings': pls.x_loadings_,
        'y_loadings': pls.y_loadings_,
        'feature_names': X_combined.columns,
        'dates': dates
    }
    
    # Save results
    loadings_df = pd.DataFrame(
        pls.x_loadings_,
        columns=[f'PC{i+1}' for i in range(n_components)],
        index=X_combined.columns
    )
    loadings_df.to_csv(os.path.join(output_dir, 'component_loadings.csv'))
    
    scores_df = pd.DataFrame(
        pls.x_scores_,
        columns=[f'Component_{i+1}' for i in range(n_components)],
        index=dates
    )
    scores_df.to_csv(os.path.join(output_dir, 'component_scores.csv'))
    
    coef_df = pd.DataFrame(
        results['coefficients'],
        index=X_combined.columns,
        columns=['Coefficient']
    )
    coef_df.to_csv(os.path.join(output_dir, 'regression_coefficients.csv'))
    
    # Create and save mathematical documentation
    doc = create_math_documentation()
    with open(os.path.join(output_dir, 'mathematical_documentation.txt'), 'w') as f:
        f.write(doc)
    
    # Print final summary
    print(f"\nModel R-squared: {results['r2']:.4f}")
    
    print("\nComponent Loadings:")
    for i in range(results['n_components']):
        print(f"\nPC{i+1} significant loadings (|loading| > 0.35):")
        loadings = pd.Series(results['x_loadings'][:, i], index=X_combined.columns)
        significant = loadings[abs(loadings) > 0.35]
        for var, loading in significant.items():
            print(f"{var}: {loading:.4f}")
    
    # Create detailed analysis
    print("\nDetailed time period analysis:")
    periods = [
        ('1996-2000', '1996-01-01', '2000-12-31'),
        ('2001-2006', '2001-01-01', '2006-12-31'),
        ('2007-2012', '2007-01-01', '2012-12-31'),
        ('2013-2018', '2013-01-01', '2018-12-31'),
        ('2019-2023', '2019-01-01', '2023-12-31')
    ]
    
    for period_name, start_date, end_date in periods:
        period_scores = scores_df[(scores_df.index >= start_date) & (scores_df.index <= end_date)]
        period_y = pd.Series(y.flatten(), index=dates)
        period_y = period_y[(period_y.index >= start_date) & (period_y.index <= end_date)]
        
        print(f"\n{period_name}:")
        print(f"Average HAI: {period_y.mean():.2f}")
        for comp_num in range(n_components):
            comp_contribution = period_scores[f'Component_{comp_num+1}'].mean()
            print(f"Component {comp_num+1} average contribution: {comp_contribution:.4f}")
            print("Top contributing variables:")
            loadings = pd.Series(results['x_loadings'][:, comp_num], index=X_combined.columns)
            top_loadings = loadings.abs().sort_values(ascending=False)[:3]
            for var, loading in top_loadings.items():
                print(f"  {var}: {abs(loadings[var]):.4f}")
    
    print(f"\nAnalysis complete! Results saved in: {output_dir}")

if __name__ == "__main__":
    main()

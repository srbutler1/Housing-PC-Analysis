import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from scipy import stats
from pca_by_period import load_data, split_into_periods, analyze_periods
from statsmodels.stats.outliers_influence import variance_inflation_factor

def perform_regression_analysis(data, period_name):
    """
    Performs regression analysis for each period to validate PCA findings
    """
    # Define features explicitly
    feature_cols = ['housing_starts', 'population', 'housing_prices', 
                   'construction_prices', 'gdp', 'unemployment', 
                   'income', 'mortgage_rates', 'cpi', 'investment']
    
    # Prepare X and y
    X = data[feature_cols].copy()
    y = data['housing_prices'] / data['income']  # affordability metric
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Add constant for statsmodels
    X_with_const = sm.add_constant(X_scaled)
    
    # Fit regression
    model = sm.OLS(y, X_with_const)
    results = model.fit()
    
    # Calculate VIF
    vif_data = pd.DataFrame()
    vif_data["Variable"] = X_scaled.columns
    vif_data["VIF"] = [variance_inflation_factor(X_scaled.values, i) 
                       for i in range(X_scaled.shape[1])]
    
    # Perform additional statistical tests
    residuals = results.resid
    normality_test = stats.normaltest(residuals)
    durbin_watson = durbin_watson_test(residuals)
    
    return {
        'model_results': results,
        'vif_data': vif_data,
        'normality_test': normality_test,
        'durbin_watson': durbin_watson,
        'r_squared': results.rsquared,
        'adj_r_squared': results.rsquared_adj
    }

def durbin_watson_test(residuals):
    """Calculate Durbin-Watson statistic"""
    return np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)

def variance_inflation_factor(exog, exog_idx):
    """Calculate VIF for a single feature"""
    r_squared = sm.OLS(exog[:, exog_idx], 
                      sm.add_constant(np.delete(exog, exog_idx, axis=1))
                     ).fit().rsquared
    return 1. / (1. - r_squared)

def categorize_impacts(model):
    """
    Categorizes variable impacts based on coefficient significance and magnitude
    """
    impacts = {}
    
    # Define impact thresholds
    thresholds = {
        'strong': 0.5,
        'moderate': 0.3,
        'weak': 0.1
    }
    
    for var, coef in model.params.items():
        if var == 'const':
            continue
            
        # Get p-value
        p_val = model.pvalues[var]
        
        # Determine significance and direction
        if p_val < 0.05:
            magnitude = abs(coef)
            direction = 'positive' if coef > 0 else 'negative'
            
            if magnitude > thresholds['strong']:
                strength = 'Strong'
            elif magnitude > thresholds['moderate']:
                strength = 'Moderate'
            else:
                strength = 'Weak'
                
            impacts[var] = f"{strength} {direction}"
        else:
            impacts[var] = "Not significant"
            
    return impacts

def validate_pca_with_regression(data):
    """
    Validates PCA findings with regression analysis across different periods
    """
    # Split data into periods using the same function as PCA
    periods = split_into_periods(data)  # Use the imported split_into_periods function
    
    regression_results = {}
    for period_name, period_data in periods.items():
        regression_results[period_name] = perform_regression_analysis(period_data, period_name)
    
    return regression_results

def print_regression_validation(results):
    """
    Prints regression analysis results in a formatted way
    """
    print("\nRegression Analysis Validation of PCA Findings")
    print("=" * 80)
    
    for period, result in results.items():
        print(f"\nPeriod: {period}")
        print("-" * 40)
        print(f"R-squared: {result['r_squared']:.3f}")
        print(f"Adjusted R-squared: {result['adj_r_squared']:.3f}")
        print("\nVariable Impacts:")
        
        # Group impacts by strength
        impacts = result['model_results'].params
        strong_impacts = {k: v for k, v in impacts.items() if abs(v) > 0.5}
        moderate_impacts = {k: v for k, v in impacts.items() if 0.3 < abs(v) <= 0.5}
        weak_impacts = {k: v for k, v in impacts.items() if 0.1 < abs(v) <= 0.3}
        
        print("\nStrong Impacts:")
        for var, impact in strong_impacts.items():
            print(f"  - {var}: {impact}")
            
        print("\nModerate Impacts:")
        for var, impact in moderate_impacts.items():
            print(f"  - {var}: {impact}")
            
        print("\nWeak Impacts:")
        for var, impact in weak_impacts.items():
            print(f"  - {var}: {impact}")
        
        print("\nModel Summary:")
        print(result['model_results'].summary().tables[1])
        print("\n" + "=" * 80)

def compare_with_pca(pca_results, regression_results):
    """
    Compares PCA and regression results for each period
    """
    for period in pca_results.keys():
        print(f"\nPeriod: {period}")
        print("-" * 50)
        
        # Get PCA loadings for this period
        pca_loadings = pca_results[period]['loadings']
        explained_var = pca_results[period]['explained_variance']
        
        # Get regression results for this period
        reg_results = regression_results[period]['model_results']
        
        print(f"PCA Explained Variance (first 3 components):")
        for i, var in enumerate(explained_var[:3]):
            print(f"PC{i+1}: {var:.4f} ({var*100:.2f}%)")
        
        print("\nTop contributing factors (absolute loadings) for first 3 PCs:")
        for i in range(3):
            pc_col = f'PC{i+1}'
            sorted_loadings = pca_loadings[pc_col].abs().sort_values(ascending=False)
            print(f"\nPC{i+1}:")
            print(sorted_loadings.head().to_string())
        
        print("\nRegression Results:")
        print(f"R-squared: {reg_results.rsquared:.4f}")
        print(f"Adjusted R-squared: {reg_results.rsquared_adj:.4f}")
        
        # Print significant features from regression
        print("\nSignificant features (p < 0.05):")
        significant = reg_results.pvalues[reg_results.pvalues < 0.05]
        for feature, pval in significant.items():
            print(f"{feature}: p={pval:.4f}, coef={reg_results.params[feature]:.4f}")

def check_correlations(data):
    """
    Analyze and visualize correlations between variables
    """
    # Select numeric columns, excluding the quarter column
    numeric_cols = [col for col in data.columns if col != 'quarter']
    
    # Calculate correlation matrix
    corr_matrix = data[numeric_cols].corr()
    
    # Print high correlations (above 0.7 or below -0.7)
    print("High Correlations (|r| > 0.7):")
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                print(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_matrix.iloc[i, j]:.3f}")
    
    return corr_matrix

# Execute validation
if __name__ == "__main__":
    try:
        print("Loading data...")
        data = load_data()
        
        print("Performing PCA analysis...")
        pca_results = analyze_periods()  # Changed from validate_results to analyze_periods
        
        print("Performing regression validation...")
        regression_results = validate_pca_with_regression(data)
        
        print("Printing results...")
        print_regression_validation(regression_results)
        
        print("Comparing PCA and regression results...")
        compare_with_pca(pca_results, regression_results)
        
        print("Checking correlations...")
        correlations = check_correlations(data)
        print("\nFull Correlation Matrix:")
        print(correlations.round(3))
        
    except Exception as e:
        print(f"Error during validation: {str(e)}")
        import traceback
        print(traceback.format_exc())
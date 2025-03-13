import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

# Add parent directory to path to import functions from pls_regression_analysis.py
sys.path.append('/Users/appleowner/Downloads/Thesis/Data/PC Analysis ')
from pls_regression_analysis import load_all_data

def confirm_r2():
    """Confirm the R² value of the PLS regression with 2 components"""
    print("Loading data...")
    
    # Load all data
    X_combined, y, dates = load_all_data()
    if X_combined is None or y is None:
        print("Error loading data")
        return
    
    print(f"\nData loaded successfully:")
    print(f"X shape: {X_combined.shape}")
    print(f"y shape: {y.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")
    
    # Run PLS regression with 2 components
    print("\nRunning PLS regression with 2 components...")
    pls = PLSRegression(n_components=2)
    pls.fit(X_combined.values, y)
    
    # Calculate predictions and R²
    y_pred = pls.predict(X_combined.values)
    r2 = r2_score(y, y_pred)
    
    print(f"\nR² value: {r2:.6f}")
    print(f"Explained variance: {r2 * 100:.2f}%")
    
    # Calculate component scores
    T = pls.transform(X_combined.values)
    
    # Calculate loadings
    loadings = pls.x_loadings_
    
    # Get variable names
    variable_names = X_combined.columns
    
    # Print loadings for each component
    print("\nComponent loadings:")
    print("-" * 60)
    print(f"{'Variable':<15} {'Component 1':<15} {'Component 2':<15}")
    print("-" * 60)
    
    for i, var in enumerate(variable_names):
        print(f"{var:<15} {loadings[i, 0]:>12.6f}   {loadings[i, 1]:>12.6f}")
    
    return r2

if __name__ == "__main__":
    confirm_r2()

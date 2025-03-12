import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score

# Add parent directory to path to import functions from pls_regression_analysis.py
sys.path.append('/Users/appleowner/Downloads/Thesis/Data/PC Analysis ')
from pls_regression_analysis import load_all_data, calculate_rvi

def validate_optimal_components():
    """Validate the optimal number of components for PLS regression"""
    print("Loading data...")
    
    # Load all data
    X_combined, y, dates = load_all_data()
    if X_combined is None or y is None:
        print("Error loading data")
        return
    
    # Create output directory
    output_dir = 'scratch_runs'
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate RVI for different numbers of components
    max_components = 10
    n = X_combined.shape[0]
    rvi_values = []
    r2_values = []
    
    print("\nCalculating RVI and R² for different numbers of components:")
    print("-" * 60)
    print(f"{'Components':<10} {'RVI':<15} {'Change (%)':<15} {'R²':<15}")
    print("-" * 60)
    
    for a in range(1, max_components + 1):
        # Fit PLS with a components
        pls = PLSRegression(n_components=a)
        pls.fit(X_combined.values, y)
        
        # Calculate predictions and R²
        y_pred = pls.predict(X_combined.values)
        r2 = r2_score(y, y_pred)
        r2_values.append(r2)
        
        # Calculate residuals and RVI
        residuals = y - y_pred
        rvi = np.sum(residuals**2) / (n - a - 1)
        rvi_values.append(rvi)
        
        # Calculate change in RVI
        if a > 1:
            change = (rvi_values[-1] - rvi_values[-2]) / rvi_values[-2] * 100
            print(f"{a:<10} {rvi:.8f}  {change:>+8.3f}%      {r2:.8f}")
        else:
            print(f"{a:<10} {rvi:.8f}  {'N/A':>10}    {r2:.8f}")
    
    # Plot RVI values
    plt.figure(figsize=(12, 8))
    
    # Plot RVI
    plt.subplot(2, 1, 1)
    plt.plot(range(1, max_components + 1), rvi_values, 'o-', color='#1f77b4', linewidth=2)
    plt.axhline(y=rvi_values[1], color='gray', linestyle='--', alpha=0.7)
    plt.title('Residual Variance Indicator (RVI) by Number of Components', fontsize=14)
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('RVI', fontsize=12)
    plt.xticks(range(1, max_components + 1))
    plt.grid(True, alpha=0.3)
    
    # Annotate the RVI values
    for i, rvi in enumerate(rvi_values):
        plt.annotate(f"{rvi:.6f}", 
                    (i+1, rvi), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    # Annotate the change percentages
    for i in range(1, len(rvi_values)):
        change = (rvi_values[i] - rvi_values[i-1]) / rvi_values[i-1] * 100
        plt.annotate(f"{change:+.2f}%", 
                    (i+1, rvi_values[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    fontsize=9,
                    color='red' if abs(change) < 5 else 'black')
    
    # Plot R² values
    plt.subplot(2, 1, 2)
    plt.plot(range(1, max_components + 1), r2_values, 'o-', color='#ff7f0e', linewidth=2)
    plt.title('R² by Number of Components', fontsize=14)
    plt.xlabel('Number of Components', fontsize=12)
    plt.ylabel('R²', fontsize=12)
    plt.xticks(range(1, max_components + 1))
    plt.grid(True, alpha=0.3)
    
    # Annotate the R² values
    for i, r2 in enumerate(r2_values):
        plt.annotate(f"{r2:.4f}", 
                    (i+1, r2), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9)
    
    # Annotate the change in R²
    for i in range(1, len(r2_values)):
        change = r2_values[i] - r2_values[i-1]
        rel_change = change / r2_values[i-1] * 100 if r2_values[i-1] != 0 else float('inf')
        plt.annotate(f"{change:+.4f} ({rel_change:+.2f}%)", 
                    (i+1, r2_values[i]), 
                    textcoords="offset points", 
                    xytext=(0,-15), 
                    ha='center',
                    fontsize=9,
                    color='red' if abs(rel_change) < 5 else 'black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'component_validation.png'), dpi=300, bbox_inches='tight')
    
    print("\nAnalysis complete. Results saved to scratch_runs/component_validation.png")
    
    # Return the optimal number of components based on RVI stabilization (< 5% change)
    for i in range(1, len(rvi_values)):
        change = abs((rvi_values[i] - rvi_values[i-1]) / rvi_values[i-1] * 100)
        if change < 5:
            print(f"\nOptimal number of components based on RVI stabilization (< 5% change): {i+1}")
            return i+1
    
    print(f"\nNo clear stabilization point found. Consider using {len(rvi_values)} components.")
    return len(rvi_values)

if __name__ == "__main__":
    validate_optimal_components()

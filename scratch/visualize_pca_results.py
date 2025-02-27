import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_explained_variance_plot():
    # Data from PCA analysis
    periods = ['1996-1999', '2000-2003', '2004-2007', '2008-2011', 
              '2012-2015', '2016-2019', '2020-2022']
    
    pc1_variance = [62.35, 69.56, 61.87, 50.92, 54.39, 56.91, 61.95]
    pc2_variance = [14.71, 14.61, 19.79, 22.87, 17.47, 16.21, 22.89]
    pc3_variance = [11.74, 6.83, 8.18, 13.21, 9.24, 11.86, 6.39]
    
    # Create stacked bar chart
    plt.figure(figsize=(15, 8))
    bottom = np.zeros(len(periods))
    
    plt.bar(periods, pc1_variance, label='PC1', color='#2ecc71')
    plt.bar(periods, pc2_variance, bottom=pc1_variance, label='PC2', color='#3498db')
    plt.bar(periods, pc3_variance, bottom=np.array(pc1_variance) + np.array(pc2_variance), 
            label='PC3', color='#e74c3c')
    
    plt.title('Explained Variance by Principal Components Across Periods', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Explained Variance (%)', fontsize=12)
    plt.legend(title='Principal Components')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add total variance labels on top of each bar
    for i in range(len(periods)):
        total = pc1_variance[i] + pc2_variance[i] + pc3_variance[i]
        plt.text(i, total + 1, f'{total:.1f}%', ha='center')
    
    plt.tight_layout()
    plt.savefig('pca_variance_by_period.png')
    plt.close()

def create_factor_importance_heatmap():
    # Create a matrix of factor importance across periods
    factors = ['GDP', 'Population', 'Housing Prices', 'Construction Prices', 
               'Unemployment', 'Income', 'Mortgage Rates', 'Housing Starts', 
               'Investment', 'CPI']
    
    periods = ['1996-1999', '2000-2003', '2004-2007', '2008-2011', 
               '2012-2015', '2016-2019', '2020-2022']
    
    # Importance scores (based on PC1 loadings, scaled 0-10)
    importance = [
        # 96-99  00-03  04-07  08-11  12-15  16-19  20-22
        [9,      8,     7,     6,     8,     7,     7],    # GDP
        [9,      8,     7,     9,     9,     8,     7],    # Population
        [7,      7,     6,     7,     8,     7,     9],    # Housing Prices
        [6,      7,     9,     7,     7,     8,     9],    # Construction Prices
        [5,      6,     7,     8,     9,     9,     6],    # Unemployment
        [7,      6,     6,     9,     7,     7,     8],    # Income
        [8,      6,     7,     8,     8,     8,     7],    # Mortgage Rates
        [6,      7,     7,     6,     7,     6,     8],    # Housing Starts
        [8,      8,     7,     8,     8,     8,     6],    # Investment
        [5,      6,     6,     7,     6,     7,     8]     # CPI
    ]
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(importance, annot=True, cmap='YlOrRd', fmt='.0f',
                xticklabels=periods, yticklabels=factors)
    
    plt.title('Factor Importance Across Time Periods', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Factors', fontsize=12)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('factor_importance_heatmap.png')
    plt.close()

if __name__ == "__main__":
    print("Creating PCA visualization plots...")
    create_explained_variance_plot()
    create_factor_importance_heatmap()
    print("Plots saved as 'pca_variance_by_period.png' and 'factor_importance_heatmap.png'")

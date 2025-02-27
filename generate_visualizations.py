import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.dates import DateFormatter
import matplotlib.ticker as ticker

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Create output directory
os.makedirs('visualizations', exist_ok=True)

# Load data
component_scores = pd.read_csv('PCA Cleaned Data/PLS Single Analysis/component_scores.csv')
component_loadings = pd.read_csv('PCA Cleaned Data/PLS Single Analysis/component_loadings.csv')
regression_coefficients = pd.read_csv('PCA Cleaned Data/PLS Single Analysis/regression_coefficients.csv')

# Convert DATE to datetime
component_scores['DATE'] = pd.to_datetime(component_scores['DATE'])
component_scores.set_index('DATE', inplace=True)

# Define time periods
periods = [
    ('1996-2000', '1996-01-01', '2000-12-31'),
    ('2001-2006', '2001-01-01', '2006-12-31'),
    ('2007-2012', '2007-01-01', '2012-12-31'),
    ('2013-2018', '2013-01-01', '2018-12-31'),
    ('2019-2023', '2019-01-01', '2023-12-31')
]

# 1. Create component scores charts for each period
for period_name, start_date, end_date in periods:
    period_scores = component_scores[(component_scores.index >= start_date) & (component_scores.index <= end_date)]
    
    plt.figure(figsize=(12, 6))
    plt.plot(period_scores.index, period_scores['Component_1'], label='Component 1: Housing Market', color='#1f77b4', linewidth=2.5)
    plt.plot(period_scores.index, period_scores['Component_2'], label='Component 2: Economic Conditions', color='#ff7f0e', linewidth=2.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add shaded regions for interpretation
    plt.axhspan(0, 4, alpha=0.1, color='red', label='Reduced Affordability')
    plt.axhspan(-4, 0, alpha=0.1, color='green', label='Improved Affordability')
    
    # Format the plot
    plt.title(f'Component Scores: {period_name}', fontsize=16)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Component Score', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show quarters
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-Q%q'))
    plt.xticks(rotation=45)
    
    # Add annotations for key events
    if period_name == '1996-2000':
        plt.annotate('Dot-com Boom', xy=(pd.Timestamp('1999-07-01'), 1.8), 
                    xytext=(pd.Timestamp('1999-03-01'), 2.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
    
    elif period_name == '2001-2006':
        plt.annotate('Housing Bubble Forms', xy=(pd.Timestamp('2006-04-01'), 2.5), 
                    xytext=(pd.Timestamp('2005-06-01'), 3.2),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
    
    elif period_name == '2007-2012':
        plt.annotate('Financial Crisis', xy=(pd.Timestamp('2008-10-01'), -3.6), 
                    xytext=(pd.Timestamp('2008-04-01'), -4.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
    
    elif period_name == '2013-2018':
        plt.annotate('Post-Crisis Recovery', xy=(pd.Timestamp('2016-01-01'), -1.6), 
                    xytext=(pd.Timestamp('2015-04-01'), -2.3),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
    
    elif period_name == '2019-2023':
        plt.annotate('COVID-19 Pandemic', xy=(pd.Timestamp('2020-04-01'), -3.9), 
                    xytext=(pd.Timestamp('2019-10-01'), -4.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
        
        plt.annotate('2022 Affordability Crisis', xy=(pd.Timestamp('2022-04-01'), 3.0), 
                    xytext=(pd.Timestamp('2021-10-01'), 3.5),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f'visualizations/component_scores_{period_name}.png', dpi=300, bbox_inches='tight')
    plt.close()

# 2. Create component loadings visualization
plt.figure(figsize=(12, 8))

# Create a scatter plot
plt.scatter(component_loadings['PC1'], component_loadings['PC2'], s=100, alpha=0.7)

# Add labels for each point
for i, txt in enumerate(component_loadings.iloc[:, 0]):
    plt.annotate(txt, (component_loadings.iloc[i, 1], component_loadings.iloc[i, 2]), 
                fontsize=11, ha='center', va='center',
                bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))

# Add axis lines
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)

# Add significance threshold circles
circle1 = plt.Circle((0, 0), 0.35, fill=False, color='gray', linestyle='--', alpha=0.7)
plt.gca().add_patch(circle1)

# Format the plot
plt.title('Component Loadings', fontsize=16)
plt.xlabel('Component 1: Housing Market Dynamics', fontsize=12)
plt.ylabel('Component 2: Economic Conditions', fontsize=12)
plt.grid(True, alpha=0.3)

# Add quadrant labels
plt.text(0.85, 0.85, 'High Impact on Both Components', 
         ha='center', va='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', ec="gray", alpha=0.8))

plt.text(-0.85, 0.85, 'High Economic Impact\nLow/Negative Housing Impact', 
         ha='center', va='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', ec="gray", alpha=0.8))

plt.text(0.85, -0.85, 'High Housing Impact\nLow/Negative Economic Impact', 
         ha='center', va='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', ec="gray", alpha=0.8))

plt.text(-0.85, -0.85, 'Low/Negative Impact on Both Components', 
         ha='center', va='center', fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', ec="gray", alpha=0.8))

plt.xlim(-1.1, 1.1)
plt.ylim(-1.1, 1.1)

plt.tight_layout()
plt.savefig('visualizations/component_loadings.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Create regression coefficients visualization
# Sort by absolute coefficient value
regression_coefficients = regression_coefficients.iloc[1:, :].copy()  # Skip the first row if it's an index
regression_coefficients['Variable'] = component_loadings.iloc[:, 0]
regression_coefficients['AbsCoef'] = regression_coefficients['Coefficient'].abs()
regression_coefficients = regression_coefficients.sort_values('AbsCoef', ascending=False)

# Create a horizontal bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(regression_coefficients['Variable'], regression_coefficients['Coefficient'], 
        color=[('#1f77b4' if x > 0 else '#d62728') for x in regression_coefficients['Coefficient']])

# Add value labels
for bar in bars:
    width = bar.get_width()
    label_x_pos = width + 0.002 if width > 0 else width - 0.008
    plt.text(label_x_pos, bar.get_y() + bar.get_height()/2, f'{width:.4f}',
             va='center', fontsize=10)

# Format the plot
plt.title('Direct Impact of Variables on Housing Affordability Index', fontsize=16)
plt.xlabel('Regression Coefficient', fontsize=12)
plt.ylabel('Variable', fontsize=12)
plt.axvline(x=0, color='gray', linestyle='--', alpha=0.7)
plt.grid(True, alpha=0.3)

# Add interpretation
plt.text(0.04, 1, 'Positive coefficients: Increases reduce affordability', 
         transform=plt.gca().transAxes, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', ec="#1f77b4", alpha=0.8))

plt.text(0.04, 0.95, 'Negative coefficients: Increases improve affordability', 
         transform=plt.gca().transAxes, fontsize=10, 
         bbox=dict(boxstyle="round,pad=0.3", fc='#f0f0f0', ec="#d62728", alpha=0.8))

plt.tight_layout()
plt.savefig('visualizations/regression_coefficients.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Create a heatmap of component scores by year
# Resample to annual average
annual_scores = component_scores.resample('A').mean()
annual_scores.index = annual_scores.index.year

# Create a heatmap
plt.figure(figsize=(14, 8))
heatmap_data = annual_scores.T
heatmap = sns.heatmap(heatmap_data, cmap='RdBu_r', center=0, 
                     annot=True, fmt='.2f', linewidths=.5, 
                     cbar_kws={'label': 'Component Score'})

# Format the plot
plt.title('Annual Average Component Scores (1996-2023)', fontsize=16)
plt.ylabel('Component', fontsize=12)
plt.xlabel('Year', fontsize=12)

# Adjust y-axis labels
plt.yticks([0.5, 1.5], ['Housing Market', 'Economic Conditions'], rotation=0)

plt.tight_layout()
plt.savefig('visualizations/annual_component_scores_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Create a combined visualization for 2022-2023
recent_scores = component_scores[(component_scores.index >= '2022-01-01') & (component_scores.index <= '2023-12-31')]

plt.figure(figsize=(12, 6))
plt.plot(recent_scores.index, recent_scores['Component_1'], label='Component 1: Housing Market', color='#1f77b4', linewidth=2.5)
plt.plot(recent_scores.index, recent_scores['Component_2'], label='Component 2: Economic Conditions', color='#ff7f0e', linewidth=2.5)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)

# Add shaded regions for interpretation
plt.axhspan(0, 4, alpha=0.1, color='red', label='Reduced Affordability')
plt.axhspan(-4, 0, alpha=0.1, color='green', label='Improved Affordability')

# Format the plot
plt.title('2022-2023 Affordability Crisis: Quarterly Component Scores', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Component Score', fontsize=12)
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# Format x-axis to show quarters
plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-Q%q'))
plt.xticks(rotation=45)

# Add annotations for key events
plt.annotate('Peak Unaffordability', xy=(pd.Timestamp('2022-04-01'), 3.0), 
            xytext=(pd.Timestamp('2022-02-01'), 3.8),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10)

plt.annotate('Signs of Moderation', xy=(pd.Timestamp('2023-01-01'), 0.5), 
            xytext=(pd.Timestamp('2023-03-01'), 1.5),
            arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
            fontsize=10)

plt.tight_layout()
plt.savefig('visualizations/2022_2023_crisis.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations created successfully in the 'visualizations' directory.")

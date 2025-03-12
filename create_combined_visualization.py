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

# Convert DATE to datetime
component_scores['DATE'] = pd.to_datetime(component_scores['DATE'])
component_scores.set_index('DATE', inplace=True)

# Define time periods
period1 = ('1996-2012', '1996-01-01', '2012-12-31')
period2 = ('2013-2023', '2013-01-01', '2023-12-31')

# Function to create visualization for a specific date range
def create_period_visualization(period_name, start_date, end_date, filename):
    period_scores = component_scores[(component_scores.index >= start_date) & (component_scores.index <= end_date)]
    
    plt.figure(figsize=(16, 8))
    
    # Plot the component scores
    plt.plot(period_scores.index, period_scores['Component_1'], 
             label='Component 1: Housing Market Dynamics', color='#1f77b4', linewidth=2.5)
    plt.plot(period_scores.index, period_scores['Component_2'], 
             label='Component 2: Economic Conditions', color='#ff7f0e', linewidth=2.5)
    plt.axhline(y=0, color='gray', linestyle='--', alpha=0.7)
    
    # Add shaded regions for interpretation
    plt.axhspan(0, 4, alpha=0.1, color='red', label='Reduced Affordability')
    plt.axhspan(-4, 0, alpha=0.1, color='green', label='Improved Affordability')
    
    # Format the plot
    plt.title(f'Housing Affordability Component Scores ({period_name})', fontsize=18)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Component Score', fontsize=14)
    
    # Only show legend in the first chart
    if period_name == '1996-2012':
        plt.legend(loc='upper right', fontsize=12)
    
    plt.grid(True, alpha=0.3)
    
    # Format x-axis to show years and quarters
    if period_name == '1996-2012':
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y'))
        plt.xticks(rotation=45)
    else:
        plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-Q%q'))
        plt.xticks(rotation=45)
    
    # Add vertical lines to separate sub-periods if needed
    if period_name == '1996-2012':
        sub_period_boundaries = [
            pd.Timestamp('2000-12-31'),
            pd.Timestamp('2006-12-31')
        ]
        
        sub_period_labels = [
            "Pre-Housing Boom\n1996-2000",
            "Housing Boom\n2001-2006",
            "Financial Crisis & Recovery\n2007-2012"
        ]
        
        for boundary in sub_period_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='-', alpha=0.5)
        
        # Add period labels at the bottom of the chart
        for i, label in enumerate(sub_period_labels):
            if i == 0:
                x_pos = pd.Timestamp('1998-06-01')
            elif i == len(sub_period_labels) - 1:
                x_pos = pd.Timestamp('2010-01-01')
            else:
                start = sub_period_boundaries[i-1]
                end = sub_period_boundaries[i] if i < len(sub_period_boundaries) else pd.Timestamp(end_date)
                x_pos = start + (end - start) / 2
            
            plt.text(x_pos, -4.5, label, ha='center', va='top', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))
    
    elif period_name == '2013-2023':
        sub_period_boundaries = [
            pd.Timestamp('2018-12-31')
        ]
        
        sub_period_labels = [
            "Post-Crisis Period\n2013-2018",
            "Recent Period\n2019-2023"
        ]
        
        for boundary in sub_period_boundaries:
            plt.axvline(x=boundary, color='gray', linestyle='-', alpha=0.5)
        
        # Add period labels at the bottom of the chart
        for i, label in enumerate(sub_period_labels):
            if i == 0:
                x_pos = pd.Timestamp('2016-01-01')
            else:
                x_pos = pd.Timestamp('2021-01-01')
            
            plt.text(x_pos, -4.5, label, ha='center', va='top', fontsize=12,
                     bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))
    
    # Add annotations for key events
    key_events = []
    
    if period_name == '1996-2012':
        key_events = [
            # Period 1: 1996-2000
            ('Dot-com Boom', pd.Timestamp('1999-07-01'), 1.8, pd.Timestamp('1999-03-01'), 2.5),
            
            # Period 2: 2001-2006
            ('Housing Bubble Forms', pd.Timestamp('2006-04-01'), 2.5, pd.Timestamp('2005-06-01'), 3.2),
            ('Post-Dot-Com Recovery', pd.Timestamp('2003-07-01'), 1.2, pd.Timestamp('2002-10-01'), 2.0),
            
            # Period 3: 2007-2012
            # Removed Financial Crisis and Housing Market Collapse arrows as requested
            ('Early Recovery', pd.Timestamp('2011-07-01'), -1.0, pd.Timestamp('2011-01-01'), -1.8)
        ]
    else:  # 2013-2023
        key_events = [
            # Period 4: 2013-2018
            ('Post-Crisis Recovery', pd.Timestamp('2016-01-01'), -1.6, pd.Timestamp('2015-04-01'), -2.3),
            ('Fed Rate Normalization', pd.Timestamp('2017-07-01'), -0.8, pd.Timestamp('2017-01-01'), -1.5),
            
            # Period 5: 2019-2023
            # Removed arrows for points that have red callouts
            ('Inflation Pressure', pd.Timestamp('2021-07-01'), 1.5, pd.Timestamp('2021-01-01'), 2.2)
        ]
    
    for event, xy_date, xy_score, text_date, text_score in key_events:
        plt.annotate(event, xy=(xy_date, xy_score), 
                    xytext=(text_date, text_score),
                    arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="gray", alpha=0.8))
    
    # Add additional annotations for extreme points (red callouts only)
    extreme_points = []
    
    if period_name == '1996-2012':
        extreme_points = [
            ('Q3 2008: Housing Market Pressure', pd.Timestamp('2008-07-01'), 2.0),
            ('Q4 2008: Economic Collapse', pd.Timestamp('2008-10-01'), -3.6)
        ]
    else:  # 2013-2023
        extreme_points = [
            ('Q2 2022: Peak Unaffordability\nBoth components positive', pd.Timestamp('2022-04-01'), 3.1),
            ('Q2 2020: COVID Impact\nEconomic collapse', pd.Timestamp('2020-04-01'), -3.9),  
            ('Q3 2020: Strong Rebound\nEconomic recovery', pd.Timestamp('2020-07-01'), 4.3)
        ]
    
    for label, x_date, y_score in extreme_points:
        if (x_date >= pd.Timestamp(start_date)) and (x_date <= pd.Timestamp(end_date)):
            plt.plot(x_date, y_score, 'ro', markersize=8)
            # Position labels to avoid overlap
            if 'COVID Impact' in label:
                x_offset, y_offset = 15, 0
            elif 'Strong Rebound' in label:
                x_offset, y_offset = 15, 0  
            elif 'Peak Unaffordability' in label:
                x_offset, y_offset = 15, 15
            elif 'Housing Market Pressure' in label:
                x_offset, y_offset = 15, 15
            elif 'Economic Collapse' in label:
                x_offset, y_offset = 15, -15
            else:
                x_offset, y_offset = 15, 0
                
            plt.annotate(label, xy=(x_date, y_score), 
                        xytext=(x_offset, y_offset), 
                        textcoords="offset points", 
                        ha='left' if x_offset > 0 else 'right', 
                        va='center',
                        bbox=dict(boxstyle="round,pad=0.3", fc='white', ec="red", alpha=0.8),
                        fontsize=10)
    
    # Set y-axis limits with some padding
    plt.ylim(-5, 5)
    
    # Only add component explanation to the first chart
    if period_name == '1996-2012':
        plt.figtext(0.5, 0.01, 
                   "Component 1 (Housing Market): Mortgage rates, Housing starts, PRFI, Vacancy, Population\n" +
                   "Component 2 (Economic Conditions): GDP, Unemployment, Construction costs, Housing prices", 
                   ha="center", fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.5", fc='#f0f0f0', ec="gray"))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.savefig(f'visualizations/{filename}', dpi=300, bbox_inches='tight')
    plt.close()

# Create visualizations for each period
create_period_visualization(period1[0], period1[1], period1[2], 'component_scores_1996_2012.png')
create_period_visualization(period2[0], period2[1], period2[2], 'component_scores_2013_2023.png')

print("Split visualizations created successfully in the 'visualizations' directory.")

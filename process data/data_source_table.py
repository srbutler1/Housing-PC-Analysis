import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, HTML

def create_research_table():
    # Create the data
    data = {
        'Variable': [
            'Y: Housing Affordability Index',
            'X₁: Housing Starts',
            'X₂: Population',
            'X₃: Housing Prices',
            'X₄: Construction Prices',
            'X₅: GDP',
            'X₆: Unemployment Rate',
            'X₇: Household Income',
            'X₈: Mortgage Rate',
            'X₉: Dollar Index',
            'X₁₀: Consumer Price Index',
            'X₁₁: Fixed Investment'
        ],
        'Description': [
            'Housing Affordability Index for Single-Family Homes (DQYDJ)',
            'Single-Family Housing Starts (thousands of units) (FRED)',
            'Resident Population (thousands) (FRED)',
            'Median Sale Price of Single-Family Homes (FRED)',
            'PPI: Net Inputs to Residential Construction (FRED)',
            'Gross Domestic Product (FRED)',
            'Unemployment Rate (FRED)',
            'Real Median Household Income (FRED)',
            '30-Year Fixed Mortgage Rate (FRED)',
            'Trade-Weighted U.S. Dollar Index (FRED)',
            'Consumer Price Index (BLS)',
            'Private Fixed Investment in Single-Family Construction (BEA)'
        ]
    }

    # Create DataFrame with just two columns
    df = pd.DataFrame(data)
    
    # Create figure and axis with appropriate size
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Hide axes
    ax.axis('tight')
    ax.axis('off')
    
    # Create table
    table = ax.table(cellText=df.values,
                    colLabels=df.columns,
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.3, 0.7])  # Adjusted column widths
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(11)  # Increased font size
    table.scale(1.2, 1.8)  # Increased row height
    
    # Style header
    for j, cell in enumerate(table._cells[(0, j)] for j in range(len(df.columns))):
        cell.set_text_props(weight='bold')
        cell.set_facecolor('#f0f0f0')
    
    # Adjust layout and save
    plt.title('Table 1: Variables Used in Housing Affordability Analysis', pad=20, fontsize=12)
    plt.tight_layout()
    plt.savefig('variables_table.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

# Create and save the table
df = create_research_table()

# Save the table to a file
df.to_csv('variables_table.csv', index=False)

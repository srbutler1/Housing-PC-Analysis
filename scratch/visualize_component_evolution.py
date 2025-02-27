import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_pc1_evolution_plot():
    periods = ['1996-1999', '2000-2003', '2004-2007', '2008-2011', 
               '2012-2015', '2016-2019', '2020-2022']
    
    # PC1 composition over time (top 5 factors for each period)
    data = {
        'GDP':                  [0.349, 0.325, 0.347, 0.250, 0.375, 0.357, 0.344],
        'Population':           [0.348, 0.327, 0.346, 0.376, 0.375, 0.361, 0.333],
        'Construction Prices':  [0.220, 0.280, 0.348, 0.359, 0.310, 0.359, 0.349],
        'Housing Prices':       [0.342, 0.313, 0.285, 0.407, 0.367, 0.280, 0.347],
        'Income':              [0.341, 0.291, 0.233, 0.364, 0.290, 0.275, 0.300],
        'Investment':          [0.348, 0.325, 0.300, 0.350, 0.373, 0.360, 0.290],
        'Unemployment':        [0.280, 0.270, 0.329, 0.325, 0.373, 0.364, 0.314]
    }
    
    # Create line plot
    plt.figure(figsize=(15, 8))
    
    for factor in data.keys():
        plt.plot(periods, data[factor], marker='o', linewidth=2, label=factor)
    
    plt.title('Evolution of Factor Importance in PC1 (Market Fundamentals)', fontsize=14)
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Loading Value', fontsize=12)
    plt.legend(title='Factors', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig('pc1_evolution.png', bbox_inches='tight')
    plt.close()

def create_component_focus_plot():
    periods = ['1996-1999', '2000-2003', '2004-2007', '2008-2011', 
               '2012-2015', '2016-2019', '2020-2022']
    
    # Primary focus of each component by period
    pc1_focus = ['Economic\nGrowth', 'Market\nSize', 'Cost\nPressure', 'Household\nFactors', 
                 'Economic\nRecovery', 'Market\nHealth', 'Price\nPressure']
    pc2_focus = ['Financial', 'Cost', 'Supply', 'Price', 'Financial', 'Financial', 'Supply/Income']
    pc3_focus = ['Activity', 'Inflation', 'Financial', 'Activity', 'Cost', 'Price', 'Activity']
    
    # Create figure with subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))
    
    # Plot for PC1
    ax1.bar(periods, [1]*len(periods), color='lightblue')
    ax1.set_title('PC1 Primary Focus Over Time', pad=20)
    for i, txt in enumerate(pc1_focus):
        ax1.text(i, 0.5, txt, ha='center', va='center')
    ax1.set_ylim(0, 1)
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # Plot for PC2
    ax2.bar(periods, [1]*len(periods), color='lightgreen')
    ax2.set_title('PC2 Primary Focus Over Time', pad=20)
    for i, txt in enumerate(pc2_focus):
        ax2.text(i, 0.5, txt, ha='center', va='center')
    ax2.set_ylim(0, 1)
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # Plot for PC3
    ax3.bar(periods, [1]*len(periods), color='lightcoral')
    ax3.set_title('PC3 Primary Focus Over Time', pad=20)
    for i, txt in enumerate(pc3_focus):
        ax3.text(i, 0.5, txt, ha='center', va='center')
    ax3.set_ylim(0, 1)
    ax3.set_xticks(range(len(periods)))
    ax3.set_xticklabels(periods, rotation=45)
    ax3.set_yticks([])
    
    plt.suptitle('Evolution of Principal Component Focus Areas', fontsize=16, y=0.95)
    plt.tight_layout()
    plt.savefig('component_focus_evolution.png')
    plt.close()

if __name__ == "__main__":
    print("Creating component evolution visualizations...")
    create_pc1_evolution_plot()
    create_component_focus_plot()
    print("Plots saved as 'pc1_evolution.png' and 'component_focus_evolution.png'")

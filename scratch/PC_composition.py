import pandas as pd


def create_html_component_table():
    # Create data
    data = {
        'Period': ['1996-1999', '2000-2003', '2004-2007', '2008-2011', '2012-2015', '2016-2019', '2020-2022'],
        'Primary Component': [
            'Economic Growth (GDP, Population)',
            'Market Size (Population, Investment)',
            'Cost Pressure (Construction, GDP)',
            'Household (Population, Income)',
            'Economic Recovery (Population, GDP)',
            'Market Health (Unemployment, Population)',
            'Price Pressure (Construction, Housing)'
        ],
        'Secondary Component': [
            'Financial (Mortgage Rates)',
            'Cost (Construction Prices)',
            'Supply (Investment)',
            'Price (CPI, GDP)',
            'Financial (Mortgage Rates)',
            'Financial (Mortgage Rates)',
            'Supply/Income (Housing Starts)'
        ],
        'Explained Variance': [
            '88.80%',
            '91.00%',
            '89.84%',
            '87.00%',
            '81.10%',
            '84.98%',
            '91.23%'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create HTML table with styling
    html_table = """
    <style>
        .component-table {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }
        .component-table th {
            background-color: #f2f2f2;
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
            font-weight: bold;
        }
        .component-table td {
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }
        .component-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .table-caption {
            font-weight: bold;
            font-size: 1.1em;
            margin-bottom: 10px;
            text-align: left;
        }
    </style>
    <div class="table-caption">Table 2: Evolution of Housing Market Components (1996-2022)</div>
    """
    
    # Convert DataFrame to HTML and add styling
    html_table += df.to_html(classes='component-table', 
                            index=False)
    
    # Save to file
    with open('market_evolution_table.html', 'w') as f:
        f.write(html_table)
    
    return html_table

def create_detailed_component_table():
    # Create data with all years, only PC1 and PC2
    data = {
        'Period': ['1996-1999', '2000-2003', '2004-2007', '2008-2011', '2012-2015', '2016-2019', '2020-2022'],
        'PC1 Components (Variance)': [
            'GDP (0.349), Population (0.348), Investment (0.348), Housing Prices (0.342), Income (0.341) [62.35%]',
            'Population (0.327), Investment (0.325), GDP (0.325), Housing Prices (0.313), Housing Starts (0.310) [69.56%]',
            'Construction Prices (0.348), GDP (0.347), Population (0.346), Price Changes (0.332), Unemployment (0.329) [61.87%]',
            'Population (0.376), Income (0.364), Mortgage Rates (0.361), Investment (0.350), Unemployment (0.325) [50.92%]',
            'Population (0.375), GDP (0.375), Unemployment (0.373), Investment (0.373), Housing Prices (0.367) [54.39%]',
            'Unemployment (0.364), Population (0.361), Investment (0.360), Construction Prices (0.359), GDP (0.357) [56.91%]',
            'Construction Prices (0.349), Housing Prices (0.347), GDP (0.344), CPI (0.335), Population (0.333) [61.95%]'
        ],
        'PC2 Components (Variance)': [
            'Mortgage Rates (0.607), Construction Price Changes (0.220) [14.71%]',
            'Construction Prices (0.480), Income (0.291), Mortgage Rates (0.236) [14.61%]',
            'Investment (0.573), Housing Starts (0.366), Housing Prices (0.285) [19.79%]',
            'CPI (0.443), GDP (0.426), Housing Prices (0.407), Construction Prices (0.359) [22.87%]',
            'Mortgage Rates (0.570), Price Changes (0.481), Construction Prices (0.310) [17.47%]',
            'Mortgage Rates (0.525), Housing Starts (0.209) [16.21%]',
            'Housing Starts (0.550), Income (0.442), Mortgage Rates (0.323) [22.89%]'
        ]
    }
    
    df = pd.DataFrame(data)
    
    # Create HTML table with styling
    html_table = """
    <style>
        .component-table {
            font-family: Arial, sans-serif;
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
            font-size: 14px;
        }
        .component-table th {
            background-color: #f2f2f2;
            border: 1px solid #ddd;
            padding: 15px;
            text-align: left;
            font-weight: bold;
            font-size: 16px;
        }
        .component-table td {
            border: 1px solid #ddd;
            padding: 15px;
            text-align: left;
            line-height: 1.4;
        }
        .component-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .table-caption {
            font-weight: bold;
            font-size: 18px;
            margin-bottom: 15px;
            text-align: left;
        }
    </style>
    <div class="table-caption">Table 3: Principal Components Analysis Results - PC1 and PC2 (1996-2022)</div>
    """
    
    # Convert DataFrame to HTML and add styling
    html_table += df.to_html(classes='component-table', 
                            index=False)
    
    # Save to file
    with open('detailed_components_table.html', 'w') as f:
        f.write(html_table)
    
    return html_table

# Create the HTML table
html_output = create_html_component_table()
html_output = create_detailed_component_table()
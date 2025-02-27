# Housing Affordability Analysis (1996-2023)

This repository contains a comprehensive analysis of housing affordability in the United States from 1996 to 2023 using Partial Least Squares (PLS) regression. The analysis examines how various economic and housing market factors influence the Housing Affordability Index (HAI) over time.

## Repository Structure

- **Documentation/**
  - `Housing_Affordability_Analysis_Documentation.md` - Detailed markdown documentation
  - `Housing_Affordability_Analysis_Documentation.html` - Interactive HTML version with embedded visualizations
  - `Documentation_Summary.md` - Overview of all documentation files
  - `PLS_Regression_Methodology_Analysis.md` - Analysis of the PLS regression methodology

- **Code/**
  - `pls_regression_analysis.py` - Main PLS regression model
  - `generate_visualizations.py` - Script to generate all visualizations
  - `process data/` - Scripts for data preprocessing and normalization

- **Data/**
  - `PCA Cleaned Data/` - Normalized data files
  - `PCA Raw Data/` - Original raw data files
  - `PLS Single Analysis/` - Results of the PLS regression analysis

- **Visualizations/**
  - Component scores charts for each time period
  - Component loadings visualization
  - Regression coefficients visualization
  - Annual component scores heatmap

## Key Findings

The analysis reveals that housing affordability is driven by two main components:

1. **Housing Market Dynamics** (mortgage rates, housing starts, residential investment)
2. **Economic Conditions** (GDP, unemployment, prices)

Different time periods show distinct patterns of influence:
- 1996-2000: Pre-Housing Boom
- 2001-2006: Housing Boom
- 2007-2012: Financial Crisis and Recovery
- 2013-2018: Post-Crisis Period
- 2019-2023: Recent Period including COVID-19 and Inflation

The 2022 affordability crisis was unique in showing strong positive scores in both components simultaneously, creating unprecedented pressure on housing affordability.

## Methodology

The analysis uses Partial Least Squares (PLS) regression with 2 components, selected based on the Residual Variance Indicator (RVI) stabilization criterion. The model achieves an R² of 0.3913, explaining 39.13% of the variance in housing affordability.

Variables were selected based on economic theory and organized into three categories:
- **Supply Factors**: Housing starts, PRFI, Vacancy rate
- **Demand Factors**: Population, DPI, Unemployment
- **Market Conditions**: Mortgage rate, MSPUS, CPI, PPI, GDP

All variables were normalized using max absolute value scaling to preserve the direction of change while making them comparable.

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/housing-affordability-analysis.git
cd housing-affordability-analysis

# Install required packages
pip install -r requirements.txt
```

### Running the Analysis

```bash
# Run the main PLS regression analysis
python pls_regression_analysis.py

# Generate visualizations
python generate_visualizations.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data sources: Federal Reserve, Census Bureau, Bureau of Labor Statistics, Bureau of Economic Analysis
- Methodology based on the E3S paper framework for housing market analysis

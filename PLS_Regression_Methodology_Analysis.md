# Analysis of PLS Regression Methodology

## Was PLS Regression the Best Approach?

Partial Least Squares (PLS) regression was an appropriate choice for this housing affordability analysis for several key reasons:

### Advantages of PLS Regression in This Context

1. **Handling Multicollinearity**: Economic and housing variables often exhibit high multicollinearity (correlation between predictors). PLS regression is specifically designed to handle multicollinearity by creating orthogonal components that capture the maximum covariance with the dependent variable (HAI).

2. **Dimensionality Reduction**: With 11 predictor variables, PLS effectively reduced the dimensionality to just 2 components while preserving the most important information related to housing affordability. This simplification makes the model more interpretable.

3. **Focus on Prediction**: Unlike Principal Component Analysis (PCA) which only considers variance in the predictor variables, PLS specifically maximizes the covariance between predictors and the response variable. This makes it more suitable for predictive modeling of housing affordability.

4. **Interpretable Components**: The resulting components have clear interpretations (Housing Market Dynamics and Economic Conditions), which provides valuable insights into the underlying factors affecting housing affordability.

### Potential Alternative Approaches

While PLS regression was appropriate, other methods could have been considered:

1. **Multiple Linear Regression**: Would struggle with the multicollinearity present in economic data.

2. **Ridge or Lasso Regression**: Could handle multicollinearity but wouldn't provide the same level of dimensionality reduction and interpretable components.

3. **Principal Component Regression (PCR)**: Similar to PLS but focuses only on variance in predictors without considering their relationship with the response variable.

4. **Random Forest or Gradient Boosting**: Might achieve higher predictive accuracy but would sacrifice interpretability of the underlying economic relationships.

5. **Vector Autoregression (VAR)**: Could better capture time-series dynamics but would be more complex and potentially overfit with limited data.

The R² value of 0.3913 indicates that the model explains about 39% of the variance in housing affordability. While this might seem modest, it's actually reasonable for complex economic phenomena where many external factors can influence outcomes. The interpretability gained through the PLS approach provides valuable insights that might be lost with black-box models that could achieve marginally higher R² values.

## Why Were Only 2 Components Used?

The decision to use 2 components was based on a systematic approach using the Residual Variance Indicator (RVI):

### Component Selection Process

1. **RVI Calculation**: The code calculates the Residual Variance Indicator for each potential number of components:
   ```python
   def calculate_rvi(X, y, max_components=10):
       n = X.shape[0]
       rvi_values = []
       
       for a in range(1, max_components + 1):
           # Fit PLS with a components
           pls = PLSRegression(n_components=a)
           pls.fit(X, y)
           
           # Calculate residuals
           y_pred = pls.predict(X)
           residuals = y - y_pred
           
           # Calculate RVI
           rvi = np.sum(residuals**2) / (n - a - 1)
           rvi_values.append(rvi)
           
           # Check if RVI has stabilized (less than 5% change)
           if a > 1:
               change = abs(rvi_values[-1] - rvi_values[-2]) / rvi_values[-2]
               if change < 0.05:
                   print(f"RVI stabilized at {a} components (change: {change:.3%})")
                   return a
       
       return len(rvi_values)
   ```

2. **Stabilization Criterion**: The algorithm stops adding components when the RVI stabilizes (change less than 5%), indicating that additional components would not significantly improve the model.

3. **Results**: According to the summary documentation, the RVI values were:
   - Initial RVI (1 component): 0.013019
   - Final RVI (2 components): 0.012683
   - Change: 2.584% (below the 5% threshold)

This indicates that adding a third component would provide minimal additional explanatory power while increasing model complexity.

### Validation of the 2-Component Decision

The 2-component solution is further validated by:

1. **Interpretability**: The two components have clear and meaningful interpretations (Housing Market Dynamics and Economic Conditions).

2. **Significant Loadings**: Both components show multiple variables with significant loadings (|loading| > 0.35), indicating they capture meaningful patterns in the data.

3. **Parsimony**: The principle of parsimony suggests using the simplest model that adequately explains the data. Two components provide a good balance between explanatory power and simplicity.

4. **Variance Explained**: The two components together explain 39.13% of the variance in housing affordability, which is reasonable for economic data with inherent noise and external influences.

## How Were the Variables Selected?

The variable selection process appears to have been based on economic theory, data availability, and a structured approach to categorizing housing market factors:

### Variable Selection Process

1. **Theoretical Framework**: The code references the E3S paper methodology as a theoretical foundation:
   ```python
   # Define variable groups based on the E3S paper methodology but using our available variables
   VARIABLE_GROUPS = {
       'Supply': ['housing_starts', 'prfi', 'vacancy'],  # Housing starts, Private Residential Fixed Investment, Vacancy Rate
       'Demand': ['population', 'dpi', 'unemployment'],  # Population, Disposable Personal Income, Unemployment
       'Market': ['mortgage', 'mspus', 'cpi', 'ppi', 'gdp']  # Mortgage rate, Median Sales Price, CPI, PPI, GDP
   }
   ```

2. **Comprehensive Coverage**: The selected variables cover the major aspects of housing markets:
   - **Supply Factors**: Construction activity (housing starts), investment (PRFI), and availability (vacancy rates)
   - **Demand Factors**: Population growth, income (DPI), and employment conditions (unemployment rate)
   - **Market Conditions**: Financing (mortgage rates), prices (MSPUS), and broader economic indicators (CPI, PPI, GDP)

3. **Data Availability**: The variables were likely chosen based on data availability for the entire study period (1996-2023). This is evident from the data processing scripts in the `process data/` directory.

4. **Normalization Approach**: All variables were normalized using max absolute value scaling to preserve the direction of change while making them comparable:
   ```
   x_normalized = Δx / max(|Δx|)
   ```

### Potential Limitations in Variable Selection

1. **Omitted Variables**: Some potentially relevant factors might be missing, such as:
   - Housing policy changes
   - Demographic shifts beyond population growth
   - Foreign investment in housing markets
   - Regional variations (the analysis appears to be national-level)

2. **Temporal Aggregation**: Using quarterly data may smooth out some short-term fluctuations that could be important.

3. **No Feature Selection Algorithm**: There's no evidence of using algorithmic feature selection methods (like forward selection, backward elimination, or LASSO). The variables appear to have been selected based on domain knowledge.

## Conclusion

The PLS regression approach with 2 components using the selected 11 variables represents a well-reasoned methodology for analyzing housing affordability:

1. **PLS Regression**: Appropriate for handling multicollinearity while maintaining interpretability.

2. **Two Components**: Justified by the RVI stabilization criterion and validated by the clear interpretability of the resulting components.

3. **Variable Selection**: Based on economic theory with comprehensive coverage of supply, demand, and market factors.

The model's R² of 0.3913 indicates reasonable explanatory power for a complex economic phenomenon. The approach successfully balances predictive power with interpretability, providing valuable insights into the drivers of housing affordability over the 27-year study period.

The methodology could potentially be enhanced by:
- Testing additional variables
- Exploring regional variations
- Comparing with alternative modeling approaches
- Incorporating non-linear relationships

However, the current approach provides a solid foundation for understanding housing affordability dynamics and offers clear, interpretable insights into how different economic factors influence affordability over time.

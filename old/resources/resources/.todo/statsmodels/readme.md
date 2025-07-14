# Statsmodels Cheatsheet

## 1. Installing Statsmodels
- pip install statsmodels  # Install Statsmodels

## 2. Importing Libraries
- import statsmodels.api as sm  # Import Statsmodels API
- import pandas as pd  # Import Pandas for data handling

## 3. Loading Data
- df = pd.read_csv('data.csv')  # Load data from a CSV file

## 4. Descriptive Statistics
- df.describe()  # Get summary statistics
- df['column_name'].mean()  # Calculate mean of a column
- df['column_name'].std()  # Calculate standard deviation

## 5. OLS Regression
- X = df[['predictor1', 'predictor2']]  # Define predictors
- y = df['response']  # Define response variable
- X = sm.add_constant(X)  # Add intercept
- model = sm.OLS(y, X).fit()  # Fit OLS model

## 6. Model Summary
- print(model.summary())  # Print model summary

## 7. Making Predictions
- predictions = model.predict(X)  # Make predictions

## 8. Residual Analysis
- residuals = model.resid  # Get residuals
- sm.qqplot(residuals, line='s')  # Q-Q plot for residuals

## 9. Logistic Regression
- model_logit = sm.Logit(y, X).fit()  # Fit logistic regression model
- print(model_logit.summary())  # Print logistic regression summary

## 10. ANOVA
- from statsmodels.formula.api import ols, anova_lm  # Import OLS and ANOVA
- model_anova = ols('response ~ C(group)', data=df).fit()  # Fit ANOVA model
- anova_results = anova_lm(model_anova)  # Perform ANOVA
- print(anova_results)  # Print ANOVA results

## 11. Time Series Analysis
- import statsmodels.tsa.api as tsa  # Import time series analysis
- model_ts = tsa.ARIMA(df['time_series'], order=(p,d,q)).fit()  # Fit ARIMA model

## 12. Forecasting
- forecast = model_ts.forecast(steps=10)  # Forecast next 10 periods

## 13. Handling Multicollinearity
- from statsmodels.stats.outliers_influence import variance_inflation_factor  # Import VIF
- vif = pd.DataFrame()  # Create DataFrame for VIF
- vif['VIF Factor'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]  # Calculate VIF
- vif['features'] = X.columns  # Add feature names

## 14. Model Diagnostics
- sm.graphics.tsa.plot_acf(model.resid)  # ACF plot of residuals
- sm.graphics.tsa.plot_pacf(model.resid)  # PACF plot of residuals

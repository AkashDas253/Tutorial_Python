## **Time Series in Pandas**

### **Creating Time Series Data**  
| Method | Description |  
|--------|-------------|  
| `pd.to_datetime(df['date_column'])` | Converts a column to a DateTime type |  
| `pd.date_range(start, end, freq='D')` | Generates a range of dates with a specified frequency (e.g., daily) |  
| `pd.Timestamp('2025-03-19')` | Creates a single timestamp |  
| `pd.Series(data, index=pd.date_range(...))` | Creates a time series from data with a specific date range as the index |  

---

### **DateTime Indexing and Selection**  
| Method | Description |  
|--------|-------------|  
| `df.set_index('date_column')` | Sets a column as the DataFrame index (date column for time series) |  
| `df.index.year` | Extracts the year from the DateTime index |  
| `df.index.month` | Extracts the month from the DateTime index |  
| `df.index.day` | Extracts the day from the DateTime index |  
| `df.index.weekday` | Extracts the weekday (0-6) from the DateTime index |  
| `df.loc['2025-03-19']` | Selects data for a specific date |  
| `df.loc['2025-03-19':'2025-03-21']` | Selects a range of dates |  

---

### **Resampling Time Series Data**  
| Method | Description |  
|--------|-------------|  
| `df.resample('D').mean()` | Resamples the data at a daily frequency and calculates the mean |  
| `df.resample('M').sum()` | Resamples the data at a monthly frequency and calculates the sum |  
| `df.resample('A').last()` | Resamples the data at an annual frequency and gets the last entry |  
| `df.resample('H').interpolate()` | Resamples at hourly frequency and interpolates missing values |  

---

### **Shifting and Lagging**  
| Method | Description |  
|--------|-------------|  
| `df.shift(1)` | Shifts the time series by one period (e.g., shifts data by one row) |  
| `df.shift(-1)` | Shifts the time series backwards by one period |  
| `df.diff(1)` | Computes the difference between consecutive values |  
| `df.pct_change()` | Calculates the percentage change between consecutive periods |  

---

### **Rolling Window Operations**  
| Method | Description |  
|--------|-------------|  
| `df.rolling(window=3).mean()` | Applies a rolling mean with a window size of 3 |  
| `df.rolling(window=3).sum()` | Applies a rolling sum with a window size of 3 |  
| `df.rolling(window=3).std()` | Applies a rolling standard deviation with a window size of 3 |  
| `df.rolling(window=3).min()` | Applies a rolling minimum with a window size of 3 |  
| `df.rolling(window=3).max()` | Applies a rolling maximum with a window size of 3 |  

---

### **Time Series Plotting**  
| Method | Description |  
|--------|-------------|  
| `df.plot()` | Plots the time series with DateTime index on the x-axis |  
| `df.plot(x='date_column', y='value_column')` | Plots a specific column against the date column |  
| `df.resample('M').sum().plot()` | Plots resampled data (e.g., monthly sum) |  
| `df.plot(kind='line')` | Plots the time series as a line graph |  
| `df.plot(kind='bar')` | Plots the time series as a bar chart |  

---

### **Time Series Decomposition**  
| Method | Description |  
|--------|-------------|  
| `from statsmodels.tsa.seasonal import seasonal_decompose` | Imports seasonal decomposition from Statsmodels |  
| `seasonal_decompose(df['value_column'], model='additive', period=12)` | Decomposes time series into trend, seasonal, and residual components |  
| `seasonal_decompose(df['value_column'], model='multiplicative', period=12)` | Decomposes time series with multiplicative model |  

---

### **Handling Missing Data in Time Series**  
| Method | Description |  
|--------|-------------|  
| `df.interpolate()` | Fills missing values using interpolation |  
| `df.fillna(method='ffill')` | Fills missing values using forward fill |  
| `df.fillna(method='bfill')` | Fills missing values using backward fill |  
| `df.resample('D').ffill()` | Resamples the time series and applies forward fill |  

---

### **Time Zone Handling**  
| Method | Description |  
|--------|-------------|  
| `df.index.tz_localize('UTC')` | Localizes the DateTime index to a specified time zone (e.g., UTC) |  
| `df.index.tz_convert('US/Eastern')` | Converts the DateTime index to a different time zone |  
| `df.index = pd.to_datetime(df.index).tz_localize('UTC').tz_convert('US/Eastern')` | Converts DateTime index with time zone conversion |  

---

### **DateTime Properties**  
| Method | Description |  
|--------|-------------|  
| `df.index.year` | Extracts the year from the DateTime index |  
| `df.index.month` | Extracts the month from the DateTime index |  
| `df.index.dayofweek` | Extracts the day of the week (0-6) from the DateTime index |  
| `df.index.dayofyear` | Extracts the day of the year from the DateTime index |  

---

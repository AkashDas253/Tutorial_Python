### Window Functions in Pandas

Window functions in Pandas allow for the calculation of statistics over a specific window of data within a column or across columns, often used in time series or grouped data analysis. These functions enable operations such as moving averages, cumulative sums, and rankings, applied to subsets of the data.

#### Types of Window Functions

| **Function**                 | **Description**                                                                 |
|------------------------------|---------------------------------------------------------------------------------|
| `rolling()`                  | Provides a rolling window view over a series to perform various aggregation functions (e.g., sum, mean). |
| `expanding()`                | Expanding window that includes all data points up to the current point. Useful for cumulative operations. |
| `ewm()`                      | Exponential weighted functions, allowing for weighting past values with exponentially decaying weights. |
| `shift()`                    | Shifts the data by a specified number of periods, often used for time series analysis. |
| `rank()`                     | Ranks values within a specific window.                                           |


#### Syntax for Window Functions in Pandas

#### 1. **Rolling Window**

The `rolling()` function provides a rolling window view for applying aggregating functions.

```python
# Rolling window function
df['output'] = df['column'].rolling(
    window=<int>,         # Size of the window (number of periods)
    min_periods=<int>,    # Minimum number of observations in the window (default is window size)
    axis=<axis>,          # Axis along which to compute (0 for rows, 1 for columns)
    win_type=<str>,       # Type of window ('boxcar', 'triang', 'blackman', etc.)
    on=<column>,          # For time-based rolling, specify a column name to be used for the window
    closed=<str>          # Defines which side of the window is closed ('right', 'left', 'both', 'neither')
).function(<function>)   # Function to apply (e.g., mean(), sum(), median(), etc.)
```

##### Example:

```python
df['rolling_mean'] = df['column'].rolling(window=3).mean()
```

#### 2. **Expanding Window**

The `expanding()` function calculates cumulative statistics over all data points up to the current point.

```python
# Expanding window function
df['output'] = df['column'].expanding(
    min_periods=<int>,    # Minimum number of observations required in the window
    axis=<axis>           # Axis along which to compute
).function(<function>)   # Function to apply (e.g., sum(), mean(), max(), etc.)
```

##### Example:

```python
df['expanding_sum'] = df['column'].expanding().sum()
```

#### 3. **Exponential Weighted Function**

The `ewm()` function is used for exponential weighting, which assigns more importance to recent values.

```python
# Exponential weighted function
df['output'] = df['column'].ewm(
    span=<float>,            # Decay factor (influences weighting of past values)
    halflife=<float>,        # Half-life period for the exponential decay
    adjust=<bool>,           # Whether to adjust the weights (default is True)
    axis=<axis>              # Axis along which to compute
).function(<function>)      # Function to apply (e.g., mean(), sum(), var(), etc.)
```

##### Example:

```python
df['ewm_mean'] = df['column'].ewm(span=3).mean()
```

#### 4. **Shifting Data**

The `shift()` function shifts the data by a given number of periods forward or backward.

```python
# Shifting data
df['output'] = df['column'].shift(
    periods=<int>,            # Number of periods to shift (positive for forward, negative for backward)
    freq=<str>,               # Frequency string (e.g., 'D' for days, 'H' for hours) (optional)
    axis=<axis>,              # Axis along which to shift (default is 0)
    fill_value=<value>        # Value to use for missing values (default is NaN)
)
```

##### Example:

```python
df['shifted'] = df['column'].shift(1)
```

#### 5. **Ranking**

The `rank()` function assigns ranks to values in the column, with various ranking methods available.

```python
# Ranking data
df['output'] = df['column'].rank(
    method=<str>,             # Ranking method ('average', 'min', 'max', 'first', 'dense')
    axis=<axis>,              # Axis along which to compute (default is 0)
    na_option=<str>,          # Handle NaN values ('keep', 'top', 'bottom')
    ascending=<bool>,         # Whether to rank in ascending or descending order
    pct=<bool>                # Whether to return percentile rank
)
```

##### Example:

```python
df['rank'] = df['column'].rank(method='average')
```

### Common Functions Used with Window Functions

- `mean()`: Compute the average.
- `sum()`: Compute the sum of values.
- `median()`: Compute the median.
- `min()`: Compute the minimum.
- `max()`: Compute the maximum.
- `std()`: Compute the standard deviation.
- `var()`: Compute the variance.
- `count()`: Compute the number of non-null values.
- `cummax()`: Compute the cumulative maximum.
- `cummin()`: Compute the cumulative minimum.
- `cumprod()`: Compute the cumulative product.

### Usage Example for All Window Functions

```python
# Rolling mean
df['rolling_mean'] = df['column'].rolling(window=3).mean()

# Expanding sum
df['expanding_sum'] = df['column'].expanding().sum()

# Exponential weighted mean
df['ewm_mean'] = df['column'].ewm(span=3).mean()

# Shifting data by one period
df['shifted'] = df['column'].shift(1)

# Ranking data
df['rank'] = df['column'].rank(method='average')
```

#### Usage Scenarios

- **Rolling Mean/Median**: Useful for smoothing time series data or financial data.
  - Example: Calculating a 7-day moving average of stock prices.
  
- **Cumulative Sum**: Useful for cumulative counts, sums, or statistics.
  - Example: Tracking cumulative sales over a period.
  
- **Exponential Weighting**: Useful when recent data points are more important than older ones.
  - Example: Exponential smoothing in time series forecasting.
  
- **Shifting**: Helps to compare current data with past data.
  - Example: Comparing daily stock prices to the previous day.

#### Considerations
- **NaN Handling**: Most window functions by default will result in `NaN` for the initial values where the window size cannot be fully applied.
- **Performance**: For large datasets, window functions can be computationally expensive and may require optimization.

---

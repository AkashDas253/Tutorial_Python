## **Rolling Window Functions**

Rolling window functions are essential for performing calculations on a moving window of data, commonly used in time-series analysis, signal processing, and various statistical operations. They allow for calculations such as moving averages, sums, and other statistics over a specific window size.

---

### **`rolling()` – Syntax**

```python
df.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
```

#### **Parameters:**

| Parameter       | Default | Description                                                    |
|-----------------|---------|----------------------------------------------------------------|
| `window`        | Required | Size of the moving window. Defines how many rows to consider for each calculation. |
| `min_periods`   | `None`   | Minimum number of observations in the window required to return a result. If `None`, all data points are considered. |
| `center`        | `False`  | If `True`, sets labels at the center of the window. Otherwise, they are set at the right edge. |
| `win_type`      | `None`   | Type of window function: boxcar, triang, blackman, hamming, etc. Used for smoothing. |
| `on`            | `None`   | Column to use instead of index for rolling window calculations. |
| `axis`          | `0`      | Axis to apply the rolling operation on. `0` for rows, `1` for columns. |
| `closed`        | `None`   | Which side of the window is closed: `'right'`, `'left'`, `'both'`, `'neither'`. |

---

### **Common Rolling Methods**

| Method             | Description                            | Example                             |
|--------------------|----------------------------------------|-------------------------------------|
| `.mean()`          | Computes the rolling mean             | `df.rolling(3).mean()`             |
| `.sum()`           | Computes the rolling sum              | `df.rolling(3).sum()`              |
| `.max()` / `.min()`| Computes rolling max or min values    | `df.rolling(3).max()`              |
| `.std()` / `.var()`| Computes rolling standard deviation or variance | `df.rolling(3).std()`            |
| `.median()`        | Computes the rolling median           | `df.rolling(3).median()`           |
| `.apply(func)`     | Apply custom function over the rolling window | `df.rolling(3).apply(np.ptp)`     |
| `.skew()`          | Computes rolling skewness             | `df.rolling(3).skew()`             |
| `.kurt()`          | Computes rolling kurtosis             | `df.rolling(3).kurt()`             |

---

### **Examples:**

```python
# Rolling Mean with a window of 3
df.rolling(3).mean()

# Rolling Sum with a window of 3
df.rolling(3).sum()

# Rolling Standard Deviation with a window of 5
df.rolling(5).std()

# Rolling Maximum with a window of 4
df.rolling(4).max()

# Applying a custom function (peak-to-peak range) on a rolling window
df.rolling(3).apply(np.ptp)

# Rolling Skewness (a measure of asymmetry in the data) over a window of 3
df.rolling(3).skew()

# Rolling Kurtosis (a measure of the "tailedness" of the data) over a window of 3
df.rolling(3).kurt()
```

---

### **Advanced Rolling Window Operations**

- **Custom Window Types**: The `win_type` parameter allows for specifying different window functions for smoothing the data. Some options include:
  - `boxcar`: A simple moving average.
  - `triang`: A triangular window that gives more weight to values near the center.
  - `blackman`, `hamming`: Window types used in signal processing to reduce the side lobes of the Fourier transform.

  Example:

  ```python
  df.rolling(3, win_type='hamming').mean()
  ```

- **Rolling with `on` parameter**: You can perform rolling operations on a specific column instead of the index. This is especially useful when working with multi-indexed DataFrames or when the rolling window needs to be based on a column rather than the index.

  Example:

  ```python
  df.rolling(3, on='date').mean()
  ```

- **Rolling Window with Custom Function**: Use `.apply()` with custom functions to perform operations not directly available with other rolling methods.

  Example (using `np.ptp` for peak-to-peak):

  ```python
  df.rolling(3).apply(np.ptp)
  ```

---

### **Use Cases for Rolling Window Functions:**

- **Time Series Analysis**: Used for smoothing out short-term fluctuations and highlighting longer-term trends or cycles. Commonly used in stock market data, sensor data, etc.
- **Moving Averages**: Often used in technical analysis for stock prices or in economics for moving averages to track trends.
- **Signal Processing**: In signal processing, rolling windows are used for filtering and noise reduction.
- **Data Smoothing**: Using window functions like Hamming or Blackman to smooth data and reduce noise.

---

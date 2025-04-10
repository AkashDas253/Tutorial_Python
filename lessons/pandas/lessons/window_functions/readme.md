
## **Window Functions in Pandas (Generalized & Complete)**

Window functions in Pandas allow operations across **sliding**, **expanding**, or **weighted** windows. These are used for **trend analysis**, **smoothing**, **time-series**, and **group-based rolling statistics**.

---

###  **Rolling Window Functions**

```python
df.rolling(window, min_periods=None, center=False, win_type=None, on=None, axis=0, closed=None)
```

| Parameter       | Default | Description                                                    |
|----------------|---------|----------------------------------------------------------------|
| `window`        | Required | Size of the moving window                                      |
| `min_periods`   | `None`   | Minimum observations in window required to return a result     |
| `center`        | `False`  | Set labels at center of window                                 |
| `win_type`      | `None`   | Window type: boxcar, triang, blackman, hamming, etc.           |
| `on`            | `None`   | Column to use instead of index                                 |
| `axis`          | `0`      | Axis to perform the rolling on                                 |
| `closed`        | `None`   | Which sides to close: 'right', 'left', 'both', 'neither'       |

**Common Rolling Methods**

| Method             | Description                            | Example                             |
|-------------------|----------------------------------------|-------------------------------------|
| `.mean()`         | Rolling mean                           | `df.rolling(3).mean()`              |
| `.sum()`          | Rolling sum                            | `df.rolling(3).sum()`               |
| `.max()` / `.min()` | Rolling max/min                     | `df.rolling(3).max()`               |
| `.std()` / `.var()` | Rolling std/variance                | `df.rolling(3).std()`               |
| `.median()`       | Rolling median                         | `df.rolling(3).median()`            |
| `.apply(func)`    | Custom rolling logic                   | `df.rolling(3).apply(np.ptp)`       |

---

###  **Expanding Window Functions**

```python
df.expanding(min_periods=1, axis=0, method='single')
```

| Parameter       | Default | Description                                                |
|----------------|---------|------------------------------------------------------------|
| `min_periods`   | `1`      | Minimum observations needed to return a result             |
| `axis`          | `0`      | Axis for operation                                         |
| `method`        | `'single'` | Optimization flag for internal use                        |

**Common Expanding Methods**

| Method             | Description                          | Example                                |
|-------------------|--------------------------------------|----------------------------------------|
| `.sum()`          | Expanding cumulative sum             | `df.expanding().sum()`                 |
| `.mean()`         | Expanding mean                       | `df.expanding().mean()`                |
| `.std()` / `.var()` | Expanding std/variance            | `df.expanding().std()`                 |
| `.count()`        | Count of non-NA values               | `df.expanding().count()`               |
| `.apply(func)`    | Custom expanding logic               | `df.expanding().apply(lambda x: x[-1])`|

---

###  **Exponentially Weighted Window (EWM)**

```python
df.ewm(com=None, span=None, halflife=None, alpha=None, min_periods=0, adjust=True, ignore_na=False, axis=0)
```

| Parameter       | Description                                                                 |
|----------------|-----------------------------------------------------------------------------|
| `com`          | Center of mass (smoothing parameter)                                        |
| `span`         | Smoothness of decay (similar to window size)                                |
| `halflife`     | Half-life of weights                                                        |
| `alpha`        | Smoothing factor (0 < α ≤ 1)                                                 |
| `adjust`       | Whether to divide by weights or not                                         |
| `ignore_na`    | Whether to ignore missing values                                            |
| `min_periods`  | Minimum observations required to return result                              |

**Common EWM Methods**

| Method             | Description                         | Example                              |
|-------------------|-------------------------------------|--------------------------------------|
| `.mean()`         | Exponentially weighted mean         | `df['col'].ewm(span=3).mean()`       |
| `.std()` / `.var()` | EWM std/var                      | `df['col'].ewm(span=3).std()`        |
| `.corr()`         | EWM correlation                     | `df['a'].ewm(span=2).corr(df['b'])`  |

---

###  **Cumulative Functions**

```python
df['col'].cumsum()      # Cumulative sum  
df['col'].cumprod()     # Cumulative product  
df['col'].cummin()      # Cumulative min  
df['col'].cummax()      # Cumulative max  
df['col'].cumcount()    # Cumulative count (on GroupBy object)
```

> These work directly on columns or on grouped data, not via a window object.

---

###  **Group-based Window Functions**

```python
df.groupby('group')['col'].rolling(window=3).mean().reset_index(level=0, drop=True)
```

> Allows performing window operations **within each group** (e.g., per user, per category).

```python
df.groupby('group')['col'].expanding().sum()  
# Expanding operation by group
```

```python
df.groupby('group')['col'].transform(lambda x: x.ewm(span=3).mean())  
# EWM per group
```

---

###  **Centered Windows**

```python
df['col'].rolling(window=3, center=True).mean()  
# Centers the result at the middle of the window
```

---

###  **Custom Functions**

```python
df['range'] = df['col'].rolling(3).apply(lambda x: x.max() - x.min())  
# Rolling range calculation
```

---

###  **Window Type Examples (win_type)**  
Requires `scipy` for certain window types.

```python
df['col'].rolling(window=5, win_type='triang').mean()  
# Triangular weighted rolling average
```

> Other `win_type` values: `'boxcar'`, `'triang'`, `'blackman'`, `'hamming'`, `'bartlett'`, `'parzen'`, `'bohman'`, `'blackmanharris'`, `'nuttall'`, `'barthann'`.

---

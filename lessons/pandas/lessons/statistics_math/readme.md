
## **Statistics & Math Operations in Pandas**

Pandas provides a rich set of statistical and mathematical operations that work on Series and DataFrame objects, enabling quick analysis, transformation, and feature engineering.

---

### **Descriptive Statistics**

| Operation        | Description                          | Syntax / Example                            |
|------------------|--------------------------------------|---------------------------------------------|
| `count()`        | Number of non-null values            | `df.count()`                                |
| `sum()`          | Sum of values                        | `df.sum()`                                  |
| `mean()`         | Arithmetic mean                      | `df.mean()`                                 |
| `median()`       | Median                               | `df.median()`                               |
| `mode()`         | Mode(s)                              | `df.mode()`                                 |
| `std()`          | Standard deviation                   | `df.std()`                                  |
| `var()`          | Variance                             | `df.var()`                                  |
| `min()` / `max()`| Minimum / Maximum                    | `df.min()`, `df.max()`                      |
| `idxmin()`       | Index of minimum value               | `df.idxmin()`                               |
| `idxmax()`       | Index of maximum value               | `df.idxmax()`                               |
| `quantile(q)`    | Value at quantile `q` (0 ≤ q ≤ 1)    | `df.quantile(0.25)`                         |
| `describe()`     | Summary statistics                   | `df.describe()`                             |
| `skew()`         | Skewness                             | `df.skew()`                                 |
| `kurt()`         | Kurtosis                             | `df.kurt()`                                 |
| `sem()`          | Standard error of the mean           | `df.sem()`                                  |
| `mad()`          | Mean absolute deviation              | `df.mad()` *(deprecated)*                   |

---

### **Cumulative & Expanding**

| Operation        | Description                          | Syntax / Example                            |
|------------------|--------------------------------------|---------------------------------------------|
| `cumsum()`       | Cumulative sum                       | `df.cumsum()`                               |
| `cumprod()`      | Cumulative product                   | `df.cumprod()`                              |
| `cummax()`       | Cumulative max                       | `df.cummax()`                               |
| `cummin()`       | Cumulative min                       | `df.cummin()`                               |

---

### **Correlation & Covariance**

> Operate between columns in DataFrame (numerical only)

| Operation        | Description                          | Syntax / Example                            |
|------------------|--------------------------------------|---------------------------------------------|
| `corr()`         | Pearson correlation coefficient      | `df.corr()`                                 |
| `cov()`          | Covariance matrix                    | `df.cov()`                                  |
| `corrwith(other)`| Correlation with another DataFrame/Series | `df.corrwith(df2)`                    |

---

### **Mathematical Operations (Element-wise)**

These are **vectorized operations** performed element-wise across Series or DataFrames.

| Operation        | Description                          | Example                                     |
|------------------|--------------------------------------|---------------------------------------------|
| `abs()`          | Absolute value                       | `df.abs()`                                  |
| `round(decimals)`| Round to decimals                    | `df.round(2)`                               |
| `clip(lower, upper)` | Limit values between bounds     | `df.clip(0, 1)`                             |
| `apply(np.log)`  | Apply NumPy function                 | `df.apply(np.log)`                          |
| `pow(n)`         | Raise each element to power `n`      | `df.pow(2)`                                 |

---

### **Boolean/Comparison Operations**

| Operation        | Description                          | Example                                     |
|------------------|--------------------------------------|---------------------------------------------|
| `df > 10`        | Element-wise comparison              | Returns boolean DataFrame                   |
| `(df > 10).sum()`| Count of True per column             | Useful for condition count                  |
| `(df == df2)`    | Element-wise equality                | Boolean DataFrame                           |

---

### **Aggregation**

> Can be applied using `.agg()` and `.aggregate()` on Series or DataFrame.

```python
df.agg(['mean', 'sum'])             # Multiple aggregations on columns  
df.agg({'col1': 'mean', 'col2': 'sum'})  # Specific functions per column
```

Custom functions:

```python
df.aggregate(lambda x: x.max() - x.min())  # Range
```

---

### **Axis Behavior**

All operations can be applied **across rows or columns**:

| Axis | Value | Meaning               |
|------|-------|-----------------------|
| `0`  | `index` | Down the rows (column-wise) |
| `1`  | `columns` | Across columns (row-wise)  |

**Example:**

```python
df.sum(axis=0)  # Sum of each column  
df.sum(axis=1)  # Sum of each row
```

---

### **Missing Data Friendly**

Most operations **skip `NaN` values by default**. To change this:

```python
df.sum(skipna=False)
```

---

### **Value Counts & Unique Analysis**

| Method            | Description                        | Example                  |
|------------------|------------------------------------|--------------------------|
| `value_counts()` | Count unique values (Series only) | `s.value_counts()`       |
| `unique()`       | Return unique values               | `s.unique()`             |
| `nunique()`      | Count of unique values             | `s.nunique()`            |
| `isin(values)`   | Check if in list                   | `s.isin(['a', 'b'])`     |

---

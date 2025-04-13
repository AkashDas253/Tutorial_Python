
## Aggregation & Grouping in Pandas

Aggregation allows you to compute **summary statistics** (like mean, sum, count) across **groups** of data based on categorical columns.

---

### Basic GroupBy Syntax

```python
df.groupby(
    by=None,                    # Column(s) or keys to group by
    axis=0,                     # Group rows (0) or columns (1)
    level=None,                 # Level(s) in MultiIndex to group by
    as_index=True,              # Return grouped keys as index
    sort=True,                  # Sort group keys
    group_keys=True,            # Include group keys in result
    observed=False,             # For categorical groupers
    dropna=True                 # Drop NA group keys
)
```

**Example:**

```python
grouped = df.groupby("category")
```

---

## Aggregation Methods

You can apply **built-in aggregation functions** directly on grouped objects.

### Common Aggregation Functions

| Function     | Description                | Syntax Example                   |
|--------------|----------------------------|----------------------------------|
| `mean()`     | Mean of each group         | `grouped.mean()`                |
| `sum()`      | Sum of each group          | `grouped.sum()`                 |
| `count()`    | Non-NA count               | `grouped.count()`               |
| `size()`     | Total group size           | `grouped.size()`                |
| `min()`      | Minimum value              | `grouped.min()`                 |
| `max()`      | Maximum value              | `grouped.max()`                 |
| `median()`   | Median                     | `grouped.median()`              |
| `std()`      | Standard deviation         | `grouped.std()`                 |
| `var()`      | Variance                   | `grouped.var()`                 |
| `first()`    | First non-null entry       | `grouped.first()`               |
| `last()`     | Last non-null entry        | `grouped.last()`                |
| `nth(n)`     | Nth entry in each group    | `grouped.nth(2)`                |

---

### Aggregating Multiple Columns

```python
df.groupby("category")[["col1", "col2"]].mean()
```

---

### Using `.agg()` for Custom Aggregation

```python
grouped.agg(
    func=None                      # Single function or list/dict of functions
)
```

**Examples:**

```python
df.groupby("category").agg("mean")
df.groupby("category").agg(["sum", "mean"])
df.groupby("category").agg({"col1": "mean", "col2": "sum"})
```

---

### Applying Custom Functions with `.apply()`

```python
df.groupby("category").apply(lambda x: x["value"].max() - x["value"].min())
```

---

### GroupBy on Multiple Columns

```python
df.groupby(["region", "category"]).sum()
```

---

### Grouping by Index Level

```python
df.groupby(level=0).mean()
df.groupby(level=["year", "month"]).sum()
```

---

### GroupBy with `as_index=False`

Returns a DataFrame with group keys as columns, not as index.

```python
df.groupby("category", as_index=False).sum()
```

---

### Filtering Groups

Use `.filter()` to include/exclude entire groups.

```python
df.groupby("category").filter(lambda x: len(x) > 2)
```

---

### Transforming Grouped Data

`transform()` returns a **DataFrame with the same shape** as original but with grouped transformation applied.

```python
df["zscore"] = df.groupby("group")["value"].transform(
    lambda x: (x - x.mean()) / x.std()
)
```

---

### Grouping with Categorical Data

Use `observed=True` for performance and cleaner results.

```python
df.groupby("cat_column", observed=True).mean()
```

---

## Summary of GroupBy Operations

| Method      | Returns                        | Shape               | Purpose                          |
|-------------|--------------------------------|----------------------|----------------------------------|
| `agg()`     | Aggregated results             | Smaller              | Multiple metrics per group       |
| `apply()`   | Custom logic on groups         | Variable             | Complex group-wise operations    |
| `transform()` | Same shape as original        | Same                 | Group-wise transformation        |
| `filter()`  | Subset of original DataFrame   | Smaller or same      | Keep/discard groups              |

---


## **Aggregation and Grouping in Pandas**

### **GroupBy Operations**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col')` | Groups the DataFrame by the specified column |  
| `df.groupby(['col1', 'col2'])` | Groups the DataFrame by multiple columns |  
| `df.groupby('col').size()` | Counts the number of records in each group |  
| `df.groupby('col').count()` | Returns the count of non-null values for each group |  
| `df.groupby('col').sum()` | Returns the sum of numeric values for each group |  
| `df.groupby('col').mean()` | Returns the mean of numeric values for each group |  
| `df.groupby('col').median()` | Returns the median of numeric values for each group |  
| `df.groupby('col').min()` | Returns the minimum value for each group |  
| `df.groupby('col').max()` | Returns the maximum value for each group |  
| `df.groupby('col').std()` | Returns the standard deviation for each group |  
| `df.groupby('col').var()` | Returns the variance for each group |  
| `df.groupby('col').agg({'col1': 'sum', 'col2': 'mean'})` | Applies multiple aggregation functions on different columns |  

---

### **Aggregation Functions**  
| Method | Description |  
|--------|-------------|  
| `sum()` | Returns the sum of values |  
| `mean()` | Returns the mean of values |  
| `median()` | Returns the median of values |  
| `std()` | Returns the standard deviation |  
| `var()` | Returns the variance |  
| `min()` | Returns the minimum value |  
| `max()` | Returns the maximum value |  
| `count()` | Returns the number of non-null values |  
| `first()` | Returns the first value in the group |  
| `last()` | Returns the last value in the group |  
| `prod()` | Returns the product of values |  
| `cumsum()` | Returns the cumulative sum |  
| `cumprod()` | Returns the cumulative product |  

---

### **Aggregation with Multiple Functions**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col').agg(['sum', 'mean'])` | Applies multiple functions to all columns |  
| `df.groupby('col').agg({'col1': ['sum', 'min'], 'col2': 'mean'})` | Applies different functions to specific columns |  

---

### **Custom Aggregation Functions**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col').agg(lambda x: x.max() - x.min())` | Applies a custom function (e.g., range) to each group |  
| `df.groupby('col').agg(np.mean)` | Uses a custom aggregation function from `numpy` (e.g., `mean`) |  

---

### **Transformation within Groups**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col').transform(lambda x: x - x.mean())` | Performs transformations like subtracting group mean |  
| `df.groupby('col').transform('sum')` | Applies a transformation that returns the sum for each group |  

---

### **Reshaping Grouped Data**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col').unstack()` | Converts the grouped data to a DataFrame by unstacking |  
| `df.groupby('col').stack()` | Converts the grouped data to a Series by stacking |  

---

### **Handling Missing Data in GroupBy**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col').sum(min_count=1)` | Ignores missing data in the sum operation when `min_count` is set to 1 |  
| `df.groupby('col').fillna(method='ffill')` | Fills missing values in groups using forward fill |  

---

### **GroupBy and Window Functions**  
| Method | Description |  
|--------|-------------|  
| `df.groupby('col').rolling(3).mean()` | Applies a rolling window function after grouping |  
| `df.groupby('col').expanding().mean()` | Applies an expanding window function after grouping |  

---

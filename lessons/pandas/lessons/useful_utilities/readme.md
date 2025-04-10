
## Useful Utilities

Pandas provides general-purpose utilities that enhance **data inspection, transformation, and optimization** workflows. Below is a categorized and complete reference.

---

### 1. Data Copy & Conversion

```python
df.copy(deep=True)                    # Deep copy (default)
df.T                                  # Transpose DataFrame
df.values                             # Convert to NumPy array (legacy)
df.to_numpy()                         # Preferred way to convert to NumPy
```

---

### 2. Structure & Size

```python
df.shape                              # (rows, columns)
df.size                               # Total elements
df.ndim                               # Number of dimensions
df.axes                               # List of axis labels
```

---

### 3. Summary & Info

```python
df.info()                             # Summary of structure
df.describe()                         # Summary statistics
```

---

### 4. Column Membership

```python
"col" in df.columns                   # Check column existence
df.columns.isin(["col1", "col2"])     # Check multiple columns
```

---

### 5. Null Checks

```python
df.isna()                             # Detect missing (NaN)
df.notna()                            # Detect non-missing
pd.isnull(df["col"])                  # Alias of isna()
pd.notnull(df["col"])                 # Alias of notna()
```

---

### 6. Duplicate Handling

```python
df.duplicated(subset=None, keep="first")
df.drop_duplicates(subset=None, keep="first", inplace=False)
```

---

### 7. Rename & Assign Columns

```python
df.rename(columns={"old": "new"}, index={0: "row1"}, inplace=False)
df.assign(new_col=df["col1"] * 2)     # Add or overwrite columns
```

---

### 8. Data Type Utilities

```python
df["col"].astype("float")             # Convert dtype
df.convert_dtypes()                   # Auto convert to optimal dtypes

pd.api.types.is_numeric_dtype(df["col"])
pd.api.types.is_string_dtype(df["col"])
pd.api.types.is_datetime64_any_dtype(df["col"])
```

---

### 9. Memory Usage

```python
df.memory_usage(index=True, deep=False)
```

---

### 10. Value Distribution

```python
df["col"].value_counts(normalize=False, sort=True, dropna=True)
df["col"].unique()                    # Unique values
df["col"].nunique()                   # Number of unique values
```

---

### 11. Binning

```python
pd.cut(df["age"], bins=[0, 18, 35, 60, 100], labels=["Teen", "Young", "Adult", "Senior"])
pd.qcut(df["income"], q=4)           # Quantile-based binning
```

---

### 12. Clipping & Rounding

```python
df["col"].clip(lower=0, upper=100)   # Limit range
df.round(decimals=2)                 # Round all numeric columns
df["col"].round(0)                   # Round single column
```

---

### 13. Evaluate Expressions

```python
df.eval('new_col = col1 + col2')     # Inline expression with variables
```

---

### 14. Conditional Logic

```python
import numpy as np
np.where(condition, val_if_true, val_if_false)
```

---

### 15. Pandas Options

```python
pd.set_option("display.max_columns", 100)
pd.get_option("display.max_rows")
pd.reset_option("display.max_columns")
```

---

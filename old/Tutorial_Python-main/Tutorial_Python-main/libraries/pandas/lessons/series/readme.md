# **Pandas Series**

---

## **Definition**
A `Series` is a one-dimensional, labeled, homogeneous array-like structure in pandas, ideal for storing a single column of data or any list-like structure with an index.

---

## **Properties**

| Property      | Description                                                    |
|---------------|----------------------------------------------------------------|
| 1D structure  | Holds data in a single dimension (like a column)              |
| Mutable	|    The contents of a Series can be modified, and elements can be updated.  |
| Indexed       | Every value is paired with a label (index)                    |
| Homogeneous   | All elements are of the same type                             |
| NaN support   | Allows missing data (`NaN`, `None`)                           |
| Vectorized    | Fast operations using NumPy backend                           |

---

## **Declaration Syntax**

```python
pd.Series(
    data=None,         # Input data (list, array, dict, scalar)
    index=None,        # Optional custom index labels
    dtype=None,        # Optional data type
    name=None,         # Optional Series name
    copy=False,        # Copy data if True
    fastpath=False     # Internal use (skip safety checks)
)
```

### Basic Operations on Series

| **Operation**             | **Description**                                |
|---------------------------|------------------------------------------------|
| **Viewing Data**           | Use `.head()` to view the first few elements. |
| **Indexing**               | Access elements using `[]` with index labels. |
| **Slicing**                | Slice a `Series` using `.iloc[]` or `.loc[]`. |
| **Arithmetic Operations**  | Perform element-wise arithmetic operations. |
| **Handling Missing Data**  | Use `.isnull()`, `.dropna()`, and `.fillna()` for handling missing values. |
| **Element-wise Operations**| Perform operations on elements like `+`, `-`, `*`, etc. |
| **Aggregation**            | Use `.sum()`, `.mean()`, `.min()`, `.max()`, etc. for aggregations. |

### Creating a Series  

| Method | Description |  
|--------|-------------|  
| `pd.Series(data)` | Creates a Series from a list, tuple, dictionary, scalar, or NumPy array |  
| `pd.Series(data, index=index_list)` | Creates a Series with a custom index |  
| `pd.Series(data, dtype=type)` | Specifies the data type of elements |  

---

## **All Features and Operations**

---

### **Attributes (Read-only Metadata)**

| Syntax        | Description                          |
|---------------|--------------------------------------|
| `s.index`     | Index labels                         |
| `s.values`    | Underlying NumPy array               |
| `s.dtype`     | Data type of the Series              |
| `s.dtypes`    | Alias for `dtype` (same output)      |
| `s.name`      | Name of the Series                   |
| `s.size`      | Number of elements                   |
| `s.shape`     | Shape tuple (length,)                |
| `s.ndim`      | Number of dimensions (always 1)      |
| `s.empty`     | `True` if Series is empty            |

---

### **Indexing & Access**

| Syntax                     | Description                     |
|----------------------------|---------------------------------|
| `s[i]`, `s[1:4]`           | Basic index slicing             |
| `s['label']`               | Access by index label           |
| `s.loc['label']`           | Label-based access              |
| `s.iloc[0]`                | Position-based access           |
| `s.at['label']`            | Fast scalar access by label     |
| `s.iat[0]`                 | Fast scalar access by position  |

---

### **Arithmetic Operations**

| Syntax                                                     | Description                      |
|------------------------------------------------------------|----------------------------------|
| `s.add(other, level=None, fill_value=None)`                | Element-wise addition            |
| `s.sub(other, level=None, fill_value=None)`                | Element-wise subtraction         |
| `s.mul(other, level=None, fill_value=None)`                | Element-wise multiplication      |
| `s.div(other, level=None, fill_value=None)`                | Element-wise division            |
| `s.mod(other, level=None, fill_value=None)`                | Modulus operation                |
| `s.pow(other, level=None, fill_value=None)`                | Power/exponentiation             |
| `s.abs()`                                                  | Absolute value                   |

---

### **Vectorized Operations**

| Operation | Description |  
|-----------|-------------|  
| `s + n` | Adds `n` to each element |  
| `s - n` | Subtracts `n` from each element |  
| `s * n` | Multiplies each element by `n` |  
| `s / n` | Divides each element by `n` |  
| `s ** n` | Raises each element to the power of `n` |  
| `s % n` | Returns the remainder after division | 

---

### **Checking Conditions** 

| Method | Description |  
|--------|-------------|  
| `s > n` | Returns `True` for elements greater than `n` |  
| `s < n` | Returns `True` for elements less than `n` |  
| `s == n` | Returns `True` for elements equal to `n` |  
| `s[s > n]` | Filters elements greater than `n` |  

---

### **Aggregation & Statistics**

| Syntax                                | Description                    |
|---------------------------------------|--------------------------------|
| `s.sum(skipna=True)`                  | Sum of elements                |
| `s.mean(skipna=True)`                 | Mean (average)                 |
| `s.median(skipna=True)`              | Median                         |
| `s.min(skipna=True)`                 | Minimum value                  |
| `s.max(skipna=True)`                 | Maximum value                  |
| `s.std(skipna=True)`                 | Standard deviation             |
| `s.var(skipna=True)`                 | Variance                       |
| `s.mode(dropna=True)`                | Most frequent values           |
| `s.describe(percentiles=None)`       | Summary statistics             |

---

### **Value Checks & Insights**

| Syntax                             | Description                          |
|------------------------------------|--------------------------------------|
| `s.unique()`                       | Unique values                        |
| `s.nunique(dropna=True)`           | Number of unique values              |
| `s.value_counts(normalize=False)`  | Frequency count                      |
| `s.isin([val1, val2])`             | Check if values in list              |
| `s.duplicated(keep='first')`       | Detect duplicate entries             |

---

### **Applying Functions**

| Method | Description |  
|--------|-------------|  
| `s.apply(func)` | Applies a function to each element |  
| `s.map(func)` | Applies a function element-wise |  
| `s.apply(lambda x: x * 2)` | Example of applying a lambda function |  

---

### **Handling Missing Data**

| Syntax                                              | Description                         |
|-----------------------------------------------------|-------------------------------------|
| `s.isna()`                                          | Boolean mask of missing values      |
| `s.notna()`                                         | Boolean mask of non-missing values  |
| `s.fillna(value, method=None, limit=None)`          | Replace missing with a value/method |
| `s.dropna()`                                        | Remove missing entries              |
| `s.replace(to_replace, value=None)`                 | Replace matching values             |

---

### **Combining Multiple Series**  

| Method | Description |  
|--------|-------------|  
| `s1.add(s2)` | Adds two Series element-wise |  
| `s1.sub(s2)` | Subtracts two Series element-wise |  
| `s1.mul(s2)` | Multiplies two Series element-wise |  
| `s1.div(s2)` | Divides two Series element-wise |  

---

### **String Methods (`.str`)**

| Syntax                             | Description                        |
|------------------------------------|------------------------------------|
| `s.str.lower()`                    | Convert strings to lowercase       |
| `s.str.upper()`                    | Convert strings to uppercase       |
| `s.str.contains('pattern')`        | Check if string contains pattern   |
| `s.str.len()`                      | Length of each string              |
| `s.str.replace('a', 'b')`          | Replace substrings                 |
| `s.str.strip()`                    | Remove leading/trailing spaces     |
| `s.str.replace('old', 'new')` | Replaces a substring |  
| `s.str.split('delimiter')` | Splits elements based on a delimiter |  

---

### **Date/Time Methods (`.dt`)**

| Syntax           | Description                     |
|------------------|---------------------------------|
| `s.dt.year`      | Extract year                    |
| `s.dt.month`     | Extract month                   |
| `s.dt.day`       | Extract day                     |
| `s.dt.hour`      | Extract hour                    |
| `s.dt.minute`    | Extract minute                  |
| `s.dt.weekday`   | Day of week (0 = Monday)        |

> **Note:** Only works if `s` is of datetime type.

---

### **Conversion & Casting**

| Syntax                                  | Description                        |
|-----------------------------------------|------------------------------------|
| `s.astype(dtype, copy=True)`            | Change data type                   |
| `s.to_numpy(dtype=None)`                | Convert to NumPy array             |
| `s.to_list()`                           | Convert to Python list             |
| `s.to_frame(name=None)`                 | Convert Series to single-column DataFrame |
| `s.to_dict()` | Converts to a dictionary |

---

### **Sorting, Copying, and Reshaping**

| Syntax                                       | Description                      |
|----------------------------------------------|----------------------------------|
| `s.sort_values(ascending=True)`              | Sort values                      |
| `s.sort_index(ascending=True)`               | Sort by index                    |
| `s.reset_index(drop=False, inplace=False)`   | Reset index                      |
| `s.rename(name)`                             | Rename the Series                |
| `s.copy(deep=True)`                          | Deep copy of Series              |

---

### Additional Methods  

| Method | Description |  
|--------|-------------|  
| `s.idxmax()` | Returns the index of the maximum value |  
| `s.idxmin()` | Returns the index of the minimum value |  
| `s.value_counts()` | Returns a count of unique values |  
| `s.nunique()` | Returns the number of unique values |  
| `s.clip(lower, upper)` | Limits values between `lower` and `upper` |  
| `s.diff()` | Computes the difference between consecutive elements |  
| `s.pct_change()` | Computes percentage change between elements |  
| `s.cumsum()` | Cumulative sum of elements |  
| `s.cumprod()` | Cumulative product of elements |  
| `s.rolling(window).mean()` | Rolling mean over a window | 

---

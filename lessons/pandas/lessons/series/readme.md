
# **Pandas Series**

---

## **Definition**
A `Series` is a one-dimensional, labeled, homogeneous array-like structure in pandas, ideal for storing a single column of data or any list-like structure with an index.

---

## **Properties**

| Property      | Description                                                    |
|---------------|----------------------------------------------------------------|
| 1D structure  | Holds data in a single dimension (like a column)              |
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

---

## **All Features by Category with Syntax**

---

### **1. Attributes (Read-only Metadata)**

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

### **2. Indexing & Access**

| Syntax                     | Description                     |
|----------------------------|---------------------------------|
| `s[i]`, `s[1:4]`           | Basic index slicing             |
| `s['label']`               | Access by index label           |
| `s.loc['label']`           | Label-based access              |
| `s.iloc[0]`                | Position-based access           |
| `s.at['label']`            | Fast scalar access by label     |
| `s.iat[0]`                 | Fast scalar access by position  |

---

### **3. Arithmetic Operations**

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

### **4. Aggregation & Statistics**

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

### **5. Value Checks & Insights**

| Syntax                             | Description                          |
|------------------------------------|--------------------------------------|
| `s.unique()`                       | Unique values                        |
| `s.nunique(dropna=True)`           | Number of unique values              |
| `s.value_counts(normalize=False)`  | Frequency count                      |
| `s.isin([val1, val2])`             | Check if values in list              |
| `s.duplicated(keep='first')`       | Detect duplicate entries             |

---

### **6. Handling Missing Data**

| Syntax                                              | Description                         |
|-----------------------------------------------------|-------------------------------------|
| `s.isna()`                                          | Boolean mask of missing values      |
| `s.notna()`                                         | Boolean mask of non-missing values  |
| `s.fillna(value, method=None, limit=None)`          | Replace missing with a value/method |
| `s.dropna()`                                        | Remove missing entries              |
| `s.replace(to_replace, value=None)`                 | Replace matching values             |

---

### **7. String Methods (`.str`)**

| Syntax                             | Description                        |
|------------------------------------|------------------------------------|
| `s.str.lower()`                    | Convert strings to lowercase       |
| `s.str.upper()`                    | Convert strings to uppercase       |
| `s.str.contains('pattern')`        | Check if string contains pattern   |
| `s.str.len()`                      | Length of each string              |
| `s.str.replace('a', 'b')`          | Replace substrings                 |
| `s.str.strip()`                    | Remove leading/trailing spaces     |

---

### **8. Date/Time Methods (`.dt`)**

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

### **9. Conversion & Casting**

| Syntax                                  | Description                        |
|-----------------------------------------|------------------------------------|
| `s.astype(dtype, copy=True)`            | Change data type                   |
| `s.to_numpy(dtype=None)`                | Convert to NumPy array             |
| `s.to_list()`                           | Convert to Python list             |
| `s.to_frame(name=None)`                 | Convert Series to single-column DataFrame |

---

### **10. Sorting, Copying, and Reshaping**

| Syntax                                       | Description                      |
|----------------------------------------------|----------------------------------|
| `s.sort_values(ascending=True)`              | Sort values                      |
| `s.sort_index(ascending=True)`               | Sort by index                    |
| `s.reset_index(drop=False, inplace=False)`   | Reset index                      |
| `s.rename(name)`                             | Rename the Series                |
| `s.copy(deep=True)`                          | Deep copy of Series              |

---

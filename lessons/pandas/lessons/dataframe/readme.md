
# **Pandas DataFrame**

---

## **Definition**
A `DataFrame` is a **2-dimensional labeled data structure** with columns of potentially different types. It is conceptually similar to a spreadsheet or SQL table and is the most commonly used data structure in pandas.

---

## **Properties**

| Property         | Description                                                            |
|------------------|------------------------------------------------------------------------|
| 2D structure      | Holds data in rows and columns (like a table)                          |
| Labeled axes      | Both rows (index) and columns are labeled                              |
| Mixed types       | Columns can have different data types                                  |
| NaN support       | Allows missing data                                                    |
| Size-flexible     | Can grow/shrink dynamically                                            |
| NumPy-powered     | Operations are fast and vectorized using NumPy                         |

---

## **Declaration Syntax**

```python
pd.DataFrame(
    data=None,              # Dict, 2D array, Series, scalar, or list of dicts
    index=None,             # Optional row labels
    columns=None,           # Optional column labels
    dtype=None,             # Optional data type for all columns
    copy=False              # Copy data if True
)
```

---

## **All Features by Category with Syntax**

---

### **1. Attributes (Read-only Metadata)**

| Syntax             | Description                               |
|--------------------|-------------------------------------------|
| `df.index`         | Row index labels                          |
| `df.columns`       | Column labels                             |
| `df.dtypes`        | Data type of each column                  |
| `df.shape`         | Tuple: (rows, columns)                    |
| `df.size`          | Total number of elements                  |
| `df.ndim`          | Number of dimensions (always 2)           |
| `df.values`        | Numpy array of values                     |
| `df.empty`         | `True` if DataFrame is empty              |

---

### **2. Data Selection and Access**

| Syntax                      | Description                          |
|-----------------------------|--------------------------------------|
| `df[col]`                   | Select single column as Series       |
| `df[[col1, col2]]`          | Select multiple columns              |
| `df.loc[row_label, col]`    | Label-based access                   |
| `df.iloc[row_idx, col]`     | Position-based access                |
| `df.at[row, col]`           | Fast scalar access by label          |
| `df.iat[row, col]`          | Fast scalar access by position       |
| `df.get(col)`               | Safe column access                   |

---

### **3. Arithmetic & Operations**

| Syntax                                   | Description                          |
|------------------------------------------|--------------------------------------|
| `df.add(other, fill_value=0)`            | Add DataFrame or scalar              |
| `df.sub(other, fill_value=0)`            | Subtract                             |
| `df.mul(other, fill_value=1)`            | Multiply                             |
| `df.div(other, fill_value=1)`            | Divide                               |
| `df.dot(other)`                          | Matrix multiplication                |
| `df.abs()`                               | Absolute values                      |
| `df.clip(lower, upper)`                  | Clip values to bounds                |

---

### **4. Aggregation & Statistics**

| Syntax                        | Description                            |
|-------------------------------|----------------------------------------|
| `df.sum(axis=0)`              | Sum along axis                         |
| `df.mean(axis=0)`             | Mean                                   |
| `df.median(axis=0)`           | Median                                 |
| `df.min(axis=0)`              | Minimum                                |
| `df.max(axis=0)`              | Maximum                                |
| `df.std(axis=0)`              | Standard deviation                     |
| `df.var(axis=0)`              | Variance                               |
| `df.mode()`                   | Most frequent value                    |
| `df.describe()`               | Summary statistics                     |
| `df.count()`                  | Count non-NA cells                     |
| `df.cumsum()`                 | Cumulative sum                         |
| `df.cumprod()`                | Cumulative product                     |

---

### **5. Missing Data Handling**

| Syntax                                   | Description                            |
|------------------------------------------|----------------------------------------|
| `df.isna()`                              | Detect missing values                  |
| `df.notna()`                             | Detect non-missing values              |
| `df.fillna(value, method)`               | Fill missing values                    |
| `df.dropna(axis=0, how='any')`           | Drop rows/columns with NaN             |
| `df.replace(to_replace, value)`          | Replace values                         |

---

### **6. Column & Row Manipulation**

| Syntax                                       | Description                            |
|----------------------------------------------|----------------------------------------|
| `df.rename(columns={'a':'A'})`               | Rename columns                         |
| `df.drop(columns=['col'])`                   | Drop column(s)                         |
| `df.drop(index=[0,1])`                       | Drop row(s)                            |
| `df.insert(loc, column, value)`              | Insert column at specific position     |
| `df.assign(newcol=values)`                   | Add new column                         |
| `df.pop(col)`                                | Remove column and return it            |

---

### **7. Sorting & Reindexing**

| Syntax                                      | Description                            |
|---------------------------------------------|----------------------------------------|
| `df.sort_values(by='col', ascending=True)`  | Sort by column                         |
| `df.sort_index(axis=0)`                     | Sort by index                          |
| `df.reset_index(drop=True)`                 | Reset index                            |
| `df.set_index('col')`                       | Set column as index                    |
| `df.reindex(new_index)`                     | Reindex rows                           |

---

### **8. Filtering & Querying**

| Syntax                                | Description                            |
|---------------------------------------|----------------------------------------|
| `df[df['col'] > 5]`                   | Filter rows with condition             |
| `df.query('col > 5')`                 | SQL-like querying                      |
| `df.where(condition)`                 | Keep if condition is True              |
| `df.mask(condition)`                  | Replace where condition is True        |

---

### **9. String Operations (`.str`)**

| Syntax                         | Description                          |
|--------------------------------|--------------------------------------|
| `df['col'].str.lower()`        | Convert to lowercase                 |
| `df['col'].str.contains('x')`  | Check pattern match                  |
| `df['col'].str.replace('a','b')`| Replace substrings                  |
| `df['col'].str.len()`          | Length of each string                |

---

### **10. Date/Time Operations (`.dt`)**

| Syntax                | Description                              |
|------------------------|------------------------------------------|
| `df['date'].dt.year`   | Extract year                            |
| `df['date'].dt.month`  | Extract month                           |
| `df['date'].dt.day`    | Extract day                             |
| `df['date'].dt.weekday`| Day of week                             |

> **Note**: Works only for datetime-type columns.

---

### **11. Type Conversion**

| Syntax                          | Description                           |
|----------------------------------|---------------------------------------|
| `df.astype({'col': type})`       | Convert column type                   |
| `df.convert_dtypes()`            | Convert to best possible types        |
| `df.to_numpy()`                  | Convert to NumPy array                |
| `df.to_dict()`                   | Convert to Python dictionary          |
| `df.to_list()`                   | Convert to list of rows               |
| `df.to_csv('file.csv')`          | Export to CSV                         |
| `df.to_excel('file.xlsx')`       | Export to Excel                       |

---

### **12. Copying & Transposing**

| Syntax                   | Description                     |
|--------------------------|---------------------------------|
| `df.copy(deep=True)`     | Deep copy of DataFrame          |
| `df.T`                   | Transpose rows and columns      |

---


# Pandas DataFrame

---

## Definition
A `DataFrame` is a 2-dimensional labeled data structure with columns of potentially different types. It is conceptually similar to a spreadsheet or SQL table and is the most commonly used data structure in pandas.

---

## Properties

| Property         | Description                                                            |
|------------------|------------------------------------------------------------------------|
| 2D structure      | Holds data in rows and columns (like a table)                          |
| Mutable             | Both the size and contents of a `DataFrame` can be changed. |
| Labeled axes      | Both rows (index) and columns are labeled                              |
| Mixed types       | Columns can have different data types                                  |
| NaN support       | Allows missing data                                                    |
| Size-flexible     | Can grow/shrink dynamically                                            |
| NumPy-powered     | Operations are fast and vectorized using NumPy                         |

---

## Declaration Syntax

```python
pd.DataFrame(
    data=None,              # Dict, 2D array, Series, scalar, or list of dicts
    index=None,             # Optional row labels
    columns=None,           # Optional column labels
    dtype=None,             # Optional data type for all columns
    copy=False              # Copy data if True
)
```
### Creating a DataFrame  

| Method | Description |  
|--------|-------------|  
| `pd.DataFrame(data)` | Creates a DataFrame from a dictionary, list, NumPy array, or Series |  
| `pd.DataFrame(data, index=index_list)` | Assigns custom row labels |  
| `pd.DataFrame(data, columns=col_list)` | Assigns custom column labels |  
| `pd.read_csv('file.csv')` | Reads a CSV file into a DataFrame |  
| `pd.read_excel('file.xlsx')` | Reads an Excel file into a DataFrame |  
| `pd.read_json('file.json')` | Reads a JSON file into a DataFrame |

### Basic Operations on DataFrame

| Operation             | Description                                |
|---------------------------|------------------------------------------------|
| Viewing Data           | Use `.head()`, `.tail()` to view the first or last few rows. |
| Selecting Columns      | Access columns by `df['column_name']` or `df.column_name`. |
| Selecting Rows         | Use `.loc[]` (label-based) or `.iloc[]` (integer-based). |
| Filtering              | Filter rows using conditions, e.g., `df[df['Age'] > 30]`. |
| Sorting                | Use `.sort_values()` to sort by one or more columns. |
| Handling Missing Data  | Use `.isnull()`, `.dropna()`, and `.fillna()` to handle missing data. |
| Merging/Joining        | Use `.merge()` to join DataFrames on columns or indices. |
| GroupBy Operations     | Group data using `.groupby()` for aggregation. |

---

## All Features by Category with Syntax

---

### Attributes (Read-only Metadata)

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
| `df.memory_usage()` | Returns memory usage per column | 

---

### Data Selection and Access

| Syntax                      | Description                          |
|-----------------------------|--------------------------------------|
| `df['col']` | Selects a single column as a Series |  
| `df[['col1', 'col2']]` | Selects multiple columns |  
| `df.loc[row_label, col_label]` | Accesses value by label |  
| `df.iloc[row_index, col_index]` | Accesses value by position |  
| `df.loc[row_label]` | Selects row by label |  
| `df.iloc[row_index]` | Selects row by index |  
| `df.loc[:, 'col']` | Selects a column with all rows |  
| `df.iloc[:, col_index]` | Selects a column by position |  
| `df.at[row, col]`           | Fast scalar access by label          |
| `df.iat[row, col]`          | Fast scalar access by position       |
| `df.get(col)`               | Safe column access                   |

---

### Filtering Rows  

| Condition | Description |  
|-----------|-------------|  
| `df[df['col'] > n]` | Selects rows where column value is greater than `n` |  
| `df[df['col'] == 'value']` | Selects rows where column equals 'value' |  
| `df[(df['col1'] > n) & (df['col2'] < m)]` | Selects rows where multiple conditions are met |  
| `df.query('col > n')` | Queries rows using expressions |  

---

### Viewing Data  

| Method | Description |  
|--------|-------------|  
| `df.head(n)` | Returns the first `n` rows (default 5) |  
| `df.tail(n)` | Returns the last `n` rows (default 5) |  
| `df.sample(n)` | Returns `n` random rows |  
| `df.info()` | Displays DataFrame summary |  
| `df.describe()` | Returns summary statistics for numeric columns | 

---

### Selecting Data  

| Method | Description |  
|--------|-------------|  
| `df['col']` | Selects a single column as a Series |  
| `df[['col1', 'col2']]` | Selects multiple columns |  
| `df.loc[row_label, col_label]` | Accesses value by label |  
| `df.iloc[row_index, col_index]` | Accesses value by position |  
| `df.loc[row_label]` | Selects row by label |  
| `df.iloc[row_index]` | Selects row by index |  
| `df.loc[:, 'col']` | Selects a column with all rows |  
| `df.iloc[:, col_index]` | Selects a column by position |

---

### Modifying Data  

| Method | Description |  
|--------|-------------|  
| `df['col'] = value` | Assigns a constant to a column |  
| `df['col'] = df['col'] * 2` | Updates column values |  
| `df['new_col'] = df['col1'] + df['col2']` | Creates a new column |  
| `df.insert(1, 'new_col', values)` | Inserts a column at index `1` |  
| `df.rename(columns={'old': 'new'})` | Renames columns |  
| `df.rename(index={0: 'first'})` | Renames rows |  

---

### Arithmetic & Operations

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

### 4. Aggregation & Statistics

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

### Missing Data Handling

| Syntax                                   | Description                            |
|------------------------------------------|----------------------------------------|
| `df.isna()`                              | Detect missing values                  |
| `df.notna()`                             | Detect non-missing values              |
| `df.fillna(value, method)`               | Fill missing values                    |
| `df.dropna(axis=0, how='any')`           | Drop rows/columns with NaN             |
| `df.replace(to_replace, value)`          | Replace values                         |

---

### 6. Column & Row Manipulation

| Syntax                                       | Description                            |
|----------------------------------------------|----------------------------------------|
| `df.rename(columns={'a':'A'})`               | Rename columns                         |
| `df.drop(columns=['col'])`                   | Drop column(s)                         |
| `df.drop(index=[0,1])`                       | Drop row(s)                            |
| `df.insert(loc, column, value)`              | Insert column at specific position     |
| `df.assign(newcol=values)`                   | Add new column                         |
| `df.pop(col)`                                | Remove column and return it            |

---

### 7. Sorting & Reindexing

| Syntax                                      | Description                            |
|---------------------------------------------|----------------------------------------|
| `df.sort_values(by='col', ascending=True)`  | Sort by column                         |
| `df.sort_index(axis=0)`                     | Sort by index                          |
| `df.reset_index(drop=True)`                 | Reset index                            |
| `df.set_index('col')`                       | Set column as index                    |
| `df.reindex(new_index)`                     | Reindex rows                           |

---

### 8. Filtering & Querying

| Syntax                                | Description                            |
|---------------------------------------|----------------------------------------|
| `df[df['col'] > 5]`                   | Filter rows with condition             |
| `df.query('col > 5')`                 | SQL-like querying                      |
| `df.where(condition)`                 | Keep if condition is True              |
| `df.mask(condition)`                  | Replace where condition is True        |

---

### 9. String Operations (`.str`)

| Syntax                         | Description                          |
|--------------------------------|--------------------------------------|
| `df['col'].str.lower()`        | Convert to lowercase                 |
| `df['col'].str.contains('x')`  | Check pattern match                  |
| `df['col'].str.replace('a','b')`| Replace substrings                  |
| `df['col'].str.len()`          | Length of each string                |

---

### 10. Date/Time Operations (`.dt`)

| Syntax                | Description                              |
|------------------------|------------------------------------------|
| `df['date'].dt.year`   | Extract year                            |
| `df['date'].dt.month`  | Extract month                           |
| `df['date'].dt.day`    | Extract day                             |
| `df['date'].dt.weekday`| Day of week                             |

> Note: Works only for datetime-type columns.

---

### 11. Type Conversion

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

### 12. Copying & Transposing

| Syntax                   | Description                     |
|--------------------------|---------------------------------|
| `df.copy(deep=True)`     | Deep copy of DataFrame          |
| `df.T`                   | Transpose rows and columns      |

---

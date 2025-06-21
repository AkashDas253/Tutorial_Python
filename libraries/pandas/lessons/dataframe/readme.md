
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

| Condition                              | Description                                      |  
|----------------------------------------|--------------------------------------------------|  
| `df[df['col'] > n]`                    | Selects rows where column value is greater than `n` |  
| `df[df['col'] == 'value']`             | Selects rows where column equals 'value'        |  
| `df[(df['col1'] > n) & (df['col2'] < m)]` | Selects rows where multiple conditions are met |  
| `df[df['col'].isin(['value1', 'value2'])]` | Selects rows where column value is in a list   |  
| `df[~df['col'].isin(['value1', 'value2'])]` | Selects rows where column value is not in a list |  
| `df[df['col'].str.contains('substring')]` | Selects rows where column contains a substring |  
| `df.where(condition)`                  | Keep rows where condition is True              |  
| `df.mask(condition)`                   | Replace rows where condition is True           |  

---

### Querying Rows  

| Syntax                                | Description                            |  
|---------------------------------------|----------------------------------------|  
| `df.query('col > n')`                 | Queries rows using expressions         |  
| `df.query('col == "value"')`          | Queries rows where column equals value |  
| `df.query('col1 > n & col2 < m')`     | Queries rows with multiple conditions  |  
| `df.query('col in ["value1", "value2"]')` | Queries rows where column value is in a list |  
| `df.query('col not in ["value1", "value2"]')` | Queries rows where column value is not in a list |  

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

### Aggregation & Statistics

| Syntax                        | Description                                                                                          |
|-------------------------------|------------------------------------------------------------------------------------------------------|
| `df.sum(axis=0)`              | Compute the sum of values along the specified axis. `axis=0` sums over columns, `axis=1` sums over rows. |
| `df.mean(axis=0)`             | Compute the mean (average) of values along the specified axis.                                       |
| `df.median(axis=0)`           | Compute the median (middle value) of values along the specified axis.                               |
| `df.min(axis=0)`              | Compute the minimum value along the specified axis.                                                 |
| `df.max(axis=0)`              | Compute the maximum value along the specified axis.                                                 |
| `df.std(axis=0)`              | Compute the standard deviation of values along the specified axis. Measures the spread of the data. |
| `df.var(axis=0)`              | Compute the variance of values along the specified axis. Measures the variability of the data.      |
| `df.mode()`                   | Compute the mode (most frequent value) for each column. Returns a DataFrame as there can be multiple modes. |
| `df.describe()`               | Generate summary statistics for numeric columns, including count, mean, std, min, 25%, 50%, 75%, and max. |
| `df.count()`                  | Count the number of non-NA/null values for each column or row.                                       |
| `df.cumsum()`                 | Compute the cumulative sum of values along the specified axis.                                      |
| `df.cumprod()`                | Compute the cumulative product of values along the specified axis.                                  |
| `df.corr()`                   | Compute pairwise correlation of numeric columns. Returns a DataFrame with correlation coefficients ranging from -1 to 1. |
| `df.cov()`                    | Compute pairwise covariance of numeric columns. Returns a DataFrame with covariance values.         |

---

### Grouping & Aggregation  

| Syntax                                | Description                                                                                          |
|---------------------------------------|------------------------------------------------------------------------------------------------------|
| `df.groupby('col').sum()`             | Groups the DataFrame by a column and computes the sum of values for each group.                      |
| `df.groupby('col').mean()`            | Groups the DataFrame by a column and computes the mean (average) of values for each group.           |
| `df.groupby(['col1', 'col2']).count()`| Groups the DataFrame by multiple columns and counts the number of non-NA values in each group.       |
| `df.agg({'col1': 'sum', 'col2': 'mean'})` | Applies multiple aggregation functions to specified columns. For example, sums `col1` and computes the mean of `col2`. |
| `df.groupby('col').min()`             | Groups the DataFrame by a column and computes the minimum value for each group.                      |
| `df.groupby('col').max()`             | Groups the DataFrame by a column and computes the maximum value for each group.                      |
| `df.groupby('col').std()`             | Groups the DataFrame by a column and computes the standard deviation for each group.                 |
| `df.groupby('col').var()`             | Groups the DataFrame by a column and computes the variance for each group.                           |
| `df.groupby('col').median()`          | Groups the DataFrame by a column and computes the median for each group.                             |
| `df.groupby('col').first()`           | Groups the DataFrame by a column and returns the first value in each group.                          |
| `df.groupby('col').last()`            | Groups the DataFrame by a column and returns the last value in each group.                           |
| `df.groupby('col').size()`            | Groups the DataFrame by a column and returns the size (number of rows) of each group.                |
| `df.groupby('col').agg(['sum', 'mean'])` | Groups the DataFrame by a column and applies multiple aggregation functions (e.g., sum and mean) to each group. |
| `df.groupby('col').apply(func)`       | Groups the DataFrame by a column and applies a custom function `func` to each group.                 |
| `df.pivot_table(values='col', index='col1', columns='col2', aggfunc='mean')` | Creates a pivot table, aggregating data using the specified function (e.g., mean).                   |
| `df.groupby('col').transform('mean')` | Groups the DataFrame by a column and applies a transformation (e.g., mean) to each group, returning a DataFrame of the same shape. |
| `df.groupby('col').filter(func)`      | Filters groups based on a custom function `func` that returns `True` or `False` for each group.      |

---

### Pivoting & Reshaping  

| Method                                | Description                                      |  
|---------------------------------------|--------------------------------------------------|  
| `df.pivot(index, columns, values)`    | Reshapes data by pivoting a DataFrame based on index, columns, and values. |  
| `df.melt(id_vars, var_name, value_name)` | Converts wide-format data into long-format data. |  
| `df.stack()`                          | Stacks the columns of a DataFrame into rows.     |  
| `df.unstack()`                        | Unstacks rows of a DataFrame into columns.       |  

---

### Merging & Joining  

| Method                                | Description                                      |  
|---------------------------------------|--------------------------------------------------|  
| `pd.concat([df1, df2])`               | Concatenates DataFrames along rows (default).    |  
| `pd.concat([df1, df2], axis=1)`       | Concatenates DataFrames along columns.          |  
| `df1.merge(df2, on='col')`            | Merges two DataFrames on a common column.        |  
| `df1.join(df2, on='col')`             | Joins two DataFrames using their index or a key. |  

---

### Exporting Data  

| Method                                | Description                                      |  
|---------------------------------------|--------------------------------------------------|  
| `df.to_csv('filename.csv')`           | Exports DataFrame to a CSV file.                |  
| `df.to_excel('filename.xlsx')`        | Exports DataFrame to an Excel file.             |  
| `df.to_json('filename.json')`         | Exports DataFrame to a JSON file.               |  
| `df.to_sql('table_name', connection)` | Writes DataFrame to a SQL table.                |  
| `df.to_html('filename.html')`         | Exports DataFrame to an HTML file.              |  
| `df.to_pickle('filename.pkl')`        | Serializes DataFrame to a pickle file.          |  

---

### Missing Data Handling

| Syntax                                   | Description                                                                                     |
|------------------------------------------|-------------------------------------------------------------------------------------------------|
| `df.isna()`                              | Detect missing values. Returns a DataFrame of the same shape as `df`, with `True` for NaN values and `False` otherwise. |
| `df.notna()`                             | Detect non-missing values. Returns a DataFrame of the same shape as `df`, with `True` for non-NaN values and `False` otherwise. |
| `df.fillna(value, method)`               | Fill missing values. Replace NaN values with a specified value or use a method like `ffill` (forward fill) or `bfill` (backward fill). |
| `df.dropna(axis=0, how='any')`           | Drop rows or columns with NaN values. `axis=0` drops rows, `axis=1` drops columns. `how='any'` drops if any NaN is present, `how='all'` drops only if all values are NaN. |
| `df.replace(to_replace, value)`          | Replace specific values in the DataFrame. `to_replace` specifies the value(s) to replace, and `value` specifies the replacement. |
| `df.interpolate(method='linear')`        | Fill NaN values using interpolation. The `method` parameter can be `linear`, `polynomial`, `spline`, etc. |
| `df.isnull().sum()`                      | Count the number of NaN values in each column. Useful for quickly identifying missing data.     |
| `df.fillna(df.mean())`                   | Replace NaN values with the mean of each column. Can also use `median`, `mode`, etc., depending on the requirement. |
| `df.dropna(subset=['column_name'])`      | Drop rows where NaN values are present in specific columns. Replace `'column_name'` with the desired column name. |
| `df.mask(df < 0, np.nan)`                | Replace values that meet a condition (e.g., negative values) with NaN.                         |
| `df.ffill()`                             | Forward fill NaN values using the last valid observation.                                       |
| `df.bfill()`                             | Backward fill NaN values using the next valid observation.                                      |
| `df.fillna(method='pad')`                | Alias for forward fill (`ffill`).                                                              |
| `df.fillna(method='backfill')`           | Alias for backward fill (`bfill`).                                                             |
| `df.dropna(thresh=n)`                    | Drop rows or columns where fewer than `n` non-NaN values are present.                          |
| `df.fillna({'col1': 0, 'col2': 'missing'})` | Fill NaN values in specific columns with different values.                                      |
| `df.fillna(inplace=True)`                | Modify the DataFrame in place without creating a new object.                                   |
| `df.isnull().any(axis=1)`                | Detect rows with at least one NaN value. Returns a boolean Series.                             |
| `df.isnull().all(axis=1)`                | Detect rows where all values are NaN. Returns a boolean Series.                                |

---

### Column & Row Manipulation

| Syntax                                       | Description                            |
|----------------------------------------------|----------------------------------------|
| `df.rename(columns={'a':'A'})`               | Rename columns                         |
| `df.drop(columns=['col'])`                   | Drop column(s)                         |
| `df.drop(index=[0,1])`                       | Drop row(s)                            |
| `df.insert(loc, column, value)`              | Insert column at specific position     |
| `df.assign(newcol=values)`                   | Add new column                         |
| `df.pop(col)`                                | Remove column and return it            |

---

### Sorting & Reindexing

| Syntax                                      | Description                            |
|---------------------------------------------|----------------------------------------|
| `df.sort_values(by='col', ascending=True)`  | Sort by column                         |
| `df.sort_index(axis=0)`                     | Sort by index                          |
| `df.reset_index(drop=True)`                 | Reset index                            |
| `df.set_index('col')`                       | Set column as index                    |
| `df.reindex(new_index)`                     | Reindex rows                           |

---

### Data-Specific Operations

Pandas provides accessors like `.str` for strings, `.dt` for datetime, and `.cat` for categorical data. These accessors allow you to perform operations similar to Python's built-in methods but in a vectorized way, handling exceptions like `NaN` gracefully.

| Data Type       | Accessor | Example Syntax                          | Description                                                                 |
|------------------|----------|-----------------------------------------|-----------------------------------------------------------------------------|
| String           | `.str`   | `df['col'].str.upper()`                | Operates like Python string methods (e.g., `.upper()`, `.replace()`, etc.). Handles `NaN` without errors. |
| Datetime         | `.dt`    | `df['col'].dt.year`                    | Extracts parts of datetime (e.g., year, month, day) or formats them.        |
| Categorical      | `.cat`   | `df['col'].cat.codes`                  | Accesses categorical properties or converts categories to numeric codes.    |
| Numeric          | N/A      | `df['col'] + 10`                       | Operates directly like Python numeric types (e.g., addition, subtraction).  |
| General          | `.apply` | `df['col'].apply(lambda x: x + 1)`     | Applies custom Python functions to each element.                           |

### Key Notes:
- **Vectorized Operations**: These accessors allow operations on entire columns, making them faster than Python loops.
- **Exception Handling**: Operations automatically skip `NaN` values without raising errors.
- **Chaining**: You can chain multiple operations, e.g., `df['col'].str.strip().str.upper()`.

---

### Type Conversion

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

### Copying & Transposing

| Syntax                   | Description                     |
|--------------------------|---------------------------------|
| `df.copy(deep=True)`     | Deep copy of DataFrame          |
| `df.T`                   | Transpose rows and columns      |

---

### Handling Duplicates  

| Method                  | Description                          |  
|-------------------------|--------------------------------------|  
| `df.duplicated()`       | Checks for duplicate rows.          |  
| `df.drop_duplicates()`  | Removes duplicate rows.             |  

---

### Value Manipulation  

| Method                  | Description                          |  
|-------------------------|--------------------------------------|  
| `df.clip(lower, upper)` | Limits values between `lower` and `upper`. |  
| `df.nunique()`          | Returns the number of unique values per column. |  

---

### Cumulative Operations  

| Method                  | Description                          |  
|-------------------------|--------------------------------------|  
| `df.cumsum()`           | Computes the cumulative sum per column. |  
| `df.cumprod()`          | Computes the cumulative product per column. |  

---

### Difference Calculation  

| Method                  | Description                          |  
|-------------------------|--------------------------------------|  
| `df.diff()`             | Computes the difference between consecutive values. |  

---
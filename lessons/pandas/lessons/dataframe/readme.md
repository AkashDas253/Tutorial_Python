## Pandas DataFrame  

A **DataFrame** is a two-dimensional labeled data structure similar to a table. It consists of rows and columns, where each column is a **Series**. It supports various operations like indexing, filtering, aggregation, merging, and transformation.

---

### Creating a DataFrame  

| Method | Description |  
|--------|-------------|  
| `pd.DataFrame(data)` | Creates a DataFrame from a dictionary, list, NumPy array, or Series |  
| `pd.DataFrame(data, index=index_list)` | Assigns custom row labels |  
| `pd.DataFrame(data, columns=col_list)` | Assigns custom column labels |  
| `pd.read_csv('file.csv')` | Reads a CSV file into a DataFrame |  
| `pd.read_excel('file.xlsx')` | Reads an Excel file into a DataFrame |  
| `pd.read_json('file.json')` | Reads a JSON file into a DataFrame |  

---

### Attributes of DataFrame  

| Attribute | Description |  
|-----------|-------------|  
| `df.index` | Returns row labels |  
| `df.columns` | Returns column labels |  
| `df.dtypes` | Returns data types of columns |  
| `df.shape` | Returns `(rows, columns)` count |  
| `df.size` | Returns total number of elements |  
| `df.ndim` | Returns number of dimensions (2D) |  
| `df.values` | Returns data as a NumPy array |  
| `df.memory_usage()` | Returns memory usage per column |  

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

### Filtering Rows  

| Condition | Description |  
|-----------|-------------|  
| `df[df['col'] > n]` | Selects rows where column value is greater than `n` |  
| `df[df['col'] == 'value']` | Selects rows where column equals 'value' |  
| `df[(df['col1'] > n) & (df['col2'] < m)]` | Selects rows where multiple conditions are met |  
| `df.query('col > n')` | Queries rows using expressions |  

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

### Handling Missing Data  

| Method | Description |  
|--------|-------------|  
| `df.isnull()` | Returns `True` for NaN values |  
| `df.notnull()` | Returns `True` for non-null values |  
| `df.dropna()` | Drops rows with NaN values |  
| `df.dropna(axis=1)` | Drops columns with NaN values |  
| `df.fillna(value)` | Replaces NaN with a specific value |  
| `df.fillna(method='ffill')` | Forward fills missing values |  
| `df.fillna(method='bfill')` | Backward fills missing values |  

---

### Sorting Data  

| Method | Description |  
|--------|-------------|  
| `df.sort_values('col')` | Sorts by a column in ascending order |  
| `df.sort_values('col', ascending=False)` | Sorts in descending order |  
| `df.sort_values(['col1', 'col2'])` | Sorts by multiple columns |  
| `df.sort_index()` | Sorts rows by index |  

---

### Aggregation & Statistics  

| Method | Description |  
|--------|-------------|  
| `df.sum()` | Column-wise sum |  
| `df.mean()` | Column-wise mean |  
| `df.median()` | Column-wise median |  
| `df.std()` | Column-wise standard deviation |  
| `df.min(), df.max()` | Column-wise min and max values |  
| `df.count()` | Number of non-null values per column |  
| `df.corr()` | Correlation between numeric columns |  
| `df.cov()` | Covariance between numeric columns |  

---

### Grouping & Aggregation  

| Method | Description |  
|--------|-------------|  
| `df.groupby('col').sum()` | Groups by a column and sums values |  
| `df.groupby('col').mean()` | Groups by a column and computes mean |  
| `df.groupby(['col1', 'col2']).count()` | Groups by multiple columns |  
| `df.agg({'col1': 'sum', 'col2': 'mean'})` | Applies multiple aggregations |  

---

### Pivoting & Reshaping  

| Method | Description |  
|--------|-------------|  
| `df.pivot(index, columns, values)` | Pivots DataFrame |  
| `df.melt(id_vars, var_name, value_name)` | Converts wide format to long format |  
| `df.stack()` | Converts columns into rows |  
| `df.unstack()` | Converts rows into columns |  

---

### Merging & Joining  

| Method | Description |  
|--------|-------------|  
| `pd.concat([df1, df2])` | Concatenates along rows |  
| `pd.concat([df1, df2], axis=1)` | Concatenates along columns |  
| `df1.merge(df2, on='col')` | Joins DataFrames on a common column |  
| `df1.join(df2, on='col')` | Joins DataFrames by index |  

---

### Exporting Data  

| Method | Description |  
|--------|-------------|  
| `df.to_csv('file.csv')` | Saves as a CSV file |  
| `df.to_excel('file.xlsx')` | Saves as an Excel file |  
| `df.to_json('file.json')` | Saves as a JSON file |  

---

### Additional Methods  

| Method | Description |  
|--------|-------------|  
| `df.duplicated()` | Checks for duplicate rows |  
| `df.drop_duplicates()` | Removes duplicate rows |  
| `df.clip(lower, upper)` | Limits values between `lower` and `upper` |  
| `df.nunique()` | Returns number of unique values per column |  
| `df.cumsum()` | Cumulative sum per column |  
| `df.cumprod()` | Cumulative product per column |  
| `df.diff()` | Computes difference between consecutive values |  

---

### Working with Strings (if DataFrame contains text data)  

| Method | Description |  
|--------|-------------|  
| `df['col'].str.lower()` | Converts all text to lowercase |  
| `df['col'].str.upper()` | Converts all text to uppercase |  
| `df['col'].str.strip()` | Removes leading and trailing spaces |  
| `df['col'].str.contains('text')` | Checks if elements contain a substring |  
| `df['col'].str.replace('old', 'new')` | Replaces a substring |  
| `df['col'].str.split('delimiter')` | Splits elements based on a delimiter |  

---

A Pandas **DataFrame** is a flexible and efficient structure for handling structured data, supporting powerful operations for data manipulation, transformation, and analysis.
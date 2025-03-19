## Pandas Series  

A **Series** is a one-dimensional labeled array capable of holding any data type. It supports indexing, slicing, vectorized operations, and statistical methods, making it useful for numerical and categorical data processing.  

---

### Creating a Series  

| Method | Description |  
|--------|-------------|  
| `pd.Series(data)` | Creates a Series from a list, tuple, dictionary, scalar, or NumPy array |  
| `pd.Series(data, index=index_list)` | Creates a Series with a custom index |  
| `pd.Series(data, dtype=type)` | Specifies the data type of elements |  

---

### Attributes of Series  

| Attribute | Description |  
|-----------|-------------|  
| `s.index` | Returns the index labels |  
| `s.values` | Returns the data as a NumPy array |  
| `s.dtype` | Returns the data type of elements |  
| `s.shape` | Returns the shape of the Series |  
| `s.size` | Returns the total number of elements |  
| `s.nbytes` | Returns the memory usage in bytes |  
| `s.name` | Returns or sets the name of the Series |  

---

### Accessing Elements  

| Method | Description |  
|--------|-------------|  
| `s[index]` | Accesses element by position |  
| `s['label']` | Accesses element by index label |  
| `s.iloc[pos]` | Accesses by integer position |  
| `s.loc[label]` | Accesses by label |  
| `s[1:3]` | Slices elements from index 1 to 2 |  
| `s[['a', 'c']]` | Selects multiple elements by label |  

---

### Viewing Data  

| Method | Description |  
|--------|-------------|  
| `s.head(n)` | Returns the first `n` elements (default 5) |  
| `s.tail(n)` | Returns the last `n` elements (default 5) |  

---

### Vectorized Operations  

| Operation | Description |  
|-----------|-------------|  
| `s + n` | Adds `n` to each element |  
| `s - n` | Subtracts `n` from each element |  
| `s * n` | Multiplies each element by `n` |  
| `s / n` | Divides each element by `n` |  
| `s ** n` | Raises each element to the power of `n` |  
| `s % n` | Returns the remainder after division |  

---

### Applying Functions  

| Method | Description |  
|--------|-------------|  
| `s.apply(func)` | Applies a function to each element |  
| `s.map(func)` | Applies a function element-wise |  
| `s.apply(lambda x: x * 2)` | Example of applying a lambda function |  

---

### Handling Missing Data  

| Method | Description |  
|--------|-------------|  
| `s.isnull()` | Returns `True` for NaN values |  
| `s.notnull()` | Returns `True` for non-null values |  
| `s.dropna()` | Removes NaN values |  
| `s.fillna(value)` | Replaces NaN with a specific value |  
| `s.replace(old, new)` | Replaces specific values |  

---

### Statistical Operations  

| Method | Description |  
|--------|-------------|  
| `s.sum()` | Sum of all elements |  
| `s.mean()` | Mean (average) |  
| `s.median()` | Median value |  
| `s.std()` | Standard deviation |  
| `s.min(), s.max()` | Minimum and maximum values |  
| `s.describe()` | Summary statistics |  

---

### Sorting  

| Method | Description |  
|--------|-------------|  
| `s.sort_values()` | Sorts values in ascending order |  
| `s.sort_values(ascending=False)` | Sorts values in descending order |  
| `s.sort_index()` | Sorts elements by index |  

---

### Checking Conditions  

| Method | Description |  
|--------|-------------|  
| `s > n` | Returns `True` for elements greater than `n` |  
| `s < n` | Returns `True` for elements less than `n` |  
| `s == n` | Returns `True` for elements equal to `n` |  
| `s[s > n]` | Filters elements greater than `n` |  

---

### Combining Multiple Series  

| Method | Description |  
|--------|-------------|  
| `s1.add(s2)` | Adds two Series element-wise |  
| `s1.sub(s2)` | Subtracts two Series element-wise |  
| `s1.mul(s2)` | Multiplies two Series element-wise |  
| `s1.div(s2)` | Divides two Series element-wise |  

---

### Converting Series  

| Method | Description |  
|--------|-------------|  
| `s.to_list()` | Converts to a Python list |  
| `s.to_dict()` | Converts to a dictionary |  
| `s.astype(dtype)` | Changes data type |  
| `s.to_frame()` | Converts to a DataFrame |  

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

### Working with Strings (if Series contains text data)  

| Method | Description |  
|--------|-------------|  
| `s.str.lower()` | Converts all text to lowercase |  
| `s.str.upper()` | Converts all text to uppercase |  
| `s.str.strip()` | Removes leading and trailing spaces |  
| `s.str.contains('text')` | Checks if elements contain a substring |  
| `s.str.replace('old', 'new')` | Replaces a substring |  
| `s.str.split('delimiter')` | Splits elements based on a delimiter |  

---

A Pandas **Series** is a powerful one-dimensional data structure that supports efficient data manipulation, indexing, mathematical operations, handling missing data, and statistical computations. It is widely used in data analysis and machine learning.
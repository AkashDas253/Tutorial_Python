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

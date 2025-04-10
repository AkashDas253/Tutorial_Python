
## **GroupBy Operations in Pandas**  
Group data to perform aggregation, transformation, or filtering on subsets.

---

### **Basic GroupBy Syntax**

```python
df.groupby('column')  
# Create GroupBy object by single column

df.groupby(['col1', 'col2'])  
# Group by multiple columns
```

```python
grouped = df.groupby('group_col')  
# Store GroupBy object for reuse
```

---

### **Aggregation**

```python
df.groupby('group')['value'].sum()  
# Sum of values per group

df.groupby('group')['value'].agg('mean')  
# Aggregate using single function

df.groupby('group').agg(['sum', 'mean'])  
# Aggregate multiple functions
```

```python
df.groupby('group').agg({'val1': 'mean', 'val2': 'sum'})  
# Different aggregations per column
```

---

### **Built-in Aggregation Methods**

| Function       | Description                     | Syntax Example                     |
|----------------|----------------------------------|------------------------------------|
| `sum()`        | Sum of values                    | `df.groupby('grp')['val'].sum()`   |
| `mean()`       | Mean of values                   | `df.groupby('grp')['val'].mean()`  |
| `count()`      | Number of non-NaN values         | `df.groupby('grp')['val'].count()` |
| `size()`       | Total group size (including NaNs)| `df.groupby('grp').size()`         |
| `min()` / `max()` | Minimum / maximum             | `df.groupby('grp')['val'].min()`   |
| `std()` / `var()`| Standard deviation / Variance  | `df.groupby('grp')['val'].std()`   |
| `first()` / `last()` | First / Last record        | `df.groupby('grp')['val'].first()` |

---

### **Filtering Groups**

```python
df.groupby('group').filter(lambda x: len(x) > 3)  
# Keep groups with more than 3 rows

df.groupby('group')['val'].filter(lambda x: x.mean() > 10)  
# Keep groups where column mean > 10
```

---

### **Transforming Groups**

```python
df.groupby('group')['val'].transform('mean')  
# Return group mean, broadcasted to original shape

df['zscore'] = df.groupby('group')['val'].transform(lambda x: (x - x.mean()) / x.std())  
# Group-wise z-score normalization
```

---

### **Applying Custom Functions**

```python
df.groupby('group').apply(lambda x: x.sort_values('val', ascending=False))  
# Apply any custom function on each group
```

```python
def group_summary(x):  
    return pd.Series({  
        'min': x['val'].min(),  
        'max': x['val'].max(),  
        'range': x['val'].max() - x['val'].min()  
    })

df.groupby('group').apply(group_summary)  
# Custom aggregation summary
```

---

### **GroupBy and Missing Values**

```python
df.groupby('group', dropna=False)  
# Include NaN groups during grouping
```

---

### **Named Aggregation (since pandas 0.25)**

```python
df.groupby('group').agg(  
    min_val=('val', 'min'),  
    max_val=('val', 'max')  
)  
# Named columns for aggregated results
```

---

### **GroupBy with Multiple Index Levels**

```python
df.groupby(['A', 'B'])['C'].sum()  
# Group by hierarchical levels (multi-index grouping)
```

---


## **Reshaping & Pivoting in Pandas**  
Restructure DataFrames by pivoting, stacking, melting, and reshaping for better analysis and reporting.

---

### **Pivoting DataFrames**

```python
df.pivot(index='row_key', columns='col_key', values='value')  
# Reshape from long to wide format
```

> All row/column combinations must be unique (no duplicates).

---

### **Pivot Table (Aggregated Pivot)**

```python
df.pivot_table(index='group', columns='category', values='value', aggfunc='mean')  
# Aggregated pivot table with custom aggregation
```

```python
df.pivot_table(index=['A'], values='B', aggfunc='sum', fill_value=0)  
# With fill for missing values
```

---

### **Melt (Unpivot)**

```python
df.melt(id_vars=['id'], value_vars=['A', 'B'])  
# Unpivot wide format into long format
```

```python
df.melt(var_name='feature', value_name='val')  
# Rename variable and value column names
```

---

### **Stack (Pivot Columns to Index)**

```python
df.stack()  
# Pivot columns into inner row index level (Series with MultiIndex)
```

```python
df.stack(level=1)  
# Stack a specific level of column MultiIndex
```

---

### **Unstack (Pivot Index to Columns)**

```python
df.unstack()  
# Pivot row index into column (default last level)
```

```python
df.unstack(level=0)  
# Unstack a specific index level into columns
```

---

### **Wide <-> Long: Full Example**

```python
# From long to wide
df.pivot(index='date', columns='item', values='value')

# From wide to long
df.melt(id_vars='date', var_name='item', value_name='value')
```

---

### **Transpose DataFrame**

```python
df.T  
# Transpose rows to columns and columns to rows
```

---

### **Reshape with `.values.reshape()` (for NumPy)**

```python
df['col'].values.reshape(-1, 1)  
# Reshape Series for ML models
```

---

### **Set / Reset Index**

```python
df.set_index('col')  
# Move column to index

df.reset_index()  
# Move index back to column
```

---

### **MultiIndex Reshaping**

```python
df.unstack().stack()  
# Convert between wide/long on MultiIndex

df.swaplevel()  
# Swap levels in a MultiIndex
```

---


## **Merging & Joining in Pandas**  
Combine data from multiple DataFrames using merge, join, or concat operations.

---

### **merge() — Relational Join (SQL-style)**

```python
pd.merge(left, right, how='inner', on='key')  
# Default is inner join on common column names
```

```python
pd.merge(left, right, how='left', on='key')  
# Left join: keep all rows from left, matching rows from right
```

```python
pd.merge(left, right, how='right', on='key')  
# Right join: keep all rows from right, matching from left
```

```python
pd.merge(left, right, how='outer', on='key')  
# Outer join: keep all rows from both, NaN where no match
```

```python
pd.merge(left, right, left_on='col1', right_on='col2')  
# Join using different column names in both DataFrames
```

```python
pd.merge(left, right, on='key', suffixes=('_x', '_y'))  
# Add suffixes to overlapping column names
```

---

### **join() — Index-based Join**

```python
df1.join(df2, how='left')  
# Join on index (or key column if passed)
```

```python
df1.join(df2.set_index('key'), on='key')  
# Join on column but use index of df2
```

```python
df1.join([df2, df3])  
# Join multiple DataFrames at once on index
```

---

### **concat() — Append or Stack Along Axis**

```python
pd.concat([df1, df2])  
# Stack vertically (axis=0), like UNION ALL

pd.concat([df1, df2], axis=1)  
# Combine horizontally (axis=1), like SQL JOIN on index
```

```python
pd.concat([df1, df2], ignore_index=True)  
# Reset index and reindex after concatenation
```

```python
pd.concat([df1, df2], join='inner')  
# Only keep common columns
```

---

### **append() — Deprecated (Use concat)**

```python
# Deprecated in newer versions of Pandas
df1.append(df2, ignore_index=True)  
# Use pd.concat instead
```

---

### **combine_first() — Combine with Priority**

```python
df1.combine_first(df2)  
# Fill missing values in df1 with values from df2
```

---

### **merge_ordered() — Time-aware Merge (for ordered data)**

```python
pd.merge_ordered(df1, df2, on='date', fill_method='ffill')  
# Merge with order preservation and optional filling
```

---

### **merge_asof() — Closest-Key Merge (Time Series)**

```python
pd.merge_asof(df1, df2, on='date')  
# Merge by nearest prior date (like SQL asof join)
```

```python
pd.merge_asof(df1, df2, on='date', by='id')  
# Merge on nearest date **per group**
```

---

### **Handling Overlapping Column Names**

```python
pd.merge(df1, df2, on='key', suffixes=('_left', '_right'))  
# Resolve duplicate column names during merge
```

---

# **Selection & Filtering in Pandas**

---

### **Basic Column Selection**

```python
df['column']                       # Select a single column (returns Series)
df[['col1', 'col2']]               # Select multiple columns (returns DataFrame)
```

---

### **Row Selection by Index**

```python
df.loc[index_label]               # Select row(s) by label (index) – inclusive
df.iloc[index_position]           # Select row(s) by integer position – exclusive
df.at[label, 'col']               # Fast scalar access by label
df.iat[pos, col_index]            # Fast scalar access by integer position
```

---

### **Row & Column Selection Combined**

```python
df.loc[row_label, 'col']                  # Specific row and column by label
df.loc[[1, 3], ['A', 'B']]                # Multiple rows and columns by label
df.iloc[1:3, 0:2]                         # Slicing rows/columns by position
```

---

### **Conditional Filtering (Boolean Indexing)**

```python
df[df['col'] > 10]                        # Rows where col > 10
df[(df['A'] > 1) & (df['B'] < 5)]         # AND condition
df[(df['A'] > 1) | (df['B'] < 5)]         # OR condition
df[~(df['A'] > 1)]                        # NOT condition
```

---

### **Filtering with `isin()`, `between()`, `str.contains()`**

```python
df[df['col'].isin([1, 2, 3])]             # Values in list
df[df['col'].between(10, 20)]             # Values between 10 and 20 (inclusive)
df[df['col'].str.contains('abc')]         # Rows with substring in column
```

---

### **Query Method (SQL-like Syntax)**

```python
df.query('col1 > 2 and col2 == "A"')      # Simple SQL-style filtering
val = 5
df.query('col > @val')                    # Using external variable
```

---

### **Set Values Using Condition**

```python
df.loc[df['col'] > 5, 'col'] = 0          # Replace values where col > 5
```

---

### **Drop Rows or Columns**

```python
df.drop('column', axis=1)                # Drop a column
df.drop([0, 1], axis=0)                  # Drop rows by index
```

---

### **Select Columns Using `filter()`**

```python
df.filter(items=['A', 'B'])              # Select specific columns
df.filter(like='item', axis=1)           # Columns with 'item' in name
df.filter(regex='^col[0-9]$', axis=1)    # Columns matching regex pattern
```

---

## **Advanced Filtering**

```python
df[df['A'].notna()]                      # Rows where column 'A' is not NaN
df[df['A'].isna()]                       # Rows where column 'A' is NaN
df[df.duplicated()]                      # Duplicate rows
df[~df.duplicated()]                     # Unique rows
df[df['col'].str.startswith('X')]       # String starts with condition
df[df['col'].str.endswith('Z')]         # String ends with condition
df[df['col'].str.match(r'^X\d+$')]      # Regex match on string column
df[df.eval('A > B')]                    # Expression-based filtering (eval)
```

---

## **Multi-Index Filtering**

```python
df.loc[('group1', 1)]                    # Select specific row using multi-index
df.loc[('group1',)]                      # All rows with outer index 'group1'
df.loc[pd.IndexSlice[:, 1], :]           # Use IndexSlice for slicing levels
```

> Works only if the DataFrame has a MultiIndex.

```python
df.xs('group1', level=0)                 # Cross-section: rows where level 0 == 'group1'
df.xs(1, level=1, drop_level=False)      # Keep the index after filtering
```

---


## Merging / Joining / Concatenation in Pandas

These operations are used to combine multiple DataFrames into a single one by aligning data based on **index** or **column keys**.

---

### `merge()` – SQL-style Join on Columns

```python
pd.merge(
    left,                         # Left DataFrame
    right,                        # Right DataFrame
    how="inner",                  # Type of join: 'left', 'right', 'outer', 'inner', 'cross'
    on=None,                      # Column(s) to join on
    left_on=None,                 # Column(s) from left DF
    right_on=None,                # Column(s) from right DF
    left_index=False,             # Use index from left DF
    right_index=False,            # Use index from right DF
    sort=False,                   # Sort result
    suffixes=("_x", "_y"),        # Suffixes for overlapping columns
    copy=True,                    # Copy data (True by default)
    indicator=False,              # Add merge indicator column
    validate=None                 # Validate merge ('1:1', '1:m', 'm:1', 'm:m')
)
```

#### **Examples:**

```python
pd.merge(df1, df2, on="id")  # Inner join on 'id'
pd.merge(df1, df2, how="left", left_on="key1", right_on="key2")
pd.merge(df1, df2, how="outer", on=["key1", "key2"])
```
 
| Method | Description |  
|--------|-------------|  
| `pd.merge(df1, df2, on='key')` | Merges two DataFrames on a common column or index (`key`) |  
| `pd.merge(df1, df2, how='inner')` | Performs an inner join (default) between two DataFrames |  
| `pd.merge(df1, df2, how='outer')` | Performs an outer join, including all rows from both DataFrames |  
| `pd.merge(df1, df2, how='left')` | Performs a left join, keeping all rows from the left DataFrame |  
| `pd.merge(df1, df2, how='right')` | Performs a right join, keeping all rows from the right DataFrame |  
| `pd.merge(df1, df2, left_on='key1', right_on='key2')` | Merges DataFrames with different column names for merging |  
| `pd.merge(df1, df2, left_index=True, right_index=True)` | Merges DataFrames using their index instead of columns |  
| `pd.merge(df1, df2, indicator=True)` | Adds a `_merge` column to track which DataFrame the row originated from |  


---

### Join Types (for `how` parameter)

| Join Type | Description                           |
|-----------|---------------------------------------|
| `inner`   | Keep only rows with matching keys     |
| `outer`   | Keep all rows, fill with NaNs         |
| `left`    | Keep all rows from left DataFrame     |
| `right`   | Keep all rows from right DataFrame    |
| `cross`   | Cartesian product of both tables      |

---

### `.join()` – Join on Index or Column

```python
df.join(
    other,                        # Other DataFrame
    on=None,                      # Join column (if not on index)
    how="left",                   # 'left', 'right', 'outer', 'inner'
    lsuffix="",                   # Suffix for overlapping left columns
    rsuffix="",                   # Suffix for overlapping right columns
    sort=False                    # Sort result
)
```

**Examples:**

```python
df1.join(df2, on="key", how="inner")          # Join on column 'key'
df1.set_index("id").join(df2.set_index("id")) # Join on index
```

> Use `.join()` when working with indexes or combining columns horizontally.

| Method | Description |  
|--------|-------------|  
| `df1.join(df2, on='key')` | Joins two DataFrames on a specified column (`key`) |  
| `df1.join(df2, how='inner')` | Performs an inner join on two DataFrames |  
| `df1.join(df2, how='outer')` | Performs an outer join on two DataFrames |  
| `df1.join(df2, how='left')` | Performs a left join on two DataFrames |  
| `df1.join(df2, how='right')` | Performs a right join on two DataFrames |  
| `df1.join(df2, lsuffix='_left', rsuffix='_right')` | Adds suffixes to overlapping column names when joining |  


---

### `concat()` – Concatenation Along a Particular Axis

```python
pd.concat(
    objs,                         # List of DataFrames or Series
    axis=0,                       # 0 = rows (stacking), 1 = columns (side-by-side)
    join="outer",                 # 'outer' (union), 'inner' (intersection)
    ignore_index=False,           # Ignore original index and reassign new ones
    keys=None,                    # Create hierarchical index with keys
    levels=None,                  # Specific levels for keys
    names=None,                   # Name of levels in resulting MultiIndex
    verify_integrity=False,       # Check for duplicates in index
    sort=False                    # Sort if needed
)
```

**Examples:**

```python
pd.concat([df1, df2])                      # Stack rows
pd.concat([df1, df2], axis=1)             # Stack columns
pd.concat([df1, df2], ignore_index=True)  # Reindex after stacking
```

| Method | Description |  
|--------|-------------|  
| `pd.concat([df1, df2], axis=0)` | Concatenates DataFrames vertically (row-wise) |  
| `pd.concat([df1, df2], axis=1)` | Concatenates DataFrames horizontally (column-wise) |  
| `pd.concat([df1, df2], ignore_index=True)` | Concatenates DataFrames and reindexes rows |  
| `pd.concat([df1, df2], keys=['df1', 'df2'])` | Adds hierarchical index to the concatenated DataFrame |  
| `pd.concat([df1, df2], join='inner')` | Concatenates DataFrames with intersection of columns (inner join) |  
| `pd.concat([df1, df2], join='outer')` | Concatenates DataFrames with union of columns (outer join) | 

---

### `append()` (Deprecated)

```python
df1.append(
    df2,                           # DataFrame or Series to append
    ignore_index=False,            # Ignore original index
    verify_integrity=False,        # Check for index duplication
    sort=False                     # Sort columns
)
```

**Example:**

```python
df1.append(df2, ignore_index=True)
```

> Deprecated in recent versions; use `pd.concat()` instead.

---

### Use Cases & Differences

| Operation     | Axis     | Key Match         | Use Case                              |
|---------------|----------|-------------------|----------------------------------------|
| `merge()`     | N/A      | Column or Index   | SQL-style joins                        |
| `join()`      | Columns  | Index or Column   | Joining on index (or keys)             |
| `concat()`    | 0 or 1   | N/A               | Stack or combine multiple DataFrames   |
| `append()`    | 0        | N/A               | Add rows (deprecated, use `concat()`)  |

---

### **Handling Duplicates During Merge, Join, and Concatenation**  
| Method | Description |  
|--------|-------------|  
| `df.drop_duplicates()` | Removes duplicate rows from a DataFrame |  
| `df1.merge(df2, on='key', how='inner').drop_duplicates()` | Removes duplicates after merging two DataFrames |  

---

### **Performance Considerations**  
- **Merge and Join** are optimized for specific operations, especially when keys are involved. It’s efficient to use a column with unique identifiers as the merge key.
- **Concatenation** is more efficient when combining DataFrames of the same structure (columns), particularly when the number of rows is high.  
- **Handling Large DataFrames**: It’s advisable to work with **`chunksize`** or **`dask`** for better memory management during merges or concatenations.  

---

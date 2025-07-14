
## Indexing & Alignment in Pandas

---

### Index Objects

- Immutable arrays used to label axis (rows/columns).
- Types: `Index`, `RangeIndex`, `DatetimeIndex`, `MultiIndex`, etc.

```python
df.index        # Access row index
df.columns      # Access column index
```

---

### Setting Index

```python
df.set_index("col_name")               # Set column as index (returns new DataFrame)
df.set_index(["col1", "col2"])         # Set MultiIndex
df.set_index("col", inplace=True)      # Modify in place
```

---

### Resetting Index

```python
df.reset_index()                       # Move index back to column
df.reset_index(drop=True)              # Discard index entirely
```

---

### Indexing Types

| Indexing Type     | Description                      | Example                             |
|-------------------|----------------------------------|-------------------------------------|
| Label-based       | Access by labels                 | `df.loc["label"]`                   |
| Position-based    | Access by integer position       | `df.iloc[0]`                        |
| Mixed             | For Series only                  | `s["label"]` or `s[0]`              |
| Callable          | Indexing with a function         | `df.loc[lambda x: x["col"] > 0]`    |

---

### Hierarchical Indexing (MultiIndex)

```python
df.set_index(["col1", "col2"])
df.index.names         # Show index level names
df.index.levels        # Show unique values per level
```

✅ Used for multidimensional indexing in rows or columns.

---

### Accessing MultiIndex Elements

```python
df.loc[("A", "X")]                   # Access multi-level row
df.loc[("A",)]                       # Partial key for higher level
df.xs("X", level="col2")            # Cross-section
```

---

### Slicing with loc / iloc

```python
df.loc["row_label"]                 # Single row by label
df.loc["row1":"row3"]              # Label slicing (inclusive)
df.iloc[1:4]                        # Integer slicing (exclusive)
```

---

### Reindexing

```python
df.reindex([0, 2, 4])                # Change row index
df.reindex(columns=["A", "B", "C"])  # Change column index
```

- Fills missing with `NaN` by default
- Use `fill_value` or `method` to control

---

### Index Alignment (Automatic)

```python
df1 + df2        # Aligns on index and columns
s1 + s2          # Aligns on index
```

✅ If indices don’t match, result will have NaN in unmatched parts.

---

### Alignment with `add`, `sub`, etc.

```python
df1.add(df2, fill_value=0)
df1.sub(df2, fill_value=0)
```

✅ Element-wise operations with alignment and fill control.

---

### Index Attributes

| Attribute         | Description                       |
|------------------|-----------------------------------|
| `df.index`       | Row index                         |
| `df.columns`     | Column index                      |
| `df.axes`        | `[rows, columns]` list            |
| `df.shape`       | (rows, columns)                   |
| `df.ndim`        | Number of dimensions              |

---

### Index Modification

```python
df.index = ["row1", "row2", "row3"]     # Assign new labels
df.columns = ["col1", "col2"]           # Rename columns
```

---

### Checking Index Properties

```python
df.index.is_unique        # True if all labels are unique
df.index.has_duplicates   # True if duplicates exist
df.index.name             # Name of index
```

---

### Sorting Index

```python
df.sort_index()                         # Sort by row index
df.sort_index(axis=1)                   # Sort by column index
df.sort_index(ascending=False)          # Descending sort
```

---

### Summary Table

| Operation             | Purpose                        | Notes                           |
|-----------------------|--------------------------------|----------------------------------|
| `set_index()`         | Use column(s) as index         | Can create MultiIndex           |
| `reset_index()`       | Move index to column           | `drop=True` to discard          |
| `reindex()`           | Match new labels               | Fills NaN if missing            |
| `loc[]`               | Label-based indexing           | Inclusive slicing               |
| `iloc[]`              | Position-based indexing        | Exclusive slicing               |
| `xs()`                | Cross-section in MultiIndex    | Use `level` to specify depth    |
| `sort_index()`        | Sort index                     | Axis 0 (rows), axis 1 (cols)    |
| `add()`, `sub()` etc. | Aligned math ops               | Use `fill_value` for NaN        |

---

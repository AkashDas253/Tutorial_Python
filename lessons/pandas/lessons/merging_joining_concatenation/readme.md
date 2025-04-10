
## Merging / Joining / Concatenation in Pandas

These operations are used to combine multiple DataFrames into a single one by aligning data based on **index** or **column keys**.

---

### 1. `merge()` – SQL-style Join on Columns

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

**Examples:**

```python
pd.merge(df1, df2, on="id")  # Inner join on 'id'
pd.merge(df1, df2, how="left", left_on="key1", right_on="key2")
pd.merge(df1, df2, how="outer", on=["key1", "key2"])
```

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

### 2. `.join()` – Join on Index or Column

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

---

### 3. `concat()` – Concatenation Along a Particular Axis

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

---

### 4. `append()` (Deprecated)

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

## Merging in Pandas

Merging allows combining rows from two or more DataFrames based on key columns or indexes, similar to SQL joins.

---

### `merge()` – SQL-style Join on Columns

```python
pd.merge(
    left,                         # Left DataFrame
    right,                        # Right DataFrame
    how="inner",                  # Join type: 'inner', 'outer', 'left', 'right', 'cross'
    on=None,                      # Common column(s) to join on
    left_on=None,                 # Key column(s) from left DF
    right_on=None,                # Key column(s) from right DF
    left_index=False,             # Use left index for merging
    right_index=False,            # Use right index for merging
    sort=False,                   # Sort merge output
    suffixes=("_x", "_y"),        # Suffixes for overlapping column names
    copy=True,                    # Copy data (default: True)
    indicator=False,              # Add a merge source indicator column
    validate=None                 # Check merge format: '1:1', '1:m', 'm:1', 'm:m'
)
```

#### **Examples:**

```python
pd.merge(df1, df2, on="id")                                      # Inner join on 'id'
pd.merge(df1, df2, how="left", left_on="key1", right_on="key2")  # Left join on different keys
pd.merge(df1, df2, how="outer", on=["key1", "key2"])             # Outer join on multiple keys
pd.merge(df1, df2, left_index=True, right_index=True)            # Join on indexes
pd.merge(df1, df2, how="cross")                                  # Cartesian product
pd.merge(df1, df2, indicator=True)                               # Add _merge column
```

---

### Common Merge Use Cases

| Method | Description |
|--------|-------------|
| `pd.merge(df1, df2, on='key')` | Merge on a shared key column |
| `pd.merge(df1, df2, how='inner')` | Default join, returns only matches |
| `pd.merge(df1, df2, how='outer')` | All rows from both DataFrames |
| `pd.merge(df1, df2, how='left')` | All rows from left, matched from right |
| `pd.merge(df1, df2, how='right')` | All rows from right, matched from left |
| `pd.merge(df1, df2, left_on='key1', right_on='key2')` | Join with different key names |
| `pd.merge(df1, df2, left_index=True, right_index=True)` | Merge on DataFrame indexes |
| `pd.merge(df1, df2, indicator=True)` | Track row origin using `_merge` column |
| `pd.merge(df1, df2, validate='1:1')` | Raise error if not one-to-one match |

---

### Join Types (for `how` parameter)

| Join Type | Description |
|-----------|-------------|
| `inner` | Only matching keys from both DataFrames |
| `outer` | All keys, fill missing with `NaN` |
| `left` | All keys from left, with matches from right |
| `right` | All keys from right, with matches from left |
| `cross` | Cartesian product of rows (no key match needed) |

---

### Merge Indicator Values

Used when `indicator=True`:

| `_merge` Value | Description |
|----------------|-------------|
| `'left_only'` | Row appeared only in left DataFrame |
| `'right_only'` | Row appeared only in right DataFrame |
| `'both'` | Row appeared in both DataFrames |

---

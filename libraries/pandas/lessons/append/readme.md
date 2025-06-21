## `append()` (Deprecated) – Adding Rows to DataFrame

The `append()` method was used to add rows to a DataFrame, but it has been deprecated in recent versions of Pandas. The recommended approach is to use `pd.concat()` instead.

---

### `append()` – Syntax and Parameters

```python
df1.append(
    df2,                           # DataFrame or Series to append
    ignore_index=False,            # Ignore original index (reindex)
    verify_integrity=False,        # Check for index duplication
    sort=False                     # Sort columns in resulting DataFrame
)
```

#### **Example:**

```python
df1.append(df2, ignore_index=True)  # Adds rows from df2 to df1, reindexing the result
```

> **Deprecated:** The `append()` method is now deprecated and will be removed in future versions of Pandas. It's recommended to use `pd.concat()` instead.

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

When working with merge, join, or concatenation, duplicates can be a concern. You can use the `drop_duplicates()` method to remove duplicate rows from your resulting DataFrame.

| Method | Description |  
|--------|-------------|  
| `df.drop_duplicates()` | Removes duplicate rows from a DataFrame |  
| `df1.merge(df2, on='key', how='inner').drop_duplicates()` | Removes duplicates after merging two DataFrames |  
| `df1.append(df2).drop_duplicates()` | Removes duplicates after appending DataFrames (if using `append()` before it was deprecated) |  

---

### Summary of Join and Combine Methods

- **`merge()`** – SQL-style join on columns or indices.
- **`join()`** – Used to join DataFrames based on index or specified columns.
- **`concat()`** – Concatenate DataFrames along rows or columns, replacing `append()`.
- **`append()`** – Deprecated for adding rows, use `concat()` instead.

---

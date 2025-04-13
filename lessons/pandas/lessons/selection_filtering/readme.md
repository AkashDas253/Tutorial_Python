## Selection & Filtering in Pandas

---

### Column Selection
```python
df["col"]              # Single column as Series
df[["col1", "col2"]]    # Multiple columns as DataFrame
```
- Dot notation (`df.col`) only for simple column names.

---

### Row Selection
```python
df[5:10]              # Slice rows by integer index
```

---

### `loc[]` – Label-Based Selection & Slicing
```python
df.loc["row_label"]              # Select a row by index label
df.loc[:, ["col1", "col2"]]      # Select all rows for specific columns
df.loc["row1":"row3"]           # Slice rows by labels (inclusive)
df.loc["row1":"row3", "col"]    # Slice rows and select specific column
```
- `loc[]` is label-based and supports slicing (inclusive).
- Syntax: `df.loc[row_labels, column_labels]`

---

### `iloc[]` – Position-Based Selection & Slicing
```python
df.iloc[position]               # Select a row by position
df.iloc[:, [0, 2]]              # Select all rows for specific columns by index
df.iloc[1:4]                    # Slice rows by positions (exclusive of stop)
df.iloc[0:3, 1]                 # Slice rows and select column by index
```
- `iloc[]` is position-based and supports slicing (exclusive).
- Syntax: `df.iloc[row_positions, column_positions]`

---

### `at[]` and `iat[]` – Fast Scalar Access
```python
df.at[label, "col"]           # Label-based scalar access
df.iat[2, 3]                  # Position-based scalar access
```
✅ Faster for accessing/modifying a single value.

---

### Boolean Filtering
```python
df[df["col"] > 100]
df[df["col"].isin([10, 20])]
df[df["col"].str.contains("abc")]
```
✅ Use boolean masks to filter rows.

---

### Logical Conditions
```python
df[(df["A"] > 10) & (df["B"] < 5)]
df[(df["A"] == 1) | (df["B"] == 2)]
~df["A"].isnull()   # NOT operator
```
| Operator | Meaning |
|----------|---------|
| `&`      | AND     |
| `|`      | OR      |
| `~`      | NOT     |
📌 Use parentheses `()` around each condition.

---

### `query()` Method
```python
df.query("A > 10 and B < 5")
```
✅ Simplified string-based filtering.
- Use backticks for column names with spaces: `` `col name` ``

---

### `isin()` and `between()`
```python
df[df["col"].isin([1, 2, 3])]
df[df["col"].between(10, 20)]
```
✅ For checking membership or value range.

---

### `str` Accessor for String Filtering
```python
df[df["col"].str.startswith("abc")]
df[df["col"].str.endswith("xyz")]
df[df["col"].str.contains("test", na=False)]
```
- Use `.na=False` to ignore NaNs.

---

### `where()` and `mask()`
```python
df.where(df["col"] > 0, other=0)
df.mask(df["col"] < 0, other=0)
```
- `where()`: Keep where condition is True.
- `mask()`: Replace where condition is True.

---

### Filtering with `any()` / `all()`
```python
df[df[["A", "B"]] > 0].all(axis=1)  # All True row-wise
df[df[["A", "B"]] > 0].any(axis=1)  # At least one True row-wise
```

---

### Filtering Nulls
```python
df[df["col"].isnull()]
df[df["col"].notnull()]
```
- `isna()` is an alias for `isnull()`

---

### Drop Rows/Columns by Condition
```python
df[df["col"] != "drop_this"]
df.dropna(subset=["col"])
df.drop(columns=["col"])
```

---

### Selecting Columns by `dtype`
```python
df.select_dtypes(include="number")
df.select_dtypes(exclude=["object", "bool"])
```
✅ Filter columns by data type.

---

### Conditional Assignment
```python
df.loc[df["col"] < 0, "col"] = 0
```
✅ Modify values based on condition.

---

### Negation & Inversion
```python
df[~df["col"].isin([1, 2])]
df[~df["col"].str.contains("xyz", na=False)]
```

---

### `filter()` Method
```python
df.filter(items=["col1", "col2"])
df.filter(like="2022", axis=1)
df.filter(regex="^col[0-9]+$", axis=1)
```
- Filter by name, pattern, or list.

---

### `xs()` – Cross-section Selection
```python
df.xs("row_label")
df.xs("col_label", axis=1)
```
- Select row/column across levels (multi-index).

---

### Filtering by `.index` and `.columns`
```python
df[df.index > "2023-01-01"]
df[df.columns[df.columns.str.contains("sales")]]
```
- Filter by label in index or column.

---

### `eval()` Expression-Based Filtering
```python
df.eval("new_col = A + B")
df[df.eval("A < B and C > 10")]
```
- Evaluate expressions using column names.

---

### ✅ Summary Table
| Method            | Type              | Use Case             |
|-------------------|-------------------|----------------------|
| `df[col]`         | Column select     | One column           |
| `df[...]`         | Boolean filtering | Row filter           |
| `loc[]`           | Label-based       | Row/column selection |
| `iloc[]`          | Position-based    | Row/column selection |
| `at[]`, `iat[]`   | Fast scalar       | Single value access  |
| `query()`         | String filtering  | Readable conditions  |
| `eval()`          | Expression filter | Computed conditions  |
| `where()`         | Conditional mask  | Replace False values |
| `isin()`          | Membership        | Match list           |
| `between()`       | Range check       | Value range filter   |
| `str.contains()`  | Text match        | Substring filter     |
| `select_dtypes()` | Data type filter  | Filter columns       |
| `filter()`        | Name-based filter | Column/row names     |
| `xs()`            | Cross-section     | Label-level select   |

---

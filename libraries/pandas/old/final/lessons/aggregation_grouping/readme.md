## Aggregation & Grouping in Pandas

---

### `groupby()`

Group data based on column values or index.

```python
df.groupby(by=None, axis=0, level=None, as_index=True, sort=True,
           group_keys=True, observed=False, dropna=True)
```

| Parameter     | Description |
|---------------|-------------|
| `by`          | Column label(s) or mapping to group by |
| `axis`        | Axis to group along (0 = index, 1 = columns) |
| `level`       | Group by specific index level(s) |
| `as_index`    | If True, keys become index in result |
| `sort`        | Sort group keys (default True) |
| `group_keys`  | Add group keys to result index |
| `observed`    | For categorical grouping, include only observed combinations |
| `dropna`      | Drop NA groups if True |

---

### **Syntax for `agg` and `aggregate` in Pandas**

### **For DataFrame**
```python
df.agg(func, axis=0, *args, **kwargs)
df.aggregate(func, axis=0, *args, **kwargs)
```

- **Parameters**:
  - `func`: Single function, list of functions, or dictionary mapping columns to functions.
  - `axis`: `0` (default) for columns, `1` for rows.
  - `*args, **kwargs`: Additional arguments passed to the aggregation functions.

---

### **For Series**
```python
series.agg(func, *args, **kwargs)
series.aggregate(func, *args, **kwargs)
```

- **Parameters**:
  - `func`: Single function or list of functions.
  - `*args, **kwargs`: Additional arguments passed to the aggregation functions.

---

### Common Aggregation Functions

Use directly after `.groupby()`:

```python
df.groupby("col").agg_func()
```

| Function | Description |
|----------|-------------|
| `sum()`  | Sum of values |
| `mean()` | Mean of values |
| `median()` | Median value |
| `min()` / `max()` | Min / Max values |
| `count()` | Count non-NA values |
| `size()` | Count of all entries |
| `std()` / `var()` | Standard deviation / Variance |
| `first()` / `last()` | First / Last non-NA value |
| `nunique()` | Number of unique values |
| `any()` / `all()` | Any True / All True |

---

### `agg()` – Multiple Aggregations

```python
df.groupby("col").agg(["mean", "sum"])
df.groupby("col").agg({"val1": "sum", "val2": "mean"})
```

✅ Use for:
- Multiple aggregation functions
- Column-wise control

---

### `agg()` with Named Aggregation (Pandas 0.25+)

```python
df.groupby("col").agg(
    avg_val=("val", "mean"),
    total_val=("val", "sum")
)
```

Clear way to assign custom names to aggregated outputs.

---

### `apply()` – Custom Row/Group Function

```python
df.groupby("col").apply(lambda group: custom_func(group))
```

- Returns arbitrary shaped output
- Slower but more flexible than `agg()`

---

### `transform()` – Return Same Shape

```python
df.groupby("col")["val"].transform("mean")
```

✅ Use when:
- Output shape must match original
- Useful for creating new columns with group-based values

```python
df["group_mean"] = df.groupby("col")["val"].transform("mean")
```

---

### `filter()` – Filter Groups by Condition

```python
df.groupby("col").filter(lambda x: len(x) > 2)
```

Keeps only groups that satisfy condition.

---

### `get_group()` – Access Group Directly

```python
group = df.groupby("col").get_group("A")
```

Returns a DataFrame for that group.

---

### `describe()` – Summary Stats Per Group

```python
df.groupby("col")["val"].describe()
```

Returns count, mean, std, min, quartiles, and max.

---

### Chaining with `pipe()`

```python
df.groupby("col").pipe(my_custom_function)
```

Improves readability when applying custom logic.

---

### Grouping by Multiple Columns

```python
df.groupby(["col1", "col2"]).sum()
```

- Results in hierarchical index (MultiIndex)
- Use `reset_index()` to flatten

---

### Grouping by Index Levels

```python
df.groupby(level=0).sum()
```

Use `level=` if DataFrame has multi-level index.

---

### Time Series Grouping with `resample()`

```python
df.set_index("date").groupby("category").resample("M").sum()
```

Groups time-based data for each category.

---

### Grouping with Categorical Columns

```python
df.groupby("cat_col", observed=True).sum()
```

| Parameter     | Description |
|---------------|-------------|
| `observed=True` | Only show combinations that actually appear |
| `observed=False` | Include all combinations (slower) |

---

### Group Keys Behavior

```python
df.groupby("col", group_keys=False).apply(...)
```

Disables adding group keys to the index in `.apply()`.

---

### `dropna` in `groupby()`

```python
df.groupby("col", dropna=False).sum()
```

Includes NA groups if `dropna=False`.

---

### Reshaping After Grouping

- **Unstack**: pivot group keys to columns
  ```python
  df.groupby(["A", "B"]).sum().unstack()
  ```

- **Stack**: compress multi-index into single axis
  ```python
  df.groupby(["A", "B"]).sum().unstack().stack()
  ```

- **Reset index**: flatten result
  ```python
  df.groupby("A").sum().reset_index()
  ```

---

### Rolling & Expanding (Grouped Window Ops)

```python
df.groupby("group")["value"].rolling(window=3).mean()
df.groupby("group")["value"].expanding().mean()
```

Used for moving window calculations within each group.

---

### ✅ Summary Table

| Method      | Shape Change | Use Case |
|-------------|--------------|----------|
| `agg()`     | Aggregated   | Multiple functions |
| `apply()`   | Flexible     | Custom functions |
| `transform()` | Same shape | Add column with group-wise values |
| `filter()`  | Filtered rows | Keep/remove groups |

---

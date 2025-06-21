
## Sorting & Ranking in Pandas

Sorting and ranking allow you to organize and prioritize data by index, values, or custom logic.

---

### Sorting

---

#### Sorting by Index

```python
df.sort_index(
    axis=0,                  # 0 for rows, 1 for columns
    level=None,              # Index level(s) to sort on (for MultiIndex)
    ascending=True,          # Sort in ascending order
    inplace=False,           # Modify original object
    kind="quicksort",        # Sorting algorithm
    na_position="last",      # 'first' or 'last' for NaNs
    sort_remaining=True,     # For MultiIndex: sort unspecified levels
    ignore_index=False       # Reset index in result
)
```

**Example:**

```python
df.sort_index()                      # Sort rows by index
df.sort_index(axis=1)                 # Sort columns by column labels
```

---

#### Sorting by Values

```python
df.sort_values(
    by,                            # Column label(s) to sort by
    axis=0,                        # 0 for rows
    ascending=True,                # Sort order
    inplace=False,                 # Modify original object
    kind="quicksort",              # Sorting algorithm
    na_position="last",            # Where to place NaNs
    ignore_index=False,            # Reset index in result
    key=None                       # Optional function applied before sorting
)
```

**Example:**

```python
df.sort_values(by="col1", ascending=False)
df.sort_values(by=["col1", "col2"], ascending=[True, False])
```

---

### Ranking

Ranks assign numerical values indicating position relative to other values.

```python
df["ranked"] = df["col"].rank(
    method="average",             # 'average', 'min', 'max', 'first', 'dense'
    ascending=True,               # Rank in ascending order
    na_option="keep",             # 'keep', 'top', 'bottom'
    pct=False,                    # Return rank as percentage
    axis=0                        # Axis to rank along
)
```

**Example:**

```python
df["rank_avg"] = df["score"].rank()  # Average rank
df["rank_dense"] = df["score"].rank(method="dense")  # Dense rank
```

---

### Rank Method Options

| Method     | Description                                |
|------------|--------------------------------------------|
| `average`  | Average of ranks for ties (default)        |
| `min`      | Lowest rank in the group                   |
| `max`      | Highest rank in the group                  |
| `first`    | Order of appearance in dataset             |
| `dense`    | Like `min`, but ranks increase by 1 only   |

---

### Additional Options

| Parameter     | Description                                 | Values                          |
|---------------|---------------------------------------------|----------------------------------|
| `na_position` | Position of NaN in sort                     | `'first'`, `'last'`             |
| `inplace`     | Apply changes to same object                | `True`, `False` (default)       |
| `key`         | Callable applied to values before sorting   | `lambda x: x.str.lower()`       |
| `ascending`   | Sort order                                  | `True`, `False`, or list thereof|
| `ignore_index`| Reindex output DataFrame                    | `True`, `False`                 |

---

### Use in Series

```python
s.sort_values(ascending=True)
s.sort_index()
s.rank()
```

---

### Sorting with MultiIndex

```python
df.sort_index(level=[0, 1], ascending=[True, False])
df.sort_values(by=[("level1", "col1")])
```

---

### Sorting with Date/Time Data

Sorting with datetime values is often essential for time-series data or chronological events. `pandas` handles `datetime` objects well and allows sorting with time-related precision.

```python
df.sort_values(by="date_column", ascending=True)
```

- **`datetime` sorting**: If you have a datetime column, sorting will arrange the data chronologically.

---

### Sorting and Ranking in Grouped Data

When working with grouped data, you can apply sorting and ranking operations within each group separately using `groupby`.

#### Sorting within Groups

You can sort within each group in a `groupby` object by using `apply()` along with `sort_values()`.

```python
df.groupby("group_column").apply(lambda x: x.sort_values("score"))
```

#### Ranking within Groups

Similarly, you can rank data within each group.

```python
df["group_rank"] = df.groupby("group_column")["score"].rank(method="average")
```

- **`groupby`**: This allows you to apply sorting and ranking operations on a per-group basis.

---

### Sorting by Multiple Columns

You can sort by multiple columns to provide more granularity in the sorting process. This is helpful when the primary sort is tied, and you want to apply a secondary sort criterion.

```python
df.sort_values(by=["col1", "col2"], ascending=[True, False])
```

- **`ascending`**: A list can be passed where each entry corresponds to the sort order for the respective column in `by`.

---

### Sorting by Custom Key with Lambda Functions

You can use a custom sorting key by applying a function (like a lambda) to transform the values before sorting.

```python
df.sort_values(by="col1", key=lambda x: x.str.upper())  # Sort alphabetically, ignoring case
```

- **`key`**: This allows custom sorting logic, such as sorting strings in uppercase or dates in different formats.

---

### Stable Sorting

Pandas supports stable sorting, meaning if two rows have the same value in the column you're sorting by, their order relative to each other will be preserved.

```python
df.sort_values(by="col1", ascending=True)
```

- **Stable Sort**: In the case of ties, the order remains as it was before the sort, ensuring consistency when dealing with duplicates.

---

### Sorting and Ranking on Categorical Data

Pandas also supports sorting and ranking for categorical data, where categories are ordered based on predefined criteria.

```python
df["cat_col"] = pd.Categorical(df["cat_col"], categories=["low", "medium", "high"], ordered=True)
df.sort_values(by="cat_col", ascending=True)
```

- **`Categorical` data**: Sorting and ranking can respect the predefined order of categories.

---

### Performance Considerations for Sorting and Ranking

- **Time Complexity**: Sorting in pandas typically has a time complexity of O(n log n), where `n` is the number of rows.
- **Memory**: Sorting and ranking operations can be memory-intensive, especially with large datasets. Using `inplace=True` can help to reduce memory usage by modifying the original DataFrame.

---

### Ranking with `tied` Parameter

`rank()` has the `tied` parameter that controls how ties are handled.

```python
df["ranked"] = df["col"].rank(method="min", ties="random")  # Randomly assign ranks in case of ties
```

- **`tied`**: Can be set to `'random'` or `'first'`, to manage ties randomly or based on their order in the DataFrame.

---

### Ranking with Percentiles

You can rank values based on their percentile rather than absolute values:

```python
df["percentile_rank"] = df["col"].rank(pct=True)
```

- **`pct=True`**: This returns the rank as a percentage of the total, with the highest value getting `1.0` and the lowest `0.0`.

---

### Combining Sorting with Aggregations

Sorting can be combined with aggregation functions like `sum()`, `mean()`, `max()`, etc., to first sort the data and then apply aggregation.

```python
df.sort_values(by="score", ascending=False).groupby("group_col")["score"].mean()
```

- **Aggregation after sorting**: Sorting can help organize the data before applying group-wise aggregation.

---

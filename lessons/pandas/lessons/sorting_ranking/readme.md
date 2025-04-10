
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
df.sort_index(axis=1)               # Sort columns by column labels
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

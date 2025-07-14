
## Looping / Iteration in Pandas

Pandas is designed for **vectorized operations**, but in some situations, **iteration is necessary**—especially for row-wise logic, exporting, or custom updates.

---

### Iterating Over Rows

| Method | Description | Syntax / Example |
|--------|-------------|------------------|
| `df.iterrows()` | Iterates over rows as (index, Series) |  
```python
for index, row in df.iterrows():
    print(index, row['col1'])  # row is a Series
```  
|
| `df.itertuples(index=True, name='Pandas')` | Iterates over rows as namedtuples (faster) |  
```python
for row in df.itertuples():
    print(row.Index, row.col1)  # dot notation
```  
|

**Notes**:  
- `itertuples()` is **faster and more memory-efficient** than `iterrows()`  
- `iterrows()` allows **mutable row access**, `itertuples()` gives **immutable namedtuples**

---

### Iterating Over Columns

| Method | Description | Syntax / Example |
|--------|-------------|------------------|
| `for col in df:` | Iterates over column names |  
```python
for col in df:
    print(col)
```  
|
| `df.items()` | Iterates over (column_name, Series) pairs |  
```python
for col_name, col_data in df.items():
    print(col_name, col_data.mean())
```  
|

---

### Iterating Over Series

| Method | Description | Syntax / Example |
|--------|-------------|------------------|
| `for val in s:` | Simple iteration over Series values |  
```python
for val in df['col']:
    print(val)
```  
|
| `s.iteritems()` | Iterates over (index, value) pairs |  
```python
for idx, val in df['col'].items():
    print(idx, val)
```  
|

---

### Vectorized Alternative (Recommended)

Instead of looping, prefer vectorized operations whenever possible:

```python
# Instead of looping to add 10
df['col'] = df['col'] + 10
```

If logic is complex, use `apply()`:

```python
df['col2'] = df['col1'].apply(lambda x: x ** 2 if x > 0 else 0)
```

---

### Conditional Logic With Iteration

```python
results = []
for i, row in df.iterrows():
    if row['score'] > 50:
        results.append("Pass")
    else:
        results.append("Fail")

df['result'] = results
```

---

### Progress Tracking with TQDM

For large iterations:

```python
from tqdm import tqdm
for i, row in tqdm(df.iterrows(), total=len(df)):
    # do something
```

---

### When to Use Which

| Use Case | Preferred Method |
|----------|------------------|
| Simple row access | `itertuples()` |
| Column iteration | `items()` |
| Modify based on row | `iterrows()` or `apply()` |
| Performance-sensitive | Vectorized or `apply()` |
| Parallel or large dataset | `swifter`, `dask`, or `modin` for better performance |

---

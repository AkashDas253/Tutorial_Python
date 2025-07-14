## Optimization in Pandas 

### 📌 Objective  
Optimization in Pandas focuses on **speeding up computation**, **reducing memory usage**, and **making data manipulation more efficient** for large datasets.

---

### 🧠 Key Optimization Areas

- **Memory Efficiency**
- **Computation Speed**
- **Efficient I/O Operations**
- **Vectorized Operations**
- **Avoiding Inefficient Loops**
- **Indexing and Filtering Optimization**
- **Data Type Optimization**

---

### 🧮 Memory Optimization Techniques

| Technique                       | Description                                                        |
|---------------------------------|--------------------------------------------------------------------|
| `astype()`                      | Convert to more memory-efficient dtypes (e.g., `float64` → `float32`) |
| Use `category` dtype            | For repetitive string/object columns                               |
| `downcast` in `pd.to_numeric()` | Reduce numeric precision safely                                    |
| Drop unused columns             | Reduces memory and speeds up ops                                   |
| Load subset with `usecols`      | Read only needed columns from disk                                 |
| Load data in `chunksize`        | Process large files incrementally                                  |

#### Example:
```python
df['col'] = df['col'].astype('float32')
df['type'] = df['type'].astype('category')
```

---

### ⚡ Speed Optimization – Vectorization

| Technique           | Description                                 |
|---------------------|---------------------------------------------|
| Vectorized Ops      | Use built-in Pandas/Numpy functions         |
| Avoid `for` loops   | Prefer broadcasting and column-wise ops     |
| Apply `.where()`    | Efficient conditional operations            |
| Use `.query()`      | Faster filtering using string expressions   |
| `.eval()`           | Evaluate expressions with reduced overhead  |

#### Example:
```python
# Bad (slow)
df['new'] = df['a'] + df['b']

# Better
df.eval("new = a + b", inplace=True)
```

---

### 🔎 Efficient Filtering and Indexing

| Technique                | Description                                   |
|--------------------------|-----------------------------------------------|
| Use `.loc[]` and `.iloc[]`| For label- or position-based access         |
| Set Index with `set_index()` | Improves lookup with `.loc[]`             |
| Boolean indexing         | Vectorized filtering                         |
| Avoid chained indexing   | Use `.loc[]` instead of df[df['x'] > 0]['y'] |

#### Example:
```python
df = df.set_index('id')          # Faster access via .loc[id]
df = df[df['value'] > 0]         # Boolean filter
```

---

### 🧵 Efficient Grouping and Aggregation

| Optimization             | Description                                |
|--------------------------|--------------------------------------------|
| Use `groupby().agg()`    | Combine aggregations efficiently           |
| Avoid `apply()` in groups| Use named aggregates instead               |
| Use `sort=False`         | Avoid unnecessary sort during `groupby()`  |

#### Example:
```python
df.groupby('col', sort=False).agg({'value': 'sum'})
```

---

### 📂 Input/Output Optimization

| Method                        | Optimization                            |
|-------------------------------|------------------------------------------|
| Use `usecols`, `dtype`        | While reading CSV/Excel                 |
| Use binary formats (Parquet)  | Faster read/write than CSV              |
| Use `chunksize`               | Process large files in parts            |
| Compress output files         | Save disk space                         |

#### Example:
```python
df.to_parquet('data.parquet')   # Faster and smaller
```

---

### 📋 Handling Large Datasets

| Method                      | Description                                |
|-----------------------------|--------------------------------------------|
| Load in chunks              | Prevent memory overload                    |
| Process and discard chunks  | Avoid building full DataFrame              |
| Use `Dask` for distributed  | For very large data (> RAM size)           |

---

### 🔁 Apply vs Vectorization vs Numba

| Method     | Speed       | Notes                                      |
|------------|-------------|--------------------------------------------|
| `.apply()` | Slow        | Flexible but not optimized                 |
| Vectorized | Fast        | Best for most operations                   |
| Numba      | Very Fast   | Requires @jit; useful for custom functions |

#### Example:
```python
from numba import jit

@jit
def fast_function(x):
    return x * x
```

---

### 🧠 Best Practices Summary

- Prefer **vectorized operations** over row-wise or element-wise loops.
- Use **appropriate dtypes** (`int8`, `float32`, `category`) to save memory.
- Avoid `apply()` unless necessary; replace with `map()`, `agg()`, or built-in ops.
- Optimize **groupby** with minimal functions and sort disabled.
- Use efficient file formats like **Parquet** or **Feather**.
- Minimize chained operations and intermediate copies.

---

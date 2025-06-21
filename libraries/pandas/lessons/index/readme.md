
# Pandas Index

---

## Definition
An `Index` in pandas is an immutable, ordered sequence of labels used to identify rows and columns in Series and DataFrames. It's the foundation of label-based data access and alignment.

---

## Properties

| Property         | Description                                                      |
|------------------|------------------------------------------------------------------|
| Immutable        | Cannot be changed after creation (some exceptions exist)         |
| Ordered          | Keeps the order of elements                                      |
| Homogeneous      | All elements are of the same type                                |
| Uniqueness       | Can be enforced using `is_unique`                                |
| Supports slicing | Label and position-based slicing supported                       |
| Backbone         | Used for alignment and identification in Series/DataFrame        |
| Labeling          | Used to label rows or columns in a DataFrame or Series, allowing for intuitive access. |
| Efficient Lookups | Provides optimized performance for lookups, alignments, and merges. |
| Versatility       | Supports various types of labels, including integers, strings, datetime objects, and multi-level indices. |

---

## Declaration Syntax

```python
pd.Index(
    data,                   # Iterable, array, or scalar values
    dtype=None,             # Optional data type (e.g., int, float, str)
    copy=False,             # Copy data if True
    name=None,              # Optional name for index
    tupleize_cols=True      # For MultiIndex; deprecated in newer versions
)
```

### Types of Index  

| Type                     | Description                                      |  
|--------------------------|--------------------------------------------------|  
| `RangeIndex`             | Default integer-based index (0, 1, 2, ...).      |  
| `Int64Index`             | Index with integer values.                       |  
| `Float64Index`           | Index with float values.                         |  
| `DatetimeIndex`          | Index with datetime values.                      |  
| `TimedeltaIndex`         | Index with timedelta values.                     |  
| `CategoricalIndex`       | Index with categorical values.                   |  
| `MultiIndex`             | Hierarchical index for multi-level indexing.     |  
| `Index`                  | Generic index for object-based values.           |  

### Key Index Operations

| **Operation**       | **Description**                                           |
|---------------------|-----------------------------------------------------------|
| **Index Creation**   | Create index using `pd.Index()`, `pd.date_range()`, `pd.MultiIndex()`, etc. |
| **Access Data**      | Access elements using `.loc[]` (label-based) or `.iloc[]` (integer-based). |
| **Reindexing**       | Reorder or modify the index using `.reindex()`.           |
| **Alignment**        | Automatic alignment of data based on matching indices during operations. |
| **Setting Index**    | Use `.set_index()` to set a column as the index.          |
| **Resetting Index**  | Use `.reset_index()` to revert back to default integer indexing. |

---

## Operations with Index  

| Operation                     | Description                                      |  
|-------------------------------|--------------------------------------------------|  
| `df.loc[label]`               | Access row(s) by label (label-based indexing).   |  
| `df.iloc[position]`           | Access row(s) by position (integer-based indexing). |  
| `df.reindex(new_index)`       | Reindex DataFrame or Series to match a new index. |  
| `df.set_index('column_name')` | Set a column as the index of the DataFrame.      |  
| `df.reset_index()`            | Reset the index to default integer-based index.  |  

#### Example: Accessing Data  

```python
# Accessing using .loc[] (label-based indexing)
print(df.loc[0])  # Access row with label 0

# Accessing using .iloc[] (integer-based indexing)
print(df.iloc[0])  # Access first row by position
```

#### Example: Reindexing  

```python
# Reindexing example
new_index = [2, 0, 1]
df_reindexed = df.reindex(new_index)
print(df_reindexed)
```

#### Example: Index Alignment  

```python
# Example of index alignment
df1 = pd.DataFrame({'A': [1, 2, 3]}, index=['a', 'b', 'c'])
df2 = pd.DataFrame({'A': [10, 20, 30]}, index=['a', 'b', 'c'])
result = df1 + df2
print(result)
```

#### Example: Setting and Resetting Index  

```python
# Setting and resetting index
df_reset = df.reset_index()  # Resets the index
df_set = df.set_index('column_name')  # Sets a column as the index
```

---

## All Features by Category with Syntax and Descriptions

---

### Attributes

| Syntax         | Description                              |
|----------------|------------------------------------------|
| `idx.dtype`    | Data type of index values                |
| `idx.shape`    | Tuple: (length,)                         |
| `idx.size`     | Number of elements                       |
| `idx.name`     | Name of the index                        |
| `idx.nbytes`   | Memory usage in bytes                    |
| `idx.ndim`     | Number of dimensions (always 1)          |
| `idx.empty`    | Returns True if Index is empty           |
| `idx.is_unique`| True if all labels are unique            |
| `idx.hasnans`  | True if there are NaN values             |

---

### Access & Indexing

| Syntax               | Description                              |
|----------------------|------------------------------------------|
| `idx[i]`             | Get item at position `i`                 |
| `idx[1:4]`           | Slice between positions                  |
| `idx[-1]`            | Last item                                |
| `idx.get_loc(label)` | Get integer location of label            |
| `idx.get_indexer(target)` | Get index positions for `target`   |

---

### Searching & Filtering

| Syntax                            | Description                                  |
|-----------------------------------|----------------------------------------------|
| `idx.isin([val1, val2])`          | Check if elements are in a list              |
| `idx.where(condition)`            | Mask elements based on condition             |
| `idx.equals(other)`               | Compare two Index objects                    |
| `idx.duplicated()`                | Check for duplicates                         |
| `idx.unique()`                    | Unique values                                |
| `idx.contains(value)` *(via `in`)| Check if label exists (`'a' in idx`)         |

---

### Modification (Immutable Output)

| Syntax                                 | Description                            |
|----------------------------------------|----------------------------------------|
| `idx.append(other)`                    | Concatenate two indexes                |
| `idx.insert(loc, item)`                | Insert item at position (returns new)  |
| `idx.delete(loc)`                      | Remove item at position (new Index)    |
| `idx.drop(labels)`                     | Drop specified labels                  |
| `idx.rename(name)`                     | Rename index                           |
| `idx.repeat(n)`                        | Repeat index values                    |
| `idx.map(func)`                        | Apply function to each element         |

---

### Sorting & Reordering

| Syntax                          | Description                              |
|----------------------------------|------------------------------------------|
| `idx.sort_values()`             | Sort values ascending                    |
| `idx.sort_values(ascending=False)` | Descending sort                        |
| `idx.argsort()`                 | Return indices for sorting               |
| `idx[::-1]`                     | Reverse order                            |

---

### Type Conversion

| Syntax                  | Description                         |
|--------------------------|-------------------------------------|
| `idx.astype(dtype)`     | Convert to another data type        |
| `idx.to_list()`         | Convert to Python list              |
| `idx.to_numpy()`        | Convert to NumPy array              |

---

### Set Operations

| Syntax                            | Description                                |
|-----------------------------------|--------------------------------------------|
| `idx.union(other)`                | Union of indexes                           |
| `idx.intersection(other)`         | Common values                              |
| `idx.difference(other)`           | Values in `idx` not in `other`             |
| `idx.symmetric_difference(other)` | Exclusive values in each                   |
| `idx.equals(other)`               | Check exact equality                       |

---

### Functional Utilities

| Syntax                  | Description                        |
|--------------------------|------------------------------------|
| `idx.any()`             | True if any element is True        |
| `idx.all()`             | True if all elements are True      |
| `idx.argmax()`          | Position of max value              |
| `idx.argmin()`          | Position of min value              |

---

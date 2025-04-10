
# **Pandas Index**

---

## **Definition**
An `Index` in pandas is an **immutable, ordered sequence** of labels used to identify rows and columns in Series and DataFrames. It's the foundation of label-based data access and alignment.

---

## **Properties**

| Property         | Description                                                      |
|------------------|------------------------------------------------------------------|
| Immutable        | Cannot be changed after creation (some exceptions exist)         |
| Ordered          | Keeps the order of elements                                      |
| Homogeneous      | All elements are of the same type                                |
| Uniqueness       | Can be enforced using `is_unique`                                |
| Supports slicing | Label and position-based slicing supported                       |
| Backbone         | Used for alignment and identification in Series/DataFrame        |

---

## **Declaration Syntax**

```python
pd.Index(
    data,                   # Iterable, array, or scalar values
    dtype=None,             # Optional data type (e.g., int, float, str)
    copy=False,             # Copy data if True
    name=None,              # Optional name for index
    tupleize_cols=True      # For MultiIndex; deprecated in newer versions
)
```

---

## **All Features by Category with Syntax and Descriptions**

---

### **1. Attributes**

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

### **2. Access & Indexing**

| Syntax               | Description                              |
|----------------------|------------------------------------------|
| `idx[i]`             | Get item at position `i`                 |
| `idx[1:4]`           | Slice between positions                  |
| `idx[-1]`            | Last item                                |
| `idx.get_loc(label)` | Get integer location of label            |
| `idx.get_indexer(target)` | Get index positions for `target`   |

---

### **3. Searching & Filtering**

| Syntax                            | Description                                  |
|-----------------------------------|----------------------------------------------|
| `idx.isin([val1, val2])`          | Check if elements are in a list              |
| `idx.where(condition)`            | Mask elements based on condition             |
| `idx.equals(other)`               | Compare two Index objects                    |
| `idx.duplicated()`                | Check for duplicates                         |
| `idx.unique()`                    | Unique values                                |
| `idx.contains(value)` *(via `in`)| Check if label exists (`'a' in idx`)         |

---

### **4. Modification (Immutable Output)**

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

### **5. Sorting & Reordering**

| Syntax                          | Description                              |
|----------------------------------|------------------------------------------|
| `idx.sort_values()`             | Sort values ascending                    |
| `idx.sort_values(ascending=False)` | Descending sort                        |
| `idx.argsort()`                 | Return indices for sorting               |
| `idx[::-1]`                     | Reverse order                            |

---

### **6. Type Conversion**

| Syntax                  | Description                         |
|--------------------------|-------------------------------------|
| `idx.astype(dtype)`     | Convert to another data type        |
| `idx.to_list()`         | Convert to Python list              |
| `idx.to_numpy()`        | Convert to NumPy array              |

---

### **7. Set Operations**

| Syntax                            | Description                                |
|-----------------------------------|--------------------------------------------|
| `idx.union(other)`                | Union of indexes                           |
| `idx.intersection(other)`         | Common values                              |
| `idx.difference(other)`           | Values in `idx` not in `other`             |
| `idx.symmetric_difference(other)` | Exclusive values in each                   |
| `idx.equals(other)`               | Check exact equality                       |

---

### **8. Functional Utilities**

| Syntax                  | Description                        |
|--------------------------|------------------------------------|
| `idx.any()`             | True if any element is True        |
| `idx.all()`             | True if all elements are True      |
| `idx.argmax()`          | Position of max value              |
| `idx.argmin()`          | Position of min value              |

---

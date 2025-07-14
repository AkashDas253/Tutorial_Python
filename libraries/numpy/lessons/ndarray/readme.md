
# NumPy ndarray

---

## Definition
An `ndarray` is a multidimensional, fixed-size, homogeneous data container in NumPy. It allows efficient numerical operations and broadcasting, and it forms the base of all numerical computations in NumPy.

---

## Properties

| Property           | Description                                                      |
|--------------------|------------------------------------------------------------------|
| Multidimensional   | Supports 1D, 2D, ..., nD arrays                                   |
| Homogeneous        | All elements must be of the same data type                       |
| Fixed size         | Shape of the array is immutable (though contents can change)     |
| NumPy-based        | Built using C for fast operations                                |
| Broadcasting       | Supports broadcasting rules for arithmetic operations            |

---

## Declaration Syntax

```python
np.array(
    object,              # Input data: list, tuple, nested lists, etc.
    dtype=None,          # Desired data type
    copy=True,           # Copy object if True
    order='K',           # Memory layout: 'C', 'F', 'A', or 'K'
    ndmin=0              # Minimum number of dimensions
)
```

---

## Attributes (Object Metadata)

| Syntax                | Description                                      |
|------------------------|--------------------------------------------------|
| `arr.ndim`            | Number of dimensions                             |
| `arr.shape`           | Tuple of array dimensions                        |
| `arr.size`            | Total number of elements                         |
| `arr.dtype`           | Data type of elements                            |
| `arr.itemsize`        | Byte size of one element                         |
| `arr.nbytes`          | Total bytes consumed                             |
| `arr.T`               | Transpose of the array                           |
| `arr.data`            | Memory buffer (rarely used)                      |
| `arr.strides`         | Steps to move along dimensions in memory         |
| `arr.base`            | Base object if view, else `None`                 |
| `arr.flags`           | Memory layout info (C/F/Contiguous, etc.)        |

---

## Methods (Attached to ndarray objects)

---

### a. Shape & Type Manipulation

| Syntax                    | Description                                  |
|---------------------------|----------------------------------------------|
| `arr.reshape(new_shape)` | Return a new shape without changing data     |
| `arr.resize(new_shape)`  | Resize in-place                              |
| `arr.astype(type)`       | Cast to a different data type                |
| `arr.flatten()`          | Return 1D copy of array                      |
| `arr.ravel()`            | Return 1D view (if possible)                 |
| `arr.squeeze()`          | Remove 1-element dimensions                  |
| `arr.expand_dims(axis)`  | Add dimension at given position              |
| `arr.swapaxes(a, b)`     | Swap two axes                                |
| `arr.transpose(*axes)`   | Permute axes                                 |

---

### b. Data Access & Indexing

| Syntax                      | Description                          |
|-----------------------------|--------------------------------------|
| `arr[index]`                | Basic indexing                       |
| `arr[1:4]`, `arr[:, 0]`     | Slicing                              |
| `arr.take(indices)`         | Take elements by index               |
| `arr.put(indices, values)`  | Set values at specific indices       |
| `arr.item(index)`           | Get a scalar value by index          |
| `arr.tolist()`              | Convert array to nested list         |

---

### c. Mathematical Operations

| Syntax                     | Description                             |
|----------------------------|-----------------------------------------|
| `arr.sum(axis)`            | Sum of elements                        |
| `arr.mean(axis)`           | Mean value                             |
| `arr.std(axis)`            | Standard deviation                     |
| `arr.var(axis)`            | Variance                               |
| `arr.min(axis)`            | Minimum                                |
| `arr.max(axis)`            | Maximum                                |
| `arr.argmin(axis)`         | Index of minimum                       |
| `arr.argmax(axis)`         | Index of maximum                       |
| `arr.cumsum(axis)`         | Cumulative sum                        |
| `arr.cumprod(axis)`        | Cumulative product                    |
| `arr.prod(axis)`           | Product of all elements               |

---

### d. Boolean & Condition Checks

| Syntax                        | Description                            |
|-------------------------------|----------------------------------------|
| `arr.all()`                   | True if all elements are True          |
| `arr.any()`                   | True if any element is True            |
| `arr.nonzero()`               | Indices of non-zero elements           |
| `arr.where(condition)`        | Indices/elements where condition holds |
| `arr.clip(min, max)`          | Limit values to [min, max]             |

---

### e. Copying and Views

| Syntax                  | Description                            |
|-------------------------|----------------------------------------|
| `arr.copy()`            | Deep copy of the array                 |
| `arr.view()`            | Shallow copy (view)                    |

---

### f. Sorting & Searching

| Syntax                          | Description                            |
|----------------------------------|----------------------------------------|
| `arr.sort(axis)`                | In-place sort                          |
| `arr.argsort(axis)`             | Indices to sort                        |
| `arr.searchsorted(values)`      | Find insertion indices                 |
| `np.unique(arr)`                | Unique values (not a method, but used) |

---

### g. Linear Algebra (for 2D/nd arrays)

| Syntax                        | Description                             |
|-------------------------------|-----------------------------------------|
| `arr.dot(other)`             | Dot product                             |
| `arr.T`                      | Transpose                              |
| `arr.diagonal()`             | Diagonal elements                      |
| `arr.trace()`                | Sum of diagonal elements               |

> For full linear algebra support, use `np.linalg` module.

---

### h. Miscellaneous

| Syntax                    | Description                         |
|---------------------------|-------------------------------------|
| `arr.fill(value)`        | Fill array with scalar              |
| `arr.repeat(repeats)`    | Repeat elements                     |
| `arr.round(decimals)`    | Round to decimal places             |
| `arr.tolist()`           | Convert to (nested) list            |
| `arr.tobytes()`          | Convert to raw bytes                |
| `arr.tofile(file)`       | Write to file (binary)              |
| `arr.dump(filename)`     | Pickle array to file                |
| `arr.dumps()`            | Pickle array to string              |

---

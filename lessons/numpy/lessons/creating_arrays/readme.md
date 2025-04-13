
## **Creating Arrays in NumPy**

NumPy provides multiple ways to create arrays, ranging from manually defined arrays to automatically generated arrays using built-in functions. These arrays can be 1D, 2D, or multi-dimensional.

### **Creating Arrays from Lists and Tuples**

| Syntax                               | Description                                                                                                                                                         |
|--------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)` | Creates an array from a list, tuple, or other sequence-like object. <br>**Parameters:** <br>- `object`: Input data (list, tuple, etc.) <br>- `dtype`: Desired data type (default `None`) <br>- `copy`: Whether to copy the data (default `True`) <br>- `order`: Memory layout order (`'C'`, `'F'`, or `'K'`) <br>- `subok`: If True, return a subclass of ndarray (default `False`) <br>- `ndmin`: Minimum number of dimensions (default `0`) |
| `np.asarray(a, dtype=None, order='K')` | Converts input into an ndarray (avoids copying if already an ndarray). <br>**Parameters:** <br>- `a`: Input data (list, tuple, etc.) <br>- `dtype`: Desired data type (default `None`) <br>- `order`: Memory layout order (`'C'`, `'F'`, or `'K'`) |

---

### **Creating Special Arrays**

| Syntax                                       | Description                                                                                                                                                                               |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.zeros(shape, dtype=float, order='C')`    | Creates an array filled with zeros. <br>**Parameters:** <br>- `shape`: Shape of the array (e.g., `(2, 3)`) <br>- `dtype`: Data type (default `float`) <br>- `order`: Memory layout order (`'C'` for row-major or `'F'` for column-major, default `C`) |
| `np.ones(shape, dtype=float, order='C')`     | Creates an array filled with ones. <br>**Parameters:** <br>- `shape`: Shape of the array (e.g., `(2, 3)`) <br>- `dtype`: Data type (default `float`) <br>- `order`: Memory layout order (`'C'` for row-major or `'F'` for column-major, default `C`) |
| `np.full(shape, fill_value, dtype=None, order='C')` | Creates an array filled with a specified value. <br>**Parameters:** <br>- `shape`: Shape of the array <br>- `fill_value`: Value to fill the array <br>- `dtype`: Data type (default `None`) <br>- `order`: Memory layout order (`'C'` or `'F'`, default `C`) |
| `np.eye(N, M=None, dtype=float, order='C')`   | Creates an identity matrix. <br>**Parameters:** <br>- `N`: Number of rows <br>- `M`: Number of columns (default `None`, which uses `N` for square identity matrix) <br>- `dtype`: Data type (default `float`) <br>- `order`: Memory layout order (`'C'` or `'F'`, default `C`) |
| `np.diag(v, k=0)`                            | Creates a diagonal matrix from a given array. <br>**Parameters:** <br>- `v`: Input array (1D or 2D) <br>- `k`: Diagonal index (default `0`, the main diagonal) |

---

### **Creating Sequences**

| Syntax                                       | Description                                                                                                                                                                               |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.arange([start, ]stop, [step, ], dtype=None)` | Creates values in a range with a specified step. <br>**Parameters:** <br>- `start`: Start of the sequence (default `0`) <br>- `stop`: End of the sequence (exclusive) <br>- `step`: Step size (default `1`) <br>- `dtype`: Desired data type (default `None`) |
| `np.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)` | Creates evenly spaced values between two points. <br>**Parameters:** <br>- `start`: Start value <br>- `stop`: End value <br>- `num`: Number of values to generate (default `50`) <br>- `endpoint`: Whether to include `stop` in the sequence (default `True`) <br>- `retstep`: Whether to return the step size (default `False`) <br>- `dtype`: Desired data type (default `None`) <br>- `axis`: The axis in the result along which the values are stored (default `0`) |
| `np.logspace(start, stop, num=50, endpoint=True, base=10.0, dtype=None, axis=0)` | Creates logarithmically spaced values. <br>**Parameters:** <br>- `start`: Start value <br>- `stop`: End value <br>- `num`: Number of values to generate (default `50`) <br>- `endpoint`: Whether to include `stop` in the sequence (default `True`) <br>- `base`: Base of the logarithm (default `10.0`) <br>- `dtype`: Desired data type (default `None`) <br>- `axis`: The axis in the result along which the values are stored (default `0`) |

---

### **Creating Random Arrays**

| Syntax                                       | Description                                                                                                                                                                               |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.random.rand(d0, d1, ..., dn)`            | Generates random values between 0 and 1. <br>**Parameters:** <br>- `d0, d1, ..., dn`: Dimensions of the output array |
| `np.random.randn(d0, d1, ..., dn)`           | Generates values from a standard normal distribution (mean=0, std=1). <br>**Parameters:** <br>- `d0, d1, ..., dn`: Dimensions of the output array |
| `np.random.randint(low, high=None, size=None, dtype=int)` | Generates random integers in a specified range. <br>**Parameters:** <br>- `low`: Lower bound of the range <br>- `high`: Upper bound of the range (default `None`) <br>- `size`: Output shape (default `None`) <br>- `dtype`: Desired data type (default `int`) |
| `np.random.choice(a, size=None, replace=True, p=None)`  | Randomly selects elements from a given array. <br>**Parameters:** <br>- `a`: Input array or int <br>- `size`: Number of elements to sample (default `None`) <br>- `replace`: Whether to allow sampling of the same element more than once (default `True`) <br>- `p`: Probabilities associated with each element (default `None`) |

---

### **Creating Empty and Uninitialized Arrays**

| Syntax                               | Description                                                                                                                                                     |
|--------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.empty(shape, dtype=float, order='C')` | Creates an uninitialized array with arbitrary values. <br>**Parameters:** <br>- `shape`: Shape of the array <br>- `dtype`: Data type (default `float`) <br>- `order`: Memory layout order (`'C'` for row-major or `'F'` for column-major, default `C`) |
| `np.empty_like(a, dtype=None, order='K', subok=True)` | Creates an uninitialized array with the same shape and type as another. <br>**Parameters:** <br>- `a`: Existing array <br>- `dtype`: Desired data type (default `None`) <br>- `order`: Memory layout order (`'C'` or `'F'`, default `K`) <br>- `subok`: If True, return a subclass of ndarray (default `True`) |

---

### **Creating Arrays with Data Types**

| Syntax                                       | Description                                                                                                                                                       |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.array(object, dtype=np.int32)`           | Converts an array with a specific data type. <br>**Parameters:** <br>- `dtype`: Specifies the desired data type (e.g., `np.int32`, `np.float64`, `np.bool_`) |

---

### **Creating Multi-dimensional Arrays**

| Syntax                                       | Description                                                                                                                                                       |
|----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `np.zeros((3, 3, 3))`                       | Creates a multi-dimensional array (e.g., 3x3x3 array filled with zeros). <br>**Parameters:** <br>- `shape`: Shape of the array (e.g., `(3, 3, 3)`) |

--- 

## **Creating Arrays in NumPy**

NumPy provides multiple ways to create arrays, including from lists, using built-in functions, and generating random values.

---

### **Creating Arrays from Lists and Tuples**  

| Method | Description | Syntax |
|--------|-------------|--------|
| `array()` | Creates an array from a list or tuple. | `arr = np.array([1, 2, 3])` |
| `asarray()` | Converts input into an ndarray (avoids copying if already an ndarray). | `arr = np.asarray([1, 2, 3])` |

---

### **Creating Special Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `zeros()` | Creates an array filled with zeros. | `arr = np.zeros(shape, dtype)` |
| `ones()` | Creates an array filled with ones. | `arr = np.ones(shape, dtype)` |
| `full()` | Creates an array filled with a specified value. | `arr = np.full(shape, value, dtype)` |
| `eye()` | Creates an identity matrix. | `arr = np.eye(N, dtype)` |
| `diag()` | Creates a diagonal matrix from a given array. | `arr = np.diag([values])` |

---

### **Creating Sequences**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `arange()` | Creates a range of values with a step. | `arr = np.arange(start, stop, step, dtype)` |
| `linspace()` | Creates evenly spaced values between two points. | `arr = np.linspace(start, stop, num, dtype)` |
| `logspace()` | Creates logarithmically spaced values. | `arr = np.logspace(start, stop, num, base, dtype)` |

---

### **Creating Random Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `rand()` | Generates random values between 0 and 1. | `arr = np.random.rand(shape)` |
| `randn()` | Generates values from a normal distribution. | `arr = np.random.randn(shape)` |
| `randint()` | Generates random integers in a range. | `arr = np.random.randint(low, high, size, dtype)` |
| `choice()` | Randomly selects elements from a given array. | `arr = np.random.choice(array, size)` |

---

### **Creating Empty and Uninitialized Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `empty()` | Creates an uninitialized array with arbitrary values. | `arr = np.empty(shape, dtype)` |
| `empty_like()` | Creates an empty array with the same shape as another. | `arr = np.empty_like(existing_array, dtype)` |

---

### **Summary**  
- Arrays can be created from lists, using built-in functions, or with random values.  
- `zeros()`, `ones()`, `full()`, and `empty()` create special arrays.  
- `arange()`, `linspace()`, and `logspace()` generate sequences.  
- `rand()`, `randn()`, and `randint()` generate random values.

---
---


## Creating Arrays in NumPy  

### **Overview**  
NumPy provides multiple ways to create arrays, ranging from manually defined arrays to automatically generated arrays using built-in functions. These arrays can be 1D, 2D, or multi-dimensional.

---

### **Creating Arrays from Lists and Tuples**  
NumPy can convert Python lists or tuples into arrays using `np.array()`.  

```python
import numpy as np

arr1 = np.array([1, 2, 3, 4])  # 1D array
arr2 = np.array([[1, 2], [3, 4]])  # 2D array
arr3 = np.array(((1, 2, 3), (4, 5, 6)))  # Tuple to 2D array
```

| Feature | NumPy Array (`ndarray`) | Python List |
|---------|----------------|------------|
| Performance | Faster due to contiguous memory | Slower due to dynamic allocation |
| Functionality | Supports vectorized operations | Requires explicit looping |
| Data Type | Homogeneous (single type) | Can hold mixed types |

---

### **Creating Arrays with Default Values**  
NumPy provides functions to generate arrays filled with specific values.  

| Function | Description | Example |
|----------|-------------|---------|
| `np.zeros(shape)` | Creates an array filled with zeros | `np.zeros((2,3))` |
| `np.ones(shape)` | Creates an array filled with ones | `np.ones((3,3))` |
| `np.full(shape, value)` | Creates an array filled with a specified value | `np.full((2,2), 7)` |
| `np.eye(size)` | Creates an identity matrix | `np.eye(3)` |

Example:  
```python
zeros = np.zeros((2, 3))  # 2x3 matrix filled with 0s
ones = np.ones((3, 3))    # 3x3 matrix filled with 1s
filled = np.full((2, 2), 7)  # 2x2 matrix filled with 7
identity = np.eye(3)  # 3x3 identity matrix
```

---

### **Creating Arrays with Ranges**
NumPy provides functions to generate sequences of numbers efficiently.

| Function | Description | Example |
|----------|-------------|---------|
| `np.arange(start, stop, step)` | Generates values in a range | `np.arange(1, 10, 2)` → `[1,3,5,7,9]` |
| `np.linspace(start, stop, num)` | Generates `num` evenly spaced values between `start` and `stop` | `np.linspace(0, 5, 10)` |

Example:  
```python
arr1 = np.arange(1, 10, 2)  # [1, 3, 5, 7, 9]
arr2 = np.linspace(0, 5, 10)  # 10 evenly spaced values from 0 to 5
```

---

### **Creating Random Arrays**  
NumPy's `random` module provides methods to generate random values.  

| Function | Description | Example |
|----------|-------------|---------|
| `np.random.rand(shape)` | Uniform random values (0 to 1) | `np.random.rand(3,3)` |
| `np.random.randint(low, high, shape)` | Random integers in a range | `np.random.randint(1, 10, (2,3))` |
| `np.random.randn(shape)` | Standard normal distribution (mean=0, std=1) | `np.random.randn(4,4)` |

Example:  
```python
np.random.seed(42)  # Ensures reproducibility
rand_arr = np.random.rand(3, 3)  # 3x3 matrix with values between 0 and 1
rand_ints = np.random.randint(1, 10, (2, 3))  # 2x3 matrix of random integers
```

---

### **Creating Arrays with Data Types**  
NumPy allows specifying the data type using `dtype`.  

| Data Type | Description |
|-----------|-------------|
| `np.int32` | 32-bit integer |
| `np.float64` | 64-bit floating point |
| `np.bool_` | Boolean values |
| `np.complex64` | Complex numbers |

Example:  
```python
arr = np.array([1.5, 2.3, 3.1], dtype=np.int32)  # Converts to integer
print(arr.dtype)  # int32
```

---

### **Creating Multi-dimensional Arrays**  
NumPy supports creating higher-dimensional arrays.  

```python
arr_3d = np.zeros((3, 3, 3))  # 3x3x3 array filled with zeros
```

| Dimension | Example |
|-----------|---------|
| 1D | `np.array([1, 2, 3])` |
| 2D | `np.array([[1,2,3], [4,5,6]])` |
| 3D | `np.zeros((3,3,3))` |

---

### **Conclusion**  
NumPy provides multiple ways to create arrays efficiently, including manual initialization, predefined functions, range-based arrays, random numbers, and multi-dimensional structures. Understanding these methods allows efficient numerical computations in Python.
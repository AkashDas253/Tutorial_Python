## NumPy Basics  

### **Overview**  
NumPy (Numerical Python) is a fundamental library for numerical computing in Python, providing efficient array operations, mathematical functions, and support for multidimensional arrays.  

---

### **Importing NumPy**  
To use NumPy, import it as follows:  
```python
import numpy as np  # Standard alias
```

---

### **NumPy Arrays (`ndarray`)**  
NumPy provides the `ndarray` object, which is more efficient than Python lists for numerical operations.  

#### **Creating an Array**
```python
arr = np.array([1, 2, 3, 4])  # 1D array
mat = np.array([[1, 2], [3, 4]])  # 2D array
```

#### **Key Differences Between Lists and NumPy Arrays**
| Feature      | NumPy Arrays (`ndarray`) | Python Lists |
|-------------|----------------|--------------|
| Performance | Faster due to contiguous memory | Slower due to dynamic memory allocation |
| Functionality | Supports vectorized operations | Requires explicit looping |
| Data Type | Homogeneous (single type) | Can hold mixed types |

---

### **Checking NumPy Version**
To check the installed version of NumPy:  
```python
print(np.__version__)
```

---

### **Array Properties**
| Property  | Description | Example |
|-----------|-------------|---------|
| `.shape` | Returns the dimensions of the array | `arr.shape` |
| `.size` | Total number of elements | `arr.size` |
| `.dtype` | Data type of elements | `arr.dtype` |
| `.ndim` | Number of dimensions | `arr.ndim` |
| `.itemsize` | Memory occupied by each element (in bytes) | `arr.itemsize` |

Example:  
```python
arr = np.array([[1, 2, 3], [4, 5, 6]])
print(arr.shape)  # (2,3)
print(arr.size)   # 6
print(arr.dtype)  # int32 (depends on system)
print(arr.ndim)   # 2
print(arr.itemsize)  # 4 (bytes, may vary)
```

---

### **Creating Arrays**  
#### **From Lists and Tuples**
```python
arr = np.array([1, 2, 3, 4])
mat = np.array([(1.5, 2.3, 3.1), (4.6, 5.7, 6.8)])  # 2D array
```

#### **Using Built-in Functions**
| Function | Description | Example |
|----------|-------------|---------|
| `np.zeros()` | Creates an array of zeros | `np.zeros((2,3))` |
| `np.ones()` | Creates an array of ones | `np.ones((3,3))` |
| `np.full()` | Creates an array filled with a specific value | `np.full((2,2), 7)` |
| `np.eye()` | Identity matrix | `np.eye(3)` |
| `np.arange()` | Array with a range of values | `np.arange(1, 10, 2)` |
| `np.linspace()` | Evenly spaced values in a range | `np.linspace(0, 5, 10)` |

Example:  
```python
zeros = np.zeros((2, 3))  # 2x3 matrix filled with 0s
ones = np.ones((3, 3))    # 3x3 matrix filled with 1s
seq = np.arange(1, 10, 2) # [1, 3, 5, 7, 9]
lin = np.linspace(0, 5, 10)  # 10 evenly spaced values from 0 to 5
```

---

### **Random Arrays**
NumPy provides a `random` module for generating random numbers.  
```python
np.random.seed(42)  # Ensures reproducibility
rand_arr = np.random.rand(3, 3)  # 3x3 random values between 0 and 1
rand_ints = np.random.randint(1, 10, (2, 3))  # Random integers between 1 and 9
```

---

### **Data Types in NumPy**
NumPy arrays require a consistent data type. You can specify the type using `dtype`.  

| Data Type | Description |
|-----------|-------------|
| `np.int32` | 32-bit integer |
| `np.float64` | 64-bit floating point |
| `np.complex64` | Complex numbers |
| `np.bool_` | Boolean values |

Example:  
```python
arr = np.array([1.5, 2.3, 3.1], dtype=np.int32)  # Converts to integer
print(arr.dtype)  # int32
```

---

### **Basic Operations on Arrays**
NumPy allows vectorized operations, making computations more efficient.  

#### **Arithmetic Operations**
| Operation | Example |
|-----------|---------|
| Addition | `arr + 2` |
| Subtraction | `arr - 1` |
| Multiplication | `arr * 2` |
| Division | `arr / 2` |
| Power | `arr ** 2` |

Example:  
```python
arr = np.array([1, 2, 3])
print(arr + 2)  # [3, 4, 5]
print(arr * 3)  # [3, 6, 9]
```

#### **Statistical Functions**
| Function | Description |
|----------|-------------|
| `np.mean()` | Mean of the array |
| `np.median()` | Median of the array |
| `np.std()` | Standard deviation |
| `np.var()` | Variance |
| `np.min()` | Minimum value |
| `np.max()` | Maximum value |

Example:  
```python
arr = np.array([1, 2, 3, 4, 5])
print(np.mean(arr))  # 3.0
print(np.max(arr))   # 5
print(np.min(arr))   # 1
```

---

### **Reshaping Arrays**
Reshape an array without changing its data.  
```python
arr = np.arange(6).reshape(2, 3)  # Reshape into 2x3 matrix
print(arr)
```

Output:
```
[[0 1 2]
 [3 4 5]]
```

---

### **Indexing and Slicing**
Access elements like lists but optimized for multi-dimensional arrays.  

#### **Basic Indexing**
```python
arr = np.array([10, 20, 30, 40])
print(arr[1])  # 20
```

#### **Slicing**
```python
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[1:4])  # [1, 2, 3]
print(arr[:3])   # [0, 1, 2]
print(arr[::-1]) # Reverse array
```

#### **2D Indexing**
```python
mat = np.array([[1, 2, 3], [4, 5, 6]])
print(mat[1, 2])  # Access row 1, column 2 (6)
```

---

### **Conclusion**
NumPy is an essential library for numerical computing, offering fast array operations, built-in mathematical functions, and efficient memory usage. Understanding the basics of NumPy arrays, indexing, reshaping, and mathematical operations is crucial for data science and machine learning applications.
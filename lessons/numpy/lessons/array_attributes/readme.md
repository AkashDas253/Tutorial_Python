## **Array Attributes in NumPy**

NumPy provides various attributes to access metadata about arrays, such as shape, size, dimensions, and memory usage.

---

### **Basic Array Attributes**

| Attribute | Description | Syntax |
|-----------|-------------|---------|
| `shape` | Returns a tuple representing the array's dimensions. | `arr.shape` |
| `ndim` | Returns the number of dimensions (axes) of the array. | `arr.ndim` |
| `size` | Returns the total number of elements in the array. | `arr.size` |
| `dtype` | Returns the data type of the array elements. | `arr.dtype` |
| `itemsize` | Returns the memory size (in bytes) of each element. | `arr.itemsize` |
| `nbytes` | Returns the total memory occupied by the array in bytes. | `arr.nbytes` |

---

### **Shape and Reshaping Arrays**

| Operation | Description | Syntax |
|-----------|-------------|---------|
| `reshape()` | Reshapes an array without changing data. | `new_array = arr.reshape(new_shape)` |

---

### **Flattening Arrays**

| Operation | Description | Syntax |
|-----------|-------------|---------|
| `ravel()` | Returns a flattened 1D array. | `flat_array = arr.ravel()` |
| `flatten()` | Returns a flattened copy of the array. | `flat_array = arr.flatten()` |

---

### **Data Type and Type Conversion**

| Operation | Description | Syntax |
|-----------|-------------|---------|
| `dtype` | Returns the data type of elements. | `arr.dtype` |
| `astype()` | Converts array elements to a new data type. | `new_array = arr.astype(new_dtype)` |

---

### **Memory Usage Optimization**

| Operation | Description | Syntax |
|-----------|-------------|---------|
| `itemsize` | Returns the memory size (in bytes) of each element. | `arr.itemsize` |
| `nbytes` | Returns total memory occupied by the array. | `arr.nbytes` |

---

### **Summary**
- NumPy array attributes provide information about the array’s structure and memory usage.
- `shape`, `size`, and `ndim` describe the structure.
- `dtype`, `itemsize`, and `nbytes` describe memory usage.
- Arrays can be reshaped, flattened, and converted between types for optimization.


---
---


## **Array Attributes in NumPy**

NumPy provides various attributes to access metadata about arrays, such as shape, size, dimensions, and memory usage.

---

### **Basic Array Attributes**

| Attribute | Description | Syntax |
|-----------|-------------|---------|
| `shape` | Returns a tuple representing the array's dimensions. | `arr.shape` |
| `ndim` | Returns the number of dimensions (axes) of the array. | `arr.ndim` |
| `size` | Returns the total number of elements in the array. | `arr.size` |
| `dtype` | Returns the data type of the array elements. | `arr.dtype` |
| `itemsize` | Returns the memory size (in bytes) of each element. | `arr.itemsize` |
| `nbytes` | Returns the total memory occupied by the array in bytes. | `arr.nbytes` |

---

### **Example Usage**

```python
import numpy as np

# Creating a 2D array
arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float64)

# Displaying attributes
print("Array:\n", arr)
print("Shape:", arr.shape)      # (2, 3)
print("Dimensions:", arr.ndim)  # 2
print("Total Elements:", arr.size)  # 6
print("Data Type:", arr.dtype)  # float64
print("Item Size:", arr.itemsize)  # 8 bytes (float64)
print("Total Bytes:", arr.nbytes)  # 6 elements * 8 bytes = 48
```

---

### **Shape and Reshaping Arrays**

- The `shape` attribute provides the dimensions of an array.
- The `reshape()` method modifies the shape without changing data.

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshaping to a 2x3 matrix
reshaped = arr.reshape(2, 3)

print(reshaped.shape)  # (2, 3)
print(reshaped)
```

**Output:**
```
(2, 3)
[[1 2 3]
 [4 5 6]]
```

---

Yes, the syntax for reshaping is included. Here it is explicitly:  

### **Syntax for Reshaping**  
```python
new_array = array.reshape(new_shape)
```
- `array`: The original NumPy array.  
- `new_shape`: A tuple specifying the new dimensions (must match the original number of elements).  

### **Example Usage**  
```python
arr = np.array([1, 2, 3, 4, 5, 6])  

# Reshape to (2,3)  
reshaped = arr.reshape(2, 3)  

print(reshaped.shape)  # (2, 3)  
print(reshaped)  
```

---
### **Flattening Arrays**

- Convert a multi-dimensional array into a 1D array using `.ravel()` or `.flatten()`.

```python
arr = np.array([[1, 2, 3], [4, 5, 6]])

flat_arr = arr.ravel()  # Returns [1 2 3 4 5 6]
print(flat_arr)

flat_arr2 = arr.flatten()
print(flat_arr2)
```

---

### **Data Type and Type Conversion**

- The `dtype` attribute specifies the data type of elements in the array.
- The `astype()` method converts an array to another data type.

```python
arr = np.array([1.1, 2.2, 3.3])

# Convert to integer type
int_arr = arr.astype(np.int32)

print(int_arr)  # [1 2 3]
print(int_arr.dtype)  # int32
```

---

### **Memory Usage Optimization**

- Use `astype()` to reduce memory consumption.
- Convert `float64` to `float32` or `int64` to `int32` where applicable.

```python
arr = np.array([1.5, 2.5, 3.5], dtype=np.float64)

# Convert to float32 to reduce memory usage
arr_float32 = arr.astype(np.float32)

print(arr_float32.dtype)  # float32
print("Memory usage before:", arr.nbytes, "bytes")
print("Memory usage after:", arr_float32.nbytes, "bytes")
```

---

### **Summary**
- NumPy array attributes provide information about the array’s structure and memory usage.
- `shape`, `size`, and `ndim` describe the structure.
- `dtype`, `itemsize`, and `nbytes` describe memory usage.
- Arrays can be reshaped, flattened, and converted between types for optimization.
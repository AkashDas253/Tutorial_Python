## **Advanced NumPy Topics**  

---

### **1. Vectorization and Performance Optimization**  
- Uses NumPy's built-in operations for faster computations.  
- Avoids explicit loops with element-wise operations.  

**Syntax:**  
```python
arr = np.array([1, 2, 3])
result = arr * 2  # Vectorized multiplication
```

---

### **2. Broadcasting**  
- Enables operations between arrays of different shapes without explicit loops.  

**Syntax:**  
```python
arr1 = np.array([[1], [2], [3]])
arr2 = np.array([10, 20, 30])
result = arr1 + arr2  # Broadcasting addition
```

---

### **3. Memory Layout and Strides**  
- Controls how data is stored in memory for efficiency.  
- `C_CONTIGUOUS` (row-major) vs. `F_CONTIGUOUS` (column-major).  

**Syntax:**  
```python
arr = np.array([[1, 2, 3], [4, 5, 6]], order='F')
print(arr.flags)  # Shows memory layout details
```

---

### **4. Views vs. Copies**  
- Views share memory, copies allocate new memory.  

**Syntax:**  
```python
arr = np.array([1, 2, 3])
view_arr = arr.view()  # Shares memory
copy_arr = arr.copy()  # Creates new memory
```

---

### **5. Fancy Indexing**  
- Uses arrays as indices for complex selection.  

**Syntax:**  
```python
arr = np.array([10, 20, 30, 40])
indices = np.array([0, 2])
result = arr[indices]  # Selects elements at indices 0 and 2
```

---

### **6. Universal Functions (ufuncs)**  
- Efficient element-wise operations with broadcasting support.  

**Syntax:**  
```python
arr = np.array([1, 4, 9])
result = np.sqrt(arr)  # Applies sqrt to each element
```

---

### **7. Advanced Linear Algebra**  
- Matrix operations like inverse, eigenvalues, and decomposition.  

**Syntax:**  
```python
arr = np.array([[2, 3], [4, 5]])
inverse = np.linalg.inv(arr)  # Computes inverse
```

---

### **8. Random Sampling and Distributions**  
- Generates random numbers from various distributions.  

**Syntax:**  
```python
rand_vals = np.random.normal(loc=0, scale=1, size=5)  # Normal distribution
```

---

### **9. Structured and Record Arrays**  
- Stores heterogeneous data types in a structured format.  

**Syntax:**  
```python
dtype = [('name', 'U10'), ('age', 'i4')]
arr = np.array([('Alice', 25), ('Bob', 30)], dtype=dtype)
```

---

### **10. Masked Arrays**  
- Hides invalid or missing values.  

**Syntax:**  
```python
arr = np.array([1, 2, -999, 4])
masked_arr = np.ma.masked_equal(arr, -999)  # Masks -999
```

---

### **11. NumPy with C and Cython**  
- Improves performance using C extensions.  

**Syntax (Cython example):**  
```cython
cimport numpy as np
def add_arrays(np.ndarray arr1, np.ndarray arr2):
    return arr1 + arr2
```

---

### **12. Parallel Processing with NumPy**  
- Uses multi-threading for better performance.  

**Syntax:**  
```python
from joblib import Parallel, delayed
results = Parallel(n_jobs=4)(delayed(np.sqrt)(i) for i in range(10))
```

---

### **13. Sparse Matrices (SciPy Integration)**  
- Efficiently stores large matrices with mostly zero values.  

**Syntax:**  
```python
from scipy.sparse import csr_matrix
arr = np.array([[0, 0, 3], [4, 0, 0]])
sparse_matrix = csr_matrix(arr)
```

---

### **14. Handling Large Datasets with `memmap`**  
- Allows working with large arrays without loading them entirely into memory.  

**Syntax:**  
```python
arr = np.memmap('data.npy', dtype='float32', mode='w+', shape=(1000, 1000))
```

---

### **15. Multi-dimensional Indexing and Iteration**  
- Iterates efficiently over high-dimensional arrays.  

**Syntax:**  
```python
arr = np.array([[1, 2], [3, 4]])
for index, value in np.ndenumerate(arr):
    print(index, value)
```

---

### **16. NumPy in Machine Learning and AI**  
- Used in data preprocessing, matrix operations, and deep learning frameworks.  

**Syntax:**  
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
arr_scaled = scaler.fit_transform(arr)
```

---

### **17. Profiling and Debugging NumPy Code**  
- Measures execution time to identify performance bottlenecks.  

**Syntax:**  
```python
import timeit
timeit.timeit('np.dot(a, b)', setup='import numpy as np; a=np.ones((100,100)); b=np.ones((100,100))', number=100)
```

---

### **Summary**  
- **Optimization techniques** like vectorization, broadcasting, and parallel processing improve performance.  
- **Advanced indexing, memory management, and data structures** allow efficient handling of large datasets.  
- **NumPy integrates with SciPy, Cython, and machine learning libraries** for high-performance computing.
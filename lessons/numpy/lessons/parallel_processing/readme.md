## **Parallel Processing with NumPy**  

---

### **Overview**  
- NumPy utilizes **vectorized operations** for fast computations.  
- Parallel execution is achieved using **multi-threading and multi-processing**.  
- Supports **SIMD (Single Instruction Multiple Data)** for element-wise operations.  

---

### **1. NumPy’s Built-in Parallelization**  
NumPy internally uses **multi-threading** for optimized performance in functions like:  
- `dot()`, `matmul()`, `linalg.inv()`, `linalg.eig()`, etc.  
- Controlled by **environment variables** like `OMP_NUM_THREADS`.  

**Syntax:**  
```python
import os
os.environ["OMP_NUM_THREADS"] = "4"  # Set number of threads  
```

---

### **2. Parallel Execution with `numexpr`**  
- `numexpr` optimizes NumPy expressions using **multi-threading**.  
- Reduces **memory overhead** and improves efficiency.  

**Syntax:**  
```python
import numexpr as ne  

a = np.random.rand(1000000)  
b = np.random.rand(1000000)  

result = ne.evaluate("a * b + 2")  # Faster than regular NumPy operations  
```

---

### **3. Using `multiprocessing` for Parallel Computation**  
- NumPy operations can be parallelized using **multiple CPU cores**.  
- `multiprocessing` creates **separate processes** for tasks.  

**Syntax:**  
```python
import multiprocessing as mp  

def square_array(arr):  
    return np.square(arr)  

arr = np.random.rand(1000000)  
pool = mp.Pool(processes=4)  
result = pool.map(square_array, np.array_split(arr, 4))  
pool.close()  
pool.join()  
```
- **`np.array_split(arr, 4)` divides data** into 4 chunks for processing.  

---

### **4. Parallelizing Loops with `joblib`**  
- `joblib.Parallel` enables efficient **multi-threaded computations**.  

**Syntax:**  
```python
from joblib import Parallel, delayed  

def compute(x):  
    return x * x  

arr = np.arange(1000000)  
result = Parallel(n_jobs=4)(delayed(compute)(x) for x in arr)  
```
- **`n_jobs=4` uses 4 cores** to process elements.  

---

### **5. GPU Acceleration with `cupy`**  
- `cupy` provides **GPU-accelerated** NumPy computations.  

**Syntax:**  
```python
import cupy as cp  

a = cp.array([1, 2, 3])  
b = cp.array([4, 5, 6])  
result = cp.dot(a, b)  # Faster execution on GPU  
```

---

### **Summary**  
| **Method** | **Description** |
|------------|---------------|
| **NumPy’s Built-in Parallelization** | Uses OpenMP for automatic parallel execution |
| **`numexpr`** | Optimizes expressions with multi-threading |
| **`multiprocessing`** | Parallelizes NumPy operations using multiple processes |
| **`joblib`** | Simplifies parallel loops |
| **`cupy`** | Enables GPU acceleration |

- **Use built-in NumPy optimizations when possible**.  
- **For CPU parallelism**, prefer `joblib` or `multiprocessing`.  
- **For large-scale computing**, consider GPU-based `cupy`.
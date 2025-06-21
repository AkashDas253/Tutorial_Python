## **NumPy with Cython and C**  

---

### **Overview**  
- **Cython** allows writing **C-optimized** Python code for faster execution.  
- NumPy arrays can be accessed efficiently using **typed memory views** in Cython.  
- **C integration** enables direct calling of C functions from NumPy for performance gains.  

---

### **1. Using Cython to Speed Up NumPy Operations**  
- Cython compiles Python-like syntax into **C extensions** for faster execution.  
- Supports **static typing** and **typed memory views** for NumPy arrays.  

**Installation:**  
```sh
pip install cython
```

**Cython File (`fast_operations.pyx`):**  
```cython
import numpy as np  
cimport numpy as cnp  

def sum_array(cnp.ndarray[cnp.float64_t, ndim=1] arr):  
    cdef double total = 0  
    cdef int i  
    for i in range(arr.shape[0]):  
        total += arr[i]  
    return total  
```

**Compiling Cython Code (`setup.py`):**  
```python
from setuptools import setup  
from Cython.Build import cythonize  
import numpy  

setup(  
    ext_modules=cythonize("fast_operations.pyx"),  
    include_dirs=[numpy.get_include()]  
)
```

**Compiling & Running:**  
```sh
python setup.py build_ext --inplace
```

---

### **2. Using Cython’s `nogil` for Multi-threading**  
- Cython allows **releasing Python’s GIL (Global Interpreter Lock)** for parallel execution.  

**Optimized Cython Code (`parallel_sum.pyx`):**  
```cython
from cython.parallel import prange  
cimport numpy as cnp  

def parallel_sum(cnp.ndarray[cnp.float64_t, ndim=1] arr):  
    cdef double total = 0  
    cdef int i  
    with nogil:  
        for i in prange(arr.shape[0], nogil=True):  
            total += arr[i]  
    return total  
```

---

### **3. Calling C Functions from NumPy**  
- NumPy can integrate with C code using **Python’s `ctypes` or `cffi`**.  
- Writing performance-critical functions in C improves speed.  

**C Code (`math_functions.c`):**  
```c
#include <stdio.h>

double square(double x) {  
    return x * x;  
}
```

**Compiling the C File:**  
```sh
gcc -shared -o math_functions.so -fPIC math_functions.c  
```

**Using C in Python with `ctypes`:**  
```python
import ctypes  

lib = ctypes.CDLL("./math_functions.so")  
lib.square.argtypes = [ctypes.c_double]  
lib.square.restype = ctypes.c_double  

result = lib.square(5.0)  
print(result)  # Output: 25.0  
```

---

### **4. Interfacing NumPy with Cython and C**  
- NumPy arrays can be **passed to C functions** for faster processing.  

**C Code (`array_ops.c`):**  
```c
#include <stdio.h>

void square_array(double *arr, int size) {  
    for (int i = 0; i < size; i++) {  
        arr[i] *= arr[i];  
    }  
}
```

**Cython Wrapper (`array_ops.pyx`):**  
```cython
cdef extern void square_array(double *arr, int size)  

def square_numpy_array(double[:] arr):  
    square_array(&arr[0], arr.shape[0])  
```

**Compiling and Using the Cython Wrapper:**  
```sh
python setup.py build_ext --inplace  
```

**Using in Python:**  
```python
import numpy as np  
from array_ops import square_numpy_array  

arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)  
square_numpy_array(arr)  
print(arr)  # Output: [1.0, 4.0, 9.0]  
```

---

### **Summary**  
| **Method** | **Description** |
|------------|---------------|
| **Cython** | Optimizes NumPy operations using C-like syntax |
| **Cython `nogil`** | Enables parallel processing with multi-threading |
| **`ctypes` with C** | Calls C functions directly from Python |
| **Cython + C** | Passes NumPy arrays to C for efficient processing |

- **Use Cython for in-place optimization of NumPy operations.**  
- **Use C integration (`ctypes`, `cffi`) for performance-critical functions.**  
- **For multi-threading, release GIL (`nogil`) in Cython.**
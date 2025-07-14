## **Profiling and Debugging NumPy Code**  

---

### **1. Profiling NumPy Code**  
- Profiling helps **identify performance bottlenecks** in NumPy operations.  
- Tools like `timeit`, `cProfile`, and `line_profiler` measure execution time and function calls.  

#### **1.1 Using `timeit` for Performance Measurement**  
- Measures the execution time of small NumPy operations.  

**Syntax:**  
```python
import numpy as np  
import timeit  

arr = np.random.rand(1000000)  

# Measure execution time
execution_time = timeit.timeit("np.sum(arr)", globals=globals(), number=10)  
print(f"Execution Time: {execution_time:.5f} seconds")  
```

---

#### **1.2 Using `cProfile` for Function-Level Profiling**  
- Profiles function execution, showing call count and execution time.  

**Syntax:**  
```python
import numpy as np  
import cProfile  

def compute():  
    arr = np.random.rand(1000000)  
    return np.sum(arr)  

cProfile.run("compute()")  
```

---

#### **1.3 Using `line_profiler` for Line-by-Line Profiling**  
- Provides detailed time spent on each line in a function.  

**Installation:**  
```sh
pip install line-profiler
```

**Syntax:**  
```python
import numpy as np  
from line_profiler import LineProfiler  

def compute():  
    arr = np.random.rand(1000000)  
    return np.sum(arr)  

lp = LineProfiler()  
lp.add_function(compute)  
lp.enable()  
compute()  
lp.disable()  
lp.print_stats()  
```

---

### **2. Debugging NumPy Code**  
- Debugging finds errors like **incorrect indexing, shape mismatches, and NaN values**.  
- Tools: `pdb`, `assert`, `np.seterr`, and `np.isnan`.  

#### **2.1 Using Python’s Built-in Debugger (`pdb`)**  
- Step through NumPy operations interactively.  

**Syntax:**  
```python
import numpy as np  
import pdb  

def debug_function():  
    arr = np.random.rand(5, 5)  
    pdb.set_trace()  # Debugger starts here  
    result = arr[2, 10]  # Index error  
    return result  

debug_function()  
```

---

#### **2.2 Handling Floating-Point Errors with `np.seterr`**  
- Controls NumPy’s error-handling behavior for division, overflow, etc.  

**Syntax:**  
```python
import numpy as np  

np.seterr(divide='raise', invalid='warn')  

arr = np.array([1.0, 0.0, -1.0])  
try:  
    result = 1 / arr  # Raises ZeroDivisionError  
except FloatingPointError as e:  
    print(f"Error: {e}")  
```

---

#### **2.3 Detecting NaN and Inf Values in Arrays**  
- Prevents unexpected numerical issues in computations.  

**Syntax:**  
```python
import numpy as np  

arr = np.array([1, np.nan, np.inf, -np.inf])  

# Check for NaN values  
print(np.isnan(arr))  # Output: [False  True False False]  

# Check for Inf values  
print(np.isinf(arr))  # Output: [False False  True  True]  
```

---

### **3. Summary**  
| **Method** | **Usage** |
|------------|-----------|
| **`timeit`** | Measures execution time of code snippets |
| **`cProfile`** | Profiles function-level performance |
| **`line_profiler`** | Provides line-by-line execution time |
| **`pdb`** | Debugs NumPy operations interactively |
| **`np.seterr`** | Handles floating-point errors |
| **`np.isnan`, `np.isinf`** | Detects NaN and Inf values |

- **Use `timeit` for quick performance testing.**  
- **Use `cProfile` for analyzing function execution.**  
- **Use `line_profiler` to optimize slow code lines.**  
- **Use `pdb` for debugging NumPy issues.**  
- **Use `np.seterr` to control floating-point errors.**
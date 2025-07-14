## **Universal Functions (ufuncs)**  

---

### **Overview**  
- NumPy's **universal functions (ufuncs)** perform element-wise operations on arrays.  
- They are optimized in C for speed and support **broadcasting**, **type casting**, and **reductions**.  
- Examples include mathematical, comparison, logical, and bitwise operations.  

---

### **1. Creating and Using ufuncs**  
- ufuncs automatically apply operations to each element in an array.  

**Syntax:**  
```python
import numpy as np
result = np.add([1, 2, 3], [4, 5, 6])  # Element-wise addition
```

---

### **2. Types of ufuncs**  

| **Category**     | **Examples** | **Description** |
|-----------------|-------------|----------------|
| **Arithmetic**  | `add`, `subtract`, `multiply`, `divide`, `mod`, `power` | Performs element-wise arithmetic operations. |
| **Trigonometric** | `sin`, `cos`, `tan`, `arcsin`, `arccos` | Computes trigonometric values. |
| **Exponential & Logarithmic** | `exp`, `log`, `log10`, `log2` | Computes exponentiation and logarithms. |
| **Comparison** | `greater`, `less`, `equal`, `not_equal` | Element-wise comparison of arrays. |
| **Logical** | `logical_and`, `logical_or`, `logical_xor`, `logical_not` | Performs element-wise logical operations. |
| **Bitwise** | `bitwise_and`, `bitwise_or`, `bitwise_xor`, `invert` | Applies bitwise operations. |

---

### **3. Broadcasting in ufuncs**  
- Allows operations on arrays of different shapes.  

**Syntax:**  
```python
arr = np.array([1, 2, 3])
result = arr + 10  # Broadcasts 10 to each element
```

---

### **4. Aggregation (Reduce, Accumulate, Reduceat)**  

#### **4.1 Reduce**  
- Applies a ufunc cumulatively over an array.  

**Syntax:**  
```python
arr = np.array([1, 2, 3, 4])
result = np.add.reduce(arr)  # Sum of all elements
```

#### **4.2 Accumulate**  
- Applies a ufunc step by step.  

**Syntax:**  
```python
arr = np.array([1, 2, 3, 4])
result = np.add.accumulate(arr)  # Running sum
```

#### **4.3 Reduceat**  
- Applies a ufunc on specific slices of an array.  

**Syntax:**  
```python
arr = np.array([1, 2, 3, 4, 5])
result = np.add.reduceat(arr, [0, 2, 4])  # Partial sums
```

---

### **5. Custom ufuncs with `frompyfunc`**  
- Allows defining new ufuncs from Python functions.  

**Syntax:**  
```python
def square(x):
    return x ** 2

ufunc_square = np.frompyfunc(square, 1, 1)  # 1 input, 1 output
result = ufunc_square(np.array([1, 2, 3]))
```

---

### **Summary**  
- **ufuncs provide optimized element-wise operations** for performance.  
- **Supports broadcasting and aggregation functions** like `reduce`, `accumulate`, and `reduceat`.  
- **Allows creation of custom ufuncs** using `frompyfunc`.
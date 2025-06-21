## **Mathematical Operations in NumPy**

NumPy provides a variety of mathematical functions to perform element-wise operations, aggregation, and linear algebra computations.

---

### **Element-wise Arithmetic Operations**  

| Operation | Description | Syntax |
|-----------|-------------|--------|
| Addition | Adds two arrays element-wise. | `result = arr1 + arr2` |
| Subtraction | Subtracts one array from another. | `result = arr1 - arr2` |
| Multiplication | Multiplies arrays element-wise. | `result = arr1 * arr2` |
| Division | Divides arrays element-wise. | `result = arr1 / arr2` |
| Floor Division | Performs element-wise floor division. | `result = arr1 // arr2` |
| Modulus | Computes element-wise remainder. | `result = arr1 % arr2` |
| Exponentiation | Raises elements to a power. | `result = arr1 ** arr2` |

---

### **Universal Functions (ufuncs)**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `add()` | Element-wise addition. | `result = np.add(arr1, arr2)` |
| `subtract()` | Element-wise subtraction. | `result = np.subtract(arr1, arr2)` |
| `multiply()` | Element-wise multiplication. | `result = np.multiply(arr1, arr2)` |
| `divide()` | Element-wise division. | `result = np.divide(arr1, arr2)` |
| `power()` | Raises each element to a power. | `result = np.power(arr, exponent)` |
| `mod()` | Element-wise remainder. | `result = np.mod(arr1, arr2)` |

---

### **Trigonometric Functions**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `sin()` | Computes sine of each element. | `result = np.sin(arr)` |
| `cos()` | Computes cosine of each element. | `result = np.cos(arr)` |
| `tan()` | Computes tangent of each element. | `result = np.tan(arr)` |
| `arcsin()` | Computes inverse sine. | `result = np.arcsin(arr)` |
| `arccos()` | Computes inverse cosine. | `result = np.arccos(arr)` |
| `arctan()` | Computes inverse tangent. | `result = np.arctan(arr)` |

---

### **Exponential and Logarithmic Functions**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `exp()` | Computes the exponential of each element. | `result = np.exp(arr)` |
| `log()` | Computes the natural logarithm. | `result = np.log(arr)` |
| `log10()` | Computes base-10 logarithm. | `result = np.log10(arr)` |
| `log2()` | Computes base-2 logarithm. | `result = np.log2(arr)` |

---

### **Rounding and Approximation**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `round()` | Rounds elements to the nearest integer. | `result = np.round(arr, decimals)` |
| `floor()` | Rounds down to the nearest integer. | `result = np.floor(arr)` |
| `ceil()` | Rounds up to the nearest integer. | `result = np.ceil(arr)` |

---

### **Aggregation Functions**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `sum()` | Computes the sum of elements. | `total = np.sum(arr, axis)` |
| `mean()` | Computes the average. | `avg = np.mean(arr, axis)` |
| `median()` | Computes the median. | `med = np.median(arr)` |
| `std()` | Computes the standard deviation. | `std_dev = np.std(arr)` |
| `var()` | Computes the variance. | `variance = np.var(arr)` |
| `min()` | Finds the minimum value. | `min_value = np.min(arr)` |
| `max()` | Finds the maximum value. | `max_value = np.max(arr)` |

---

### **Linear Algebra Operations**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `dot()` | Computes the dot product of two arrays. | `result = np.dot(arr1, arr2)` |
| `matmul()` | Computes matrix multiplication. | `result = np.matmul(arr1, arr2)` |
| `linalg.det()` | Computes the determinant of a matrix. | `det_value = np.linalg.det(arr)` |
| `linalg.inv()` | Computes the inverse of a matrix. | `inv_matrix = np.linalg.inv(arr)` |
| `linalg.eig()` | Computes the eigenvalues and eigenvectors. | `eig_vals, eig_vecs = np.linalg.eig(arr)` |

---

### **Summary**  
- **Element-wise operations**: `+`, `-`, `*`, `/`, `//`, `%`, `**`.  
- **Trigonometric functions**: `sin()`, `cos()`, `tan()`, `arcsin()`, `arccos()`, `arctan()`.  
- **Exponential & logarithmic**: `exp()`, `log()`, `log10()`, `log2()`.  
- **Rounding**: `round()`, `floor()`, `ceil()`.  
- **Aggregation**: `sum()`, `mean()`, `median()`, `std()`, `var()`, `min()`, `max()`.  
- **Linear algebra**: `dot()`, `matmul()`, `det()`, `inv()`, `eig()`.
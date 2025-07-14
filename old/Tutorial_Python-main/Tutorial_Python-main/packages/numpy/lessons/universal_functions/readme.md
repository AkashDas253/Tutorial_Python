
# **NumPy Universal Functions (ufuncs)**

---

## **Definition**

Universal functions (ufuncs) are **element-wise functions** provided by NumPy that operate on `ndarray` objects efficiently in C-speed, supporting **broadcasting**, **type casting**, and **multiple input/output**.

---

## **Key Features**

- Element-wise operation on arrays  
- Supports broadcasting  
- Optional output array using `out` parameter  
- Optional condition using `where` parameter  
- Most support `dtype`, `casting`, `order`, and `subok`

---

## **Categories of ufuncs**

---

### **1. Arithmetic Functions**

| Function        | Description                   |
|------------------|-------------------------------|
| `np.add(x, y)`   | Element-wise addition         |
| `np.subtract(x,y)`| Element-wise subtraction     |
| `np.multiply(x,y)`| Element-wise multiplication  |
| `np.divide(x, y)` | Element-wise division        |
| `np.floor_divide(x, y)` | Integer division      |
| `np.mod(x, y)`    | Modulus (remainder)          |
| `np.remainder(x,y)`| Remainder                   |
| `np.power(x, y)`  | Power `x ** y`               |
| `np.exp(x)`       | Exponential `e^x`            |
| `np.exp2(x)`      | `2^x`                        |
| `np.log(x)`       | Natural logarithm            |
| `np.log2(x)`      | Base-2 log                   |
| `np.log10(x)`     | Base-10 log                  |

---

### **2. Trigonometric Functions**

| Function           | Description                  |
|--------------------|------------------------------|
| `np.sin(x)`        | Sine                         |
| `np.cos(x)`        | Cosine                       |
| `np.tan(x)`        | Tangent                      |
| `np.arcsin(x)`     | Inverse sine                 |
| `np.arccos(x)`     | Inverse cosine               |
| `np.arctan(x)`     | Inverse tangent              |
| `np.deg2rad(x)`    | Degrees to radians           |
| `np.rad2deg(x)`    | Radians to degrees           |
| `np.hypot(x, y)`   | Euclidean norm               |

---

### **3. Hyperbolic Functions**

| Function           | Description                |
|--------------------|----------------------------|
| `np.sinh(x)`       | Hyperbolic sine            |
| `np.cosh(x)`       | Hyperbolic cosine          |
| `np.tanh(x)`       | Hyperbolic tangent         |
| `np.arcsinh(x)`    | Inverse hyperbolic sine    |
| `np.arccosh(x)`    | Inverse hyperbolic cosine  |
| `np.arctanh(x)`    | Inverse hyperbolic tangent |

---

### **4. Comparison Functions**

| Function           | Description                         |
|--------------------|-------------------------------------|
| `np.greater(x, y)` | Element-wise `x > y`                |
| `np.greater_equal(x, y)` | `x >= y`                     |
| `np.less(x, y)`    | `x < y`                             |
| `np.less_equal(x, y)` | `x <= y`                        |
| `np.equal(x, y)`   | `x == y`                            |
| `np.not_equal(x, y)`| `x != y`                           |
| `np.maximum(x, y)` | Element-wise max                    |
| `np.minimum(x, y)` | Element-wise min                    |
| `np.clip(x, min, max)`| Restrict values to range         |

---

### **5. Logical Functions**

| Function             | Description                     |
|----------------------|---------------------------------|
| `np.logical_and(x,y)`| Logical AND                     |
| `np.logical_or(x,y)` | Logical OR                      |
| `np.logical_xor(x,y)`| Logical XOR                     |
| `np.logical_not(x)`  | Logical NOT                     |

---

### **6. Bitwise Functions**

| Function             | Description                     |
|----------------------|---------------------------------|
| `np.bitwise_and(x,y)`| Bitwise AND                    |
| `np.bitwise_or(x,y)` | Bitwise OR                     |
| `np.bitwise_xor(x,y)`| Bitwise XOR                    |
| `np.invert(x)`       | Bitwise NOT                    |
| `np.left_shift(x,n)` | Shift left by `n` bits         |
| `np.right_shift(x,n)`| Shift right by `n` bits        |

---

### **7. Rounding Functions**

| Function            | Description                     |
|---------------------|---------------------------------|
| `np.round(x)`       | Round to nearest integer        |
| `np.floor(x)`       | Round down                     |
| `np.ceil(x)`        | Round up                       |
| `np.trunc(x)`       | Truncate decimal               |
| `np.fix(x)`         | Round toward zero              |

---

### **8. Sign & Special Number Functions**

| Function          | Description                         |
|-------------------|-------------------------------------|
| `np.sign(x)`      | Sign of x: -1, 0, or +1             |
| `np.abs(x)`       | Absolute value                     |
| `np.fabs(x)`      | Absolute (float version)           |
| `np.signbit(x)`   | True for negative numbers          |
| `np.copysign(x,y)`| Copy sign from `y` to `x`          |
| `np.nextafter(x,y)`| Next representable value          |
| `np.spacing(x)`   | Distance to next representable float|

---

### **9. Special Floating-Point Checks**

| Function            | Description                   |
|---------------------|-------------------------------|
| `np.isnan(x)`        | Check NaN                    |
| `np.isinf(x)`        | Check Inf                    |
| `np.isfinite(x)`     | Check finite (not NaN/Inf)   |

---

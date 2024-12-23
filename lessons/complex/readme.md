### **Complex Numbers in Python**

---

### **Introduction**
Python provides a built-in type for representing complex numbers, which consist of a real and an imaginary part, expressed as `a + bj`, where:
- `a` is the real part.
- `b` is the imaginary part.

---

### **Properties of Complex Numbers**

| **Property**       | **Description**                                                     |
|---------------------|---------------------------------------------------------------------|
| **Immutable**       | Complex numbers cannot be changed after creation.                  |
| **Built-in Type**   | Python has a native complex type.                                   |

---

### **Creating Complex Numbers**

| **Method**              | **Code**                        | **Description**                                        |
|--------------------------|---------------------------------|-------------------------------------------------------|
| **Using `complex` Function** | `z = complex(2, 3)`             | Creates the complex number `2 + 3j`.                  |
| **Using Direct Assignment** | `z = 2 + 3j`                   | Assigns the complex number `2 + 3j` directly.         |

---

### **Accessing Real and Imaginary Parts**

| **Part**               | **Code**               | **Description**                           |
|-------------------------|------------------------|-------------------------------------------|
| **Real Part**           | `real_part = z.real`  | Accesses the real part of the complex number. |
| **Imaginary Part**      | `imaginary_part = z.imag` | Accesses the imaginary part of the complex number. |

---

### **Basic Operations**

| **Operation**         | **Code**                  | **Description**                                    |
|------------------------|---------------------------|---------------------------------------------------|
| **Addition**           | `result = z1 + z2`       | Adds two complex numbers.                        |
| **Subtraction**        | `result = z1 - z2`       | Subtracts one complex number from another.       |
| **Multiplication**     | `result = z1 * z2`       | Multiplies two complex numbers.                  |
| **Division**           | `result = z1 / z2`       | Divides one complex number by another.           |
| **Conjugate**          | `conjugate = z1.conjugate()` | Returns the conjugate of the complex number. |

---

### **Advanced Operations**

| **Operation**           | **Code**                   | **Description**                                       |
|--------------------------|----------------------------|------------------------------------------------------|
| **Absolute Value**       | `magnitude = abs(z1)`      | Returns the magnitude (absolute value) of the complex number. |
| **Phase Angle**          | `phase = cmath.phase(z1)`  | Returns the phase angle (in radians).               |
| **Polar Coordinates**    | `polar = cmath.polar(z1)`  | Converts the complex number to polar coordinates `(r, theta)`. |
| **Rectangular Coordinates** | `rectangular = cmath.rect(r, theta)` | Converts polar coordinates to rectangular form. |

---

### **Complex Number Methods**

| **Method**            | **Code**                        | **Description**                                     |
|------------------------|---------------------------------|---------------------------------------------------|
| **`conjugate()`**      | `conjugate = z1.conjugate()`   | Returns the conjugate of the complex number.      |
| **`__add__` Method**   | `result = z1.__add__(z2)`      | Adds two complex numbers.                         |
| **`__sub__` Method**   | `result = z1.__sub__(z2)`      | Subtracts one complex number from another.        |
| **`__mul__` Method**   | `result = z1.__mul__(z2)`      | Multiplies two complex numbers.                   |
| **`__truediv__` Method** | `result = z1.__truediv__(z2)` | Divides one complex number by another.            |
| **`__abs__` Method**   | `magnitude = z1.__abs__()`     | Returns the magnitude of the complex number.      |

---

### **Functions**

| **Function**            | **Description**                                              | **Example**                | **Result**                |
|--------------------------|--------------------------------------------------------------|----------------------------|---------------------------|
| `complex(real, imag)`    | Creates a complex number from a real and imaginary part.     | `z = complex(2, 3)`        | `2 + 3j`                 |
| `abs(x)`                 | Returns the magnitude (absolute value) of a complex number. | `abs(3 + 4j)`              | `5.0`                    |
| `cmath.phase(x)`         | Returns the phase angle of the complex number.              | `cmath.phase(1 + 1j)`      | `0.7854` (radians)        |
| `cmath.polar(x)`         | Converts the complex number to polar coordinates.           | `cmath.polar(1 + 1j)`      | `(1.414, 0.7854)`         |
| `cmath.rect(r, theta)`   | Converts polar coordinates back to rectangular form.        | `cmath.rect(1.414, 0.7854)` | `1 + 1j`                 |

---

### **Example Usage**

```python
import cmath

# Creating complex numbers
z1 = 3 + 4j
z2 = 1 - 2j

# Basic operations
addition = z1 + z2
subtraction = z1 - z2
multiplication = z1 * z2
division = z1 / z2

# Accessing parts
real_part = z1.real
imaginary_part = z1.imag

# Advanced operations
conjugate = z1.conjugate()
magnitude = abs(z1)
phase = cmath.phase(z1)
polar = cmath.polar(z1)
rectangular = cmath.rect(polar[0], polar[1])
```

---

### **Summary**

- **Properties**: Immutable, built-in type.
- **Creation**: Using `complex()` or direct assignment.
- **Access**: `real` and `imag` parts.
- **Operations**: Addition, subtraction, multiplication, division, conjugate.
- **Advanced Functions**: Magnitude, phase, polar and rectangular coordinates.
- **Methods**: `conjugate`, `__add__`, `__sub__`, etc.
- **Library Support**: Use the `cmath` library for advanced operations.

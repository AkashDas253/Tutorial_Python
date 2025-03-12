## **Complex (`complex`) in Python**  

### **Definition**  
- Represents numbers in the form **a + bj**, where:  
  - `a` is the real part (float or int).  
  - `b` is the imaginary part (float or int).  
  - `j` represents the imaginary unit (√-1).  

### **Declaring a Complex Number**  
```python
z1 = 3 + 4j  # 3 is the real part, 4 is the imaginary part
z2 = -2.5 + 1.5j  # Negative real, positive imaginary
z3 = 5j  # Only imaginary part
```

---

### **Complex Number Properties**
| Property | Description | Example | Result |
|----------|------------|---------|--------|
| `z.real` | Returns real part | `(3 + 4j).real` | `3.0` |
| `z.imag` | Returns imaginary part | `(3 + 4j).imag` | `4.0` |
| `abs(z)` | Magnitude (modulus) | `abs(3 + 4j)` | `5.0` |
| `z.conjugate()` | Returns conjugate | `(3 + 4j).conjugate()` | `3 - 4j` |

```python
z = 3 + 4j
print(z.real)  # 3.0
print(z.imag)  # 4.0
print(abs(z))  # 5.0 (since sqrt(3² + 4²) = 5)
print(z.conjugate())  # 3 - 4j
```

---

### **Complex Number Operations**  
| Operation | Example | Result |
|-----------|---------|--------|
| Addition | `(2 + 3j) + (1 - 4j)` | `3 - 1j` |
| Subtraction | `(5 + 2j) - (3 + j)` | `2 + 1j` |
| Multiplication | `(2 + 3j) * (1 + 4j)` | `-10 + 11j` |
| Division | `(4 + 2j) / (1 - j)` | `1 + 3j` |
| Exponentiation | `(1 + 2j) ** 2` | `-3 + 4j` |

```python
z1 = 2 + 3j
z2 = 1 - 4j

print(z1 + z2)  # 3 - 1j
print(z1 - z2)  # 1 + 7j
print(z1 * z2)  # -10 + 11j
print(z1 / z2)  # -0.588 + 0.824j
```

---

### **Complex Number Functions (cmath Module)**
| Function | Description | Example | Result |
|----------|------------|---------|--------|
| `cmath.sqrt(z)` | Square root | `cmath.sqrt(-1)` | `1j` |
| `cmath.exp(z)` | Exponential | `cmath.exp(1j * cmath.pi)` | `-1+0j` |
| `cmath.log(z)` | Natural log | `cmath.log(1 + 1j)` | `(0.346 + 0.785j)` |
| `cmath.sin(z)` | Sine | `cmath.sin(1 + 1j)` | `(1.298 + 0.635j)` |
| `cmath.cos(z)` | Cosine | `cmath.cos(1 + 1j)` | `(0.833 - 0.987j)` |

```python
import cmath

z = 1 + 1j
print(cmath.sqrt(-1))  # 1j
print(cmath.exp(1j * cmath.pi))  # -1+0j
print(cmath.log(z))  # (0.346 + 0.785j)
print(cmath.sin(z))  # (1.298 + 0.635j)
```

---

### **Converting Complex Numbers**
| Function | Converts To | Example | Result |
|----------|------------|---------|--------|
| `complex(a, b)` | Complex | `complex(2, -3)` | `2 - 3j` |
| `str(z)` | String | `str(1 + 2j)` | `"1 + 2j"` |
| `z.real` | Float (real part) | `(3 + 4j).real` | `3.0` |
| `z.imag` | Float (imaginary part) | `(3 + 4j).imag` | `4.0` |

```python
z = complex(2, -3)
print(z)  # 2 - 3j
print(str(z))  # "2 - 3j"
print(z.real)  # 2.0
print(z.imag)  # -3.0
```

---
---

## **Complex Numbers in Python**

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

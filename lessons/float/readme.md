## **Float (`float`) in Python**  

### **Definition**  
- Represents real numbers with decimal points or scientific notation.  
- Stored as double-precision floating-point numbers (64-bit).  

### **Declaring a Float**  
```python
x = 3.14
y = -0.5
z = 2.0  # Even though it's a whole number, it's still a float
```

---

### **Float Operations**  
| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `+` | Addition | `3.5 + 2.5` | `6.0` |
| `-` | Subtraction | `5.5 - 2.0` | `3.5` |
| `*` | Multiplication | `2.0 * 3.0` | `6.0` |
| `/` | Division | `5.0 / 2.0` | `2.5` |
| `//` | Floor division | `5.0 // 2.0` | `2.0` |
| `%` | Modulus | `5.5 % 2.0` | `1.5` |
| `**` | Exponentiation | `2.0 ** 3` | `8.0` |

---

### **Type Checking & Conversion**
| Function | Converts To | Example | Result |
|----------|------------|---------|--------|
| `float(x)` | Float | `float(5)` | `5.0` |
| `int(x)` | Integer (truncates decimal) | `int(3.9)` | `3` |
| `str(x)` | String | `str(2.5)` | `"2.5"` |
| `bool(x)` | Boolean | `bool(0.0)` | `False` |

```python
x = 10.75
print(type(x))  # <class 'float'>

y = int(x)  # Converts to integer (truncation)
print(y)  # 10
```

---

### **Scientific Notation**
- `e` or `E` represents powers of 10.
```python
x = 1.5e3  # 1.5 × 10³ = 1500.0
y = 2.1E-2  # 2.1 × 10⁻² = 0.021
```

---

### **Precision Issues**
- Floating-point numbers have rounding errors due to binary representation.
```python
print(0.1 + 0.2)  # 0.30000000000000004
```
- Use `round()` to control precision:
```python
print(round(0.1 + 0.2, 2))  # 0.3
```

---

### **Math Functions for Floats**
| Function | Description | Example | Result |
|----------|------------|---------|--------|
| `round(x, n)` | Rounds to `n` decimal places | `round(3.14159, 2)` | `3.14` |
| `abs(x)` | Absolute value | `abs(-2.5)` | `2.5` |
| `pow(x, y)` | Power | `pow(2.0, 3)` | `8.0` |
| `math.floor(x)` | Rounds down | `math.floor(3.9)` | `3` |
| `math.ceil(x)` | Rounds up | `math.ceil(3.1)` | `4` |
| `math.sqrt(x)` | Square root | `math.sqrt(9.0)` | `3.0` |

```python
import math

print(math.floor(3.9))  # 3
print(math.ceil(3.1))  # 4
print(math.sqrt(16.0))  # 4.0
```

---

### **Comparing Floats (Avoid Direct Comparison)**
- Use `math.isclose()` for accurate comparisons:
```python
import math

print(math.isclose(0.1 + 0.2, 0.3))  # True
```

---

### **Infinity & NaN (Not a Number)**
```python
x = float('inf')  # Positive infinity
y = float('-inf')  # Negative infinity
z = float('nan')  # Not a Number

print(x > 1000)  # True
print(math.isnan(z))  # True
```

---
---

## **Float in Python**

---

### **Introduction**
In Python, floating-point numbers (`float`) represent real numbers, i.e., numbers with a fractional part. They follow the IEEE 754 double-precision standard.

---

### **Properties of Floats**

| **Property**          | **Description**                                                     |
|------------------------|---------------------------------------------------------------------|
| **Immutable**          | Floats cannot be changed after creation.                           |
| **Precision**          | Limited by the IEEE 754 standard.                                  |
| **Type**               | The type of a float is `float`.                                    |
| **Range**              | Floats have a range of approximately `-1.8e308` to `1.8e308`.      |

---

### **Creating Floats**

| **Method**            | **Code**                      | **Description**                                     |
|------------------------|-------------------------------|----------------------------------------------------|
| **Direct Assignment**  | `my_float = 3.14`            | Assigns a floating-point value directly to a variable. |
| **Using `float()`**    | `my_float = float("3.14")`   | Converts a string or integer to a float.          |

---

### **Basic Operations**

| **Operation**         | **Symbol** | **Example**        | **Result**        |
|------------------------|------------|--------------------|-------------------|
| **Addition**           | `+`        | `3.5 + 2.5`        | `6.0`            |
| **Subtraction**        | `-`        | `3.5 - 2.5`        | `1.0`            |
| **Multiplication**     | `*`        | `3.5 * 2`          | `7.0`            |
| **Division**           | `/`        | `7.0 / 2`          | `3.5`            |
| **Floor Division**     | `//`       | `7.0 // 2`         | `3.0`            |
| **Modulus**            | `%`        | `7.0 % 2`          | `1.0`            |
| **Exponentiation**     | `**`       | `2.0 ** 3`         | `8.0`            |
| **Negation**           | `-`        | `-3.5`             | `-3.5`           |

---

### **Type Conversion**

| **Conversion**        | **Code**                   | **Description**                                |
|------------------------|----------------------------|-----------------------------------------------|
| **To Integer**         | `int(my_float)`           | Converts a float to an integer (truncates the fractional part). |
| **To String**          | `str(my_float)`           | Converts a float to a string.                 |
| **To Boolean**         | `bool(my_float)`          | Converts to `False` if `0.0`, otherwise `True`. |

---

### **Built-in Functions**

| **Function**          | **Description**                                                    | **Example**                   | **Result**        |
|------------------------|--------------------------------------------------------------------|--------------------------------|-------------------|
| `abs(x)`              | Returns the absolute value of the float.                          | `abs(-3.14)`                  | `3.14`           |
| `pow(x, y, z=None)`   | Returns `x` raised to the power of `y`.                           | `pow(2.0, 3)`                 | `8.0`            |
| `round(x, ndigits)`   | Rounds a float to the specified number of decimal places.         | `round(3.14159, 2)`           | `3.14`           |
| `max(iterable)`       | Returns the largest float in an iterable.                         | `max([1.5, 2.3, 3.1])`        | `3.1`            |
| `min(iterable)`       | Returns the smallest float in an iterable.                        | `min([1.5, 2.3, 3.1])`        | `1.5`            |
| `sum(iterable)`       | Returns the sum of all items in an iterable.                      | `sum([1.1, 2.2, 3.3])`        | `6.6`            |

---

### **Float Methods**

| **Method**                 | **Description**                                                    | **Example**                                      | **Result**               |
|----------------------------|--------------------------------------------------------------------|------------------------------------------------|--------------------------|
| `is_integer()`             | Checks if the float is an integer.                                | `(3.0).is_integer()`                           | `True`                   |
| `as_integer_ratio()`        | Returns a pair of integers whose ratio is equal to the float.    | `(3.14).as_integer_ratio()`                    | `(157, 50)`              |
| `hex()`                    | Returns a hexadecimal string representation of the float.        | `(3.14).hex()`                                 | `'0x1.91eb851eb851fp+1'` |
| `float.fromhex(hex_str)`    | Creates a float from a hexadecimal string.                       | `float.fromhex('0x1.91eb851eb851fp+1')`        | `3.14`                   |

---

### **Conversions with Bytes**

| **Method**                   | **Description**                                                 | **Example**                                                   | **Result**        |
|-------------------------------|-----------------------------------------------------------------|---------------------------------------------------------------|-------------------|
| `num.to_bytes(length, byteorder)` | Converts an integer to bytes.                                 | `num = 1024; num.to_bytes(2, 'big')`                          | `b'\x04\x00'`     |
| `int.from_bytes(bytes, byteorder, signed)` | Converts bytes to an integer.                              | `int.from_bytes(b'\x04\x00', 'big')`                          | `1024`            |

---

### **Float Formatting**

| **Method**             | **Code**                              | **Result**         |
|-------------------------|---------------------------------------|--------------------|
| **`format` Function**   | `format(3.14159, ".2f")`             | `'3.14'`          |
| **f-Strings**           | `f"{3.14159:.2f}"`                   | `'3.14'`          |
| **`str.format` Method** | `"The value is {:.2f}".format(3.14)` | `'The value is 3.14'` |

---

### **Special Float Values**

| **Special Value**      | **Code**                  | **Description**           |
|-------------------------|---------------------------|---------------------------|
| **Positive Infinity**   | `pos_inf = float('inf')`  | Represents positive infinity. |
| **Negative Infinity**   | `neg_inf = float('-inf')` | Represents negative infinity. |
| **NaN (Not a Number)**  | `nan_value = float('nan')`| Represents an undefined or unrepresentable value. |

---

### **Checking for Special Float Values**

| **Check**              | **Code**                                 | **Result**       |
|-------------------------|------------------------------------------|------------------|
| **Infinity**            | `math.isinf(float('inf'))`              | `True`           |
| **NaN**                 | `math.isnan(float('nan'))`              | `True`           |

---

### **Summary**

- **Properties**: Precision, immutability.
- **Basic Operations**: Arithmetic and comparisons.
- **Type Conversions**: Float ↔ Integer/String/Bytes.
- **Built-in Functions**: `abs()`, `pow()`, etc.
- **Float Methods**: Including `is_integer()` and `as_integer_ratio()`.
- **Formatting**: f-Strings, `format()`, etc.
- **Special Float Values**: Infinity, NaN.
- **Checks for Special Values**: Using `math.isinf()` and `math.isnan()`.

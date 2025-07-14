## Integer (`int`) in Python  

### Definition  
- Represents whole numbers without a fractional part.  
- Can be positive, negative, or zero.  
- No size limit (depends on available memory).  

### Declaring an Integer  
```python
x = 10
y = -50
z = 0
```

---

### **Integer Operations**  
| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `+` | Addition | `10 + 5` | `15` |
| `-` | Subtraction | `10 - 5` | `5` |
| `*` | Multiplication | `10 * 5` | `50` |
| `**` | Exponentiation | `2 ** 3` | `8` |
| `/` | Division (float result) | `10 / 3` | `3.333` |
| `//` | Floor division | `10 // 3` | `3` |
| `%` | Modulus (remainder) | `10 % 3` | `1` |

---

### **Integer Type Conversion**
| Function | Converts To | Example | Result |
|----------|------------|---------|--------|
| `int(x)` | Integer | `int(3.9)` | `3` |
| `float(x)` | Float | `float(5)` | `5.0` |
| `str(x)` | String | `str(100)` | `"100"` |
| `bool(x)` | Boolean | `bool(0)` | `False` |
| `bin(x)` | Binary | `bin(10)` | `'0b1010'` |
| `oct(x)` | Octal | `oct(10)` | `'0o12'` |
| `hex(x)` | Hexadecimal | `hex(10)` | `'0xa'` |

---

### **Integer Comparisons**  
| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `==` | Equals | `10 == 10` | `True` |
| `!=` | Not equal | `10 != 5` | `True` |
| `>` | Greater than | `10 > 5` | `True` |
| `<` | Less than | `10 < 5` | `False` |
| `>=` | Greater than or equal | `10 >= 10` | `True` |
| `<=` | Less than or equal | `10 <= 5` | `False` |

---

### **Bitwise Operations on Integers**  
| Operator | Description | Example | Result |
|----------|------------|---------|--------|
| `&` | AND | `5 & 3` | `1` |
| `|` | OR | `5 | 3` | `7` |
| `^` | XOR | `5 ^ 3` | `6` |
| `~` | NOT | `~5` | `-6` |
| `<<` | Left shift | `5 << 1` | `10` |
| `>>` | Right shift | `5 >> 1` | `2` |

- **Example Calculation**  
  ```python
  a = 5  # 0b0101
  b = 3  # 0b0011
  print(a & b)  # 0b0001 → 1
  print(a | b)  # 0b0111 → 7
  print(a ^ b)  # 0b0110 → 6
  ```

---

### **Checking Integer Type**  
```python
x = 100
print(type(x))  # <class 'int'>
```

---

### **Special Integer Cases**  
- **Boolean as Integer**  
  ```python
  True + 1  # 2 (True is 1)
  False + 1  # 1 (False is 0)
  ```
- **Division Always Returns Float**  
  ```python
  print(type(10 / 2))  # <class 'float'>
  ```
- **Using `isinstance` to Check Integer**  
  ```python
  isinstance(10, int)  # True
  isinstance(10.5, int)  # False
  ```

---

### **Large Integer Handling**
- No fixed limit on integer size.
- Python automatically extends the size.
  ```python
  large_num = 10**100  # Very large number
  print(large_num)
  ```

---

### **Integer Formatting**
- Display numbers in different bases:
  ```python
  print(f"Binary: {10:b}")  # Binary: 1010
  print(f"Octal: {10:o}")   # Octal: 12
  print(f"Hex: {10:x}")     # Hex: a
  ```

---
---

## **Integer in Python**

### **Introduction**
In Python, integers (`int`) represent whole numbers, positive, negative, or zero, without fractional parts. Python provides extensive functionality for integer operations, methods, and type conversions.

---

### **Properties of Integers**

| **Property**            | **Description**                                                                                  |
|--------------------------|--------------------------------------------------------------------------------------------------|
| **Immutable**            | Integers cannot be changed after creation.                                                      |
| **Unlimited Precision**  | Can grow arbitrarily large, limited only by system memory.                                       |
| **Type**                 | The type of an integer is `int`.                                                                |

---

### **Creating Integers**

| **Method**                | **Code**                                | **Description**                                    |
|----------------------------|------------------------------------------|---------------------------------------------------|
| **Direct Assignment**      | `my_int = 42`                          | Assigns an integer value directly to a variable. |
| **Using `int()`**          | `my_int = int("42")`                   | Converts a string or float to an integer.        |
| **Hexadecimal to Integer** | `int("0x2A", 16)`                      | Converts a hexadecimal string to an integer.     |
| **Binary to Integer**      | `int("1010", 2)`                       | Converts a binary string to an integer.          |
| **Octal to Integer**       | `int("52", 8)`                         | Converts an octal string to an integer.          |

---

### **Basic Operations**

| **Operation**         | **Symbol** | **Example**       | **Result**        |
|------------------------|------------|-------------------|-------------------|
| **Addition**           | `+`        | `10 + 5`          | `15`             |
| **Subtraction**        | `-`        | `10 - 5`          | `5`              |
| **Multiplication**     | `*`        | `10 * 5`          | `50`             |
| **Division**           | `/`        | `10 / 3`          | `3.333...`       |
| **Floor Division**     | `//`       | `10 // 3`         | `3`              |
| **Modulus**            | `%`        | `10 % 3`          | `1`              |
| **Exponentiation**     | `**`       | `2 ** 3`          | `8`              |
| **Negation**           | `-`        | `-5`              | `-5`             |

---

### **Comparison Operations**

| **Operation**            | **Symbol** | **Example**         | **Result**         |
|---------------------------|------------|---------------------|--------------------|
| **Equal To**              | `==`       | `10 == 5`           | `False`           |
| **Not Equal To**          | `!=`       | `10 != 5`           | `True`            |
| **Greater Than**          | `>`        | `10 > 5`            | `True`            |
| **Less Than**             | `<`        | `10 < 5`            | `False`           |
| **Greater Than or Equal** | `>=`       | `10 >= 5`           | `True`            |
| **Less Than or Equal**    | `<=`       | `10 <= 5`           | `False`           |

---

### **Bitwise Operations**

| **Operation**     | **Symbol** | **Example** | **Result** |
|--------------------|------------|-------------|------------|
| **AND**           | `&`        | `10 & 7`    | `2`        |
| **OR**            | `|`        | `10 | 7`    | `15`       |
| **XOR**           | `^`        | `10 ^ 7`    | `13`       |
| **NOT**           | `~`        | `~10`       | `-11`      |
| **Left Shift**    | `<<`       | `10 << 2`   | `40`       |
| **Right Shift**   | `>>`       | `10 >> 2`   | `2`        |

---

### **Type Conversion**

| **Conversion**     | **Code**                        | **Description**                                    |
|---------------------|----------------------------------|---------------------------------------------------|
| **To Float**        | `float(my_int)`                | Converts an integer to a float.                  |
| **To String**       | `str(my_int)`                  | Converts an integer to a string.                 |
| **To Boolean**      | `bool(my_int)`                 | Converts to `False` if `0`, otherwise `True`.    |
| **To Binary String**| `bin(my_int)`                  | Returns the binary representation of an integer. |
| **To Octal String** | `oct(my_int)`                  | Returns the octal representation of an integer.  |
| **To Hexadecimal**  | `hex(my_int)`                  | Returns the hexadecimal representation.          |

---

### **Built-in Functions**

| **Function** | **Description**                             | **Example**                   | **Result**       |
|--------------|---------------------------------------------|--------------------------------|------------------|
| `abs(x)`     | Returns the absolute value.                | `abs(-10)`                    | `10`            |
| `pow(x, y)`  | Returns `x**y`.                            | `pow(2, 3)`                   | `8`             |
| `divmod(x, y)`| Returns `(quotient, remainder)`.          | `divmod(10, 3)`               | `(3, 1)`        |
| `round(x)`   | Rounds to the nearest integer or precision.| `round(10.5)`                 | `10`            |
| `max(iterable)`| Returns the largest item.                | `max([1, 5, 3])`              | `5`             |
| `min(iterable)`| Returns the smallest item.               | `min([1, 5, 3])`              | `1`             |
| `sum(iterable)`| Returns the sum of all items.            | `sum([1, 2, 3])`              | `6`             |

---

### **Additional Methods for Integers**

| **Method**                  | **Description**                                                                 | **Example**                         | **Result**      |
|-----------------------------|---------------------------------------------------------------------------------|-------------------------------------|-----------------|
| `int.bit_length()`          | Returns the number of bits needed to represent the integer in binary.           | `(10).bit_length()`                 | `4`             |
| `int.to_bytes(length, byteorder, signed)`| Converts to a byte array.                                           | `(10).to_bytes(2, 'big', False)`    | `b'\x00\n'`     |
| `int.from_bytes(bytes, byteorder, signed)`| Converts from byte array to integer.                                | `int.from_bytes(b'\x00\n', 'big')`  | `10`            |
| `int.is_integer()`          | Checks if a number is an integer.                                               | `float(3.0).is_integer()`           | `True`          |

---

### **Advanced Functions for Integers**

| **Function**       | **Description**                                  | **Example**                           | **Result**      |
|---------------------|-------------------------------------------------|---------------------------------------|-----------------|
| `math.gcd(a, b)`    | Returns the greatest common divisor of `a` and `b`. | `math.gcd(15, 5)`                     | `5`             |
| `math.factorial(x)` | Returns the factorial of an integer.            | `math.factorial(5)`                   | `120`           |
| `math.isqrt(x)`     | Returns the integer square root.                | `math.isqrt(16)`                      | `4`             |
| `math.prod(iterable)`| Returns the product of all elements in an iterable.| `math.prod([1, 2, 3, 4])`            | `24`            |

---

### **Summary**

- **Properties**: Immutable, unlimited precision, and type (`int`).
- **Creation**: Direct assignment, `int()`, and conversions from other bases.
- **Operations**: Arithmetic, comparison, and bitwise operations.
- **Type Conversion**: Conversion to and from other data types.
- **Built-in Functions**: Including `abs()`, `pow()`, `round()`, `max()`, and more.
- **Methods**: Integer-specific methods such as `.bit_length()` and `.to_bytes()`.
- **Advanced Functions**: Includes mathematical operations like `math.gcd()` and `math.factorial()`.

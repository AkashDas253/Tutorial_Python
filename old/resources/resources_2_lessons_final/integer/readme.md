
## Interger(int)

### In-Depth Note on Python Integers

#### Properties of Integers

- **Immutable**: Integers cannot be changed after creation.
- **Unlimited Precision**: Integers in Python can be arbitrarily large.
- **Type**: The type of an integer is `int`.

#### Creating Integers

1. **Direct Assignment**
   ```python
   my_int = 42
   ```

2. **Using `int` Function**
   ```python
   my_int = int("42")  # Converts a string to an integer
   ```

#### Basic Operations

1. **Addition**
   ```python
   result = a + b
   ```

2. **Subtraction**
   ```python
   result = a - b
   ```

3. **Multiplication**
   ```python
   result = a * b
   ```

4. **Division** (returns a float)
   ```python
   result = a / b
   ```

5. **Floor Division** (returns an integer)
   ```python
   result = a // b
   ```

6. **Modulus** (remainder of division)
   ```python
   result = a % b
   ```

7. **Exponentiation**
   ```python
   result = a ** b
   ```

#### Comparison Operations

1. **Equal To**
   ```python
   result = a == b
   ```

2. **Not Equal To**
   ```python
   result = a != b
   ```

3. **Greater Than**
   ```python
   result = a > b
   ```

4. **Less Than**
   ```python
   result = a < b
   ```

5. **Greater Than or Equal To**
   ```python
   result = a >= b
   ```

6. **Less Than or Equal To**
   ```python
   result = a <= b
   ```

#### Bitwise Operations

1. **AND**
   ```python
   result = a & b
   ```

2. **OR**
   ```python
   result = a | b
   ```

3. **XOR**
   ```python
   result = a ^ b
   ```

4. **NOT**
   ```python
   result = ~a
   ```

5. **Left Shift**
   ```python
   result = a << b
   ```

6. **Right Shift**
   ```python
   result = a >> b
   ```

#### Type Conversion

1. **To Float**
   ```python
   my_float = float(my_int)
   ```

2. **To String**
   ```python
   my_str = str(my_int)
   ```

3. **To Boolean**
   ```python
   my_bool = bool(my_int)  # False if my_int is 0, True otherwise
   ```

#### Built-in Functions

1. **`abs`**: Returns the absolute value
   ```python
   result = abs(my_int)
   ```

2. **`divmod`**: Returns a tuple of quotient and remainder
   ```python
   quotient, remainder = divmod(a, b)
   ```

3. **`pow`**: Returns the value of a raised to the power of b
   ```python
   result = pow(a, b)
   ```

4. **`round`**: Rounds a number to a given precision
   ```python
   result = round(my_int, ndigits)
   ```

5. **`max`**: Returns the largest of the input values
   ```python
   result = max(a, b, c)
   ```

6. **`min`**: Returns the smallest of the input values
   ```python
   result = min(a, b, c)
   ```

### Methods and Functions for Python Integers

#### Methods Called Using Integer Objects

1. **`int.bit_length()`**
   - **Description**: Returns the number of bits necessary to represent the integer in binary, excluding the sign and leading zeros.
   - **Usage**:
     ```python
     num_bits = my_int.bit_length()
     ```

2. **`int.to_bytes(length, byteorder, *, signed=False)`**
   - **Description**: Returns an array of bytes representing the integer.
     - `length`: The number of bytes to use.
     - `byteorder`: The byte order used to represent the integer. Can be `'big'` or `'little'`.
     - `signed` (optional): Whether two's complement is used to represent the integer. Default is `False`.
   - **Usage**:
     ```python
     byte_array = my_int.to_bytes(length=4, byteorder='big', signed=False)
     ```

3. **`int.from_bytes(bytes, byteorder, *, signed=False)`**
   - **Description**: Returns the integer represented by the given array of bytes.
     - `bytes`: The byte array to convert.
     - `byteorder`: The byte order used to represent the integer. Can be `'big'` or `'little'`.
     - `signed` (optional): Whether two's complement is used to represent the integer. Default is `False`.
   - **Usage**:
     ```python
     my_int = int.from_bytes(byte_array, byteorder='big', signed=False)
     ```

#### Functions for Integers

1. **`abs(x)`**
   - **Description**: Returns the absolute value of the integer.
     - `x`: The integer whose absolute value is to be returned.
   - **Usage**:
     ```python
     result = abs(my_int)
     ```

2. **`divmod(a, b)`**
   - **Description**: Returns a tuple containing the quotient and remainder when dividing `a` by `b`.
     - `a`: The dividend.
     - `b`: The divisor.
   - **Usage**:
     ```python
     quotient, remainder = divmod(a, b)
     ```

3. **`pow(base, exp, mod=None)`**
   - **Description**: Returns the value of `base` raised to the power of `exp`. If `mod` is provided, returns `base**exp % mod`.
     - `base`: The base number.
     - `exp`: The exponent.
     - `mod` (optional): The modulus.
   - **Usage**:
     ```python
     result = pow(base, exp, mod)
     ```

4. **`round(number, ndigits=None)`**
   - **Description**: Rounds a number to a given precision in decimal digits.
     - `number`: The number to be rounded.
     - `ndigits` (optional): The number of decimal digits to round to. If omitted, rounds to the nearest integer.
   - **Usage**:
     ```python
     result = round(my_int, ndigits)
     ```

5. **`max(iterable, *[, default, key])`**
   - **Description**: Returns the largest item in an iterable or the largest of two or more arguments.
     - `iterable`: An iterable of items.
     - `default` (optional): The default value to return if the iterable is empty.
     - `key` (optional): A function to customize the sort order.
   - **Usage**:
     ```python
     result = max([a, b, c], default=0, key=None)
     ```

6. **`min(iterable, *[, default, key])`**
   - **Description**: Returns the smallest item in an iterable or the smallest of two or more arguments.
     - `iterable`: An iterable of items.
     - `default` (optional): The default value to return if the iterable is empty.
     - `key` (optional): A function to customize the sort order.
   - **Usage**:
     ```python
     result = min([a, b, c], default=0, key=None)
     ```





## Detailed Note on Integer in Python

### Introduction
In Python, integers are a fundamental data type used to represent whole numbers. They can be positive, negative, or zero, and they do not have a fractional part.

### Creating Integers
You can create an integer by simply assigning a whole number to a variable.

```python
a = 10
b = -5
c = 0
```

### Type Checking
You can check the type of a variable using the `type()` function.

```python
a = 10
print(type(a))  # Output: <class 'int'>
```

### Integer Operations
Python supports various arithmetic operations with integers.

| Operation     | Symbol | Example          | Result |
|---------------|--------|------------------|--------|
| Addition      | `+`    | `10 + 5`         | `15`   |
| Subtraction   | `-`    | `10 - 5`         | `5`    |
| Multiplication| `*`    | `10 * 5`         | `50`   |
| Division      | `/`    | `10 / 5`         | `2.0`  |
| Floor Division| `//`   | `10 // 3`        | `3`    |
| Modulus       | `%`    | `10 % 3`         | `1`    |
| Exponentiation| `**`   | `2 ** 3`         | `8`    |

### Built-in Functions
Python provides several built-in functions for working with integers.

| Function      | Description                                      | Example                | Result |
|---------------|--------------------------------------------------|------------------------|--------|
| `abs(x)`      | Returns the absolute value of `x`                | `abs(-10)`             | `10`   |
| `pow(x, y)`   | Returns `x` raised to the power `y`              | `pow(2, 3)`            | `8`    |
| `divmod(x, y)`| Returns a tuple of quotient and remainder        | `divmod(10, 3)`        | `(3, 1)`|
| `int(x)`      | Converts `x` to an integer                       | `int(3.14)`            | `3`    |

### Type Conversion
You can convert other data types to integers using the `int()` function.

```python
a = int(3.14)  # Converts float to int, result is 3
b = int("10")  # Converts string to int, result is 10
```

### Integer Limits
Python integers have arbitrary precision, meaning they can grow as large as the memory allows.

```python
import sys
print(sys.maxsize)  # Maximum value for an integer
```

### Bitwise Operations
Python supports bitwise operations on integers.

| Operation     | Symbol | Example          | Result |
|---------------|--------|------------------|--------|
| AND           | `&`    | `10 & 7`         | `2`    |
| OR            | `|`    | `10 | 7`         | `15`   |
| XOR           | `^`    | `10 ^ 7`         | `13`   |
| NOT           | `~`    | `~10`            | `-11`  |
| Left Shift    | `<<`   | `10 << 2`        | `40`   |
| Right Shift   | `>>`   | `10 >> 2`        | `2`    |

### Summary
- **Creation**: Assign a whole number to a variable.
- **Type Checking**: Use `type()` to check if a variable is an integer.
- **Operations**: Supports addition, subtraction, multiplication, division, floor division, modulus, and exponentiation.
- **Built-in Functions**: Includes `abs()`, `pow()`, `divmod()`, and `int()`.
- **Type Conversion**: Convert other types to integers using `int()`.
- **Limits**: Integers have arbitrary precision.
- **Bitwise Operations**: Supports AND, OR, XOR, NOT, left shift, and right shift.

This note covers the essential aspects of integers in Python, including creation, operations, built-in functions, type conversion, limits, and bitwise operations.


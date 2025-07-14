
## Float

### In-Depth Note on Python Float

#### Properties of Float

- **Immutable**: Float values cannot be changed after creation.
- **Precision**: Floats are represented in double-precision (64-bit) format.
- **Range**: Floats can represent a wide range of values, including very large and very small numbers.

#### Creating a Float

1. **From a Literal**
   ```python
   my_float = 3.14
   ```

2. **From an Integer**
   ```python
   my_float = float(10)
   ```

3. **From a String**
   ```python
   my_float = float("3.14")
   ```

#### Float Operations

1. **Arithmetic Operations**
   ```python
   a = 5.0
   b = 2.0

   addition = a + b       # 7.0
   subtraction = a - b    # 3.0
   multiplication = a * b # 10.0
   division = a / b       # 2.5
   exponentiation = a ** b # 25.0
   ```

2. **Floor Division**
   ```python
   floor_div = a // b  # 2.0
   ```

3. **Modulus**
   ```python
   modulus = a % b  # 1.0
   ```

#### Float Methods

1. **`is_integer` Method**: Checks if the float is an integer
   ```python
   result = my_float.is_integer()  # Returns True if my_float is an integer
   ```

2. **`as_integer_ratio` Method**: Returns a pair of integers whose ratio is exactly equal to the float
   ```python
   ratio = my_float.as_integer_ratio()  # (314, 100) for 3.14
   ```

3. **`hex` Method**: Returns a hexadecimal string representation of the float
   ```python
   hex_value = my_float.hex()  # '0x1.91eb851eb851fp+1' for 3.14
   ```

4. **`fromhex` Method**: Creates a float from a hexadecimal string
   ```python
   my_float = float.fromhex('0x1.91eb851eb851fp+1')  # 3.14
   ```

#### Float Formatting

1. **Using `format` Function**
   ```python
   formatted = format(my_float, ".2f")  # '3.14'
   ```

2. **Using f-Strings (Python 3.6+)**
   ```python
   formatted = f"{my_float:.2f}"  # '3.14'
   ```

3. **Using `str.format` Method**
   ```python
   formatted = "{:.2f}".format(my_float)  # '3.14'
   ```

#### Special Float Values

1. **Positive Infinity**
   ```python
   pos_inf = float('inf')
   ```

2. **Negative Infinity**
   ```python
   neg_inf = float('-inf')
   ```

3. **NaN (Not a Number)**
   ```python
   nan_value = float('nan')
   ```

#### Checking for Special Float Values

1. **Checking for Infinity**
   ```python
   import math
   is_inf = math.isinf(pos_inf)  # True
   ```

2. **Checking for NaN**
   ```python
   is_nan = math.isnan(nan_value)  # True
   ```

### Python `int` Methods and Functions

#### Methods Called Using `int` Object

1. **`int.bit_length()`**
   - **Description**: Returns the number of bits necessary to represent the integer in binary, excluding the sign and leading zeros.
   - **Usage**:
     ```python
     num = 10
     bit_length = num.bit_length()  # 4
     ```

2. **`int.to_bytes(length, byteorder, (optional) *, (optional) signed=False)`**
   - **Description**: Returns an array of bytes representing the integer.
     - `length`: The number of bytes to use.
     - `byteorder`: The byte order used to represent the integer. Can be either 'big' or 'little'.
     - `(optional) signed`: Whether two's complement is used to represent the integer. Default is `False`.
   - **Usage**:
     ```python
     num = 1024
     byte_array = num.to_bytes(2, byteorder='big')  # b'\x04\x00'
     ```

3. **`int.from_bytes(bytes, byteorder, (optional) *, (optional) signed=False)`**
   - **Description**: Returns the integer represented by the given array of bytes.
     - `bytes`: The array of bytes.
     - `byteorder`: The byte order used to represent the integer. Can be either 'big' or 'little'.
     - `(optional) signed`: Whether two's complement is used to represent the integer. Default is `False`.
   - **Usage**:
     ```python
     byte_array = b'\x04\x00'
     num = int.from_bytes(byte_array, byteorder='big')  # 1024
     ```

#### Functions

1. **`abs(x)`**
   - **Description**: Returns the absolute value of the integer.
     - `x`: The integer whose absolute value is to be returned.
   - **Usage**:
     ```python
     result = abs(-10)  # 10
     ```

2. **`bin(x)`**
   - **Description**: Converts an integer number to a binary string prefixed with "0b".
     - `x`: The integer to be converted to a binary string.
   - **Usage**:
     ```python
     binary_string = bin(10)  # '0b1010'
     ```

3. **`hex(x)`**
   - **Description**: Converts an integer number to a lowercase hexadecimal string prefixed with "0x".
     - `x`: The integer to be converted to a hexadecimal string.
   - **Usage**:
     ```python
     hex_string = hex(255)  # '0xff'
     ```

4. **`oct(x)`**
   - **Description**: Converts an integer number to an octal string prefixed with "0o".
     - `x`: The integer to be converted to an octal string.
   - **Usage**:
     ```python
     octal_string = oct(8)  # '0o10'
     ```

5. **`pow(x, y, (optional) z=None)`**
   - **Description**: Returns `x` raised to the power `y`. If `z` is provided, returns `x` raised to the power `y`, modulo `z`.
     - `x`: The base integer.
     - `y`: The exponent integer.
     - `(optional) z`: The modulus integer. Default is `None`.
   - **Usage**:
     ```python
     result = pow(2, 3)  # 8
     result_mod = pow(2, 3, 3)  # 2
     ```

6. **`round(number, (optional) ndigits=None)`**
   - **Description**: Rounds a number to a given precision in decimal digits. If `ndigits` is omitted or `None`, it returns the nearest integer.
     - `number`: The number to be rounded.
     - `(optional) ndigits`: The number of decimal digits to round to. Default is `None`.
   - **Usage**:
     ```python
     rounded_value = round(5.678, 2)  # 5.68
     rounded_int = round(5.678)  # 6
     ```

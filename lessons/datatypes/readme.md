
# Datatypes:

## Built-in Data Types:

In programming, data type is an important concept.

Variables can store data of different types, and different types can do different things.

Python has the following data types built-in by default, in these categories:

- **Text Type**: `str`
- **Numeric Types**: `int`, `float`, `complex`
- **Sequence Types**: `list`, `tuple`, `range`
- **Mapping Type**: `dict`
- **Set Types**: `set`, `frozenset`
- **Boolean Type**: `bool`
- **Binary Types**: `bytes`, `bytearray`, `memoryview`
- **None Type**: `NoneType`

### Getting the Data Type: type()

You can get the data type of any object by using the `type()` function:

#### Example
```python
x = 5
print(type(x))  # Output: <class 'int'>
```

### Setting the Data Type

In Python, the data type is set when you assign a value to a variable:

#### Example
```python
x = "Hello World"  # str
x = 20  # int
x = 20.5  # float
x = 1j  # complex
x = ["apple", "banana", "cherry"]  # list
x = ("apple", "banana", "cherry")  # tuple
x = range(6)  # range
x = {"name": "John", "age": 36}  # dict
x = {"apple", "banana", "cherry"}  # set
x = frozenset({"apple", "banana", "cherry"})  # frozenset
x = True  # bool
x = b"Hello"  # bytes
x = bytearray(5)  # bytearray
x = memoryview(bytes(5))  # memoryview
x = None  # NoneType
```

### Setting the Specific Data Type

If you want to specify the data type, you can use the following constructor functions:

#### Example
```python
x = str("Hello World")  # str
x = int(20)  # int
x = float(20.5)  # float
x = complex(1j)  # complex
x = list(("apple", "banana", "cherry"))  # list
x = tuple(("apple", "banana", "cherry"))  # tuple
x = range(6)  # range
x = dict(name="John", age=36)  # dict
x = set(("apple", "banana", "cherry"))  # set
x = frozenset(("apple", "banana", "cherry"))  # frozenset
x = bool(5)  # bool
x = bytes(5)  # bytes
x = bytearray(5)  # bytearray
x = memoryview(bytes(5))  # memoryview
```

### Type Casting

Type casting is the process of converting one data type to another. Python supports several built-in functions for type casting.

#### Example
```python
# Convert to integer
x = int(1)  # x will be 1
y = int(2.8)  # y will be 2
z = int("3")  # z will be 3

# Convert to float
a = float(1)  # a will be 1.0
b = float(2.8)  # b will be 2.8
c = float("3")  # c will be 3.0
d = float("4.2")  # d will be 4.2

# Convert to string
e = str(1)  # e will be '1'
f = str(2.8)  # f will be '2.8'
g = str(3)  # g will be '3'

# Convert to list
h = list((1, 2, 3))  # h will be [1, 2, 3]

# Convert to tuple
i = tuple([1, 2, 3])  # i will be (1, 2, 3)

# Convert to set
j = set([1, 2, 3])  # j will be {1, 2, 3}

# Convert to dictionary
k = dict([(1, 'one'), (2, 'two')])  # k will be {1: 'one', 2: 'two'}

# Convert to boolean
l = bool(1)  # l will be True
m = bool(0)  # m will be False
```

### Summary of Type Casting Functions

- `int(x)`: Converts `x` to an integer.
- `float(x)`: Converts `x` to a float.
- `str(x)`: Converts `x` to a string.
- `list(x)`: Converts `x` to a list.
- `tuple(x)`: Converts `x` to a tuple.
- `set(x)`: Converts `x` to a set.
- `dict(x)`: Converts `x` to a dictionary.
- `bool(x)`: Converts `x` to a boolean.

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

## Complex

### In-Depth Note on Python Complex Numbers

#### Properties of Complex Numbers

- **Complex Number**: A complex number is represented as `a + bj`, where `a` is the real part and `b` is the imaginary part.
- **Immutable**: Complex numbers are immutable.
- **Built-in Type**: Python has a built-in complex number type.

#### Creating Complex Numbers

1. **Using `complex` Function**
   ```python
   z = complex(2, 3)  # Creates a complex number 2 + 3j
   ```

2. **Using Direct Assignment**
   ```python
   z = 2 + 3j  # Creates a complex number 2 + 3j
   ```

#### Accessing Real and Imaginary Parts

1. **Real Part**
   ```python
   real_part = z.real  # Accesses the real part of the complex number
   ```

2. **Imaginary Part**
   ```python
   imaginary_part = z.imag  # Accesses the imaginary part of the complex number
   ```

#### Basic Operations

1. **Addition**
   ```python
   z1 = 2 + 3j
   z2 = 1 + 2j
   result = z1 + z2  # Adds two complex numbers
   ```

2. **Subtraction**
   ```python
   result = z1 - z2  # Subtracts two complex numbers
   ```

3. **Multiplication**
   ```python
   result = z1 * z2  # Multiplies two complex numbers
   ```

4. **Division**
   ```python
   result = z1 / z2  # Divides two complex numbers
   ```

5. **Conjugate**
   ```python
   conjugate = z1.conjugate()  # Returns the conjugate of the complex number
   ```

#### Advanced Operations

1. **Absolute Value (Magnitude)**
   ```python
   magnitude = abs(z1)  # Returns the magnitude of the complex number
   ```

2. **Phase Angle**
   ```python
   import cmath
   phase = cmath.phase(z1)  # Returns the phase angle of the complex number
   ```

3. **Polar Coordinates**
   ```python
   polar = cmath.polar(z1)  # Converts the complex number to polar coordinates (r, theta)
   ```

4. **Rectangular Coordinates**
   ```python
   rectangular = cmath.rect(polar[0], polar[1])  # Converts polar coordinates back to rectangular form
   ```

#### Complex Number Methods

1. **`conjugate` Method**: Returns the complex conjugate
   ```python
   conjugate = z1.conjugate()
   ```

2. **`__add__` Method**: Adds two complex numbers
   ```python
   result = z1.__add__(z2)
   ```

3. **`__sub__` Method**: Subtracts two complex numbers
   ```python
   result = z1.__sub__(z2)
   ```

4. **`__mul__` Method**: Multiplies two complex numbers
   ```python
   result = z1.__mul__(z2)
   ```

5. **`__truediv__` Method**: Divides two complex numbers
   ```python
   result = z1.__truediv__(z2)
   ```

6. **`__abs__` Method**: Returns the magnitude of the complex number
   ```python
   magnitude = z1.__abs__()
   ```

#### Example Usage

1. **Creating and Operating on Complex Numbers**
   ```python
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

### Complex Number Methods and Functions

#### Methods Called Using Complex Object

1. **`complex.conjugate()`**
   - **Description**: Returns the complex conjugate of the complex number.
   - **Parameters**: None

2. **`complex.__add__(self, other)`**
   - **Description**: Adds two complex numbers.
   - **Parameters**:
     - `self`: The first complex number.
     - `other`: The second complex number to add.

3. **`complex.__sub__(self, other)`**
   - **Description**: Subtracts one complex number from another.
   - **Parameters**:
     - `self`: The first complex number.
     - `other`: The complex number to subtract.

4. **`complex.__mul__(self, other)`**
   - **Description**: Multiplies two complex numbers.
   - **Parameters**:
     - `self`: The first complex number.
     - `other`: The second complex number to multiply.

5. **`complex.__truediv__(self, other)`**
   - **Description**: Divides one complex number by another.
   - **Parameters**:
     - `self`: The first complex number.
     - `other`: The complex number to divide by.

6. **`complex.__abs__(self)`**
   - **Description**: Returns the magnitude (absolute value) of the complex number.
   - **Parameters**:
     - `self`: The complex number.

#### Functions

1. **`complex(real, imag=0.0)`**
   - **Description**: Creates a complex number from a real part and an (optional) imaginary part.
   - **Parameters**:
     - `real`: The real part of the complex number.
     - `imag` (optional): The imaginary part of the complex number. Defaults to `0.0`.

2. **`abs(x)`**
   - **Description**: Returns the magnitude (absolute value) of a complex number.
   - **Parameters**:
     - `x`: The complex number.

3. **`cmath.phase(x)`**
   - **Description**: Returns the phase angle (in radians) of a complex number.
   - **Parameters**:
     - `x`: The complex number.

4. **`cmath.polar(x)`**
   - **Description**: Converts a complex number to polar coordinates.
   - **Parameters**:
     - `x`: The complex number.

5. **`cmath.rect(r, phi)`**
   - **Description**: Converts polar coordinates to a complex number.
   - **Parameters**:
     - `r`: The magnitude (radius).
     - `phi`: The phase angle (in radians).


## String

### Properties and usage of String

#### Properties of Strings

- **Immutable**: Strings cannot be changed after creation.
- **Indexed**: Strings are indexed by position, starting from 0.
- **Iterable**: Strings can be iterated over.

#### Creating Strings

1. **Single Quotes**
   ```python
   my_str = 'Hello, World```'
   ```

2. **Double Quotes**
   ```python
   my_str = "Hello, World```"
   ```

3. **Triple Quotes** (for multi-line strings)
   ```python
   my_str = '''Hello,
   World```'''
   ```

#### Accessing Characters

1. **Using Indexing**
   ```python
   char = my_str[0]  # First character
   ```

2. **Using Slicing**
   ```python
   substring = my_str[0:5]  # Substring from index 0 to 4
   ```

#### String Operations

1. **Concatenation**
   ```python
   new_str = my_str + " How are you?"
   ```

2. **Repetition**
   ```python
   repeated_str = my_str * 3
   ```

3. **Length**
   ```python
   length = len(my_str)
   ```

4. **Membership**
   ```python
   if "Hello" in my_str:
       # Code to execute if "Hello" is in my_str
   ```

#### String Methods

1. **`upper` Method**: Converts all characters to uppercase
   ```python
   upper_str = my_str.upper()
   ```

2. **`lower` Method**: Converts all characters to lowercase
   ```python
   lower_str = my_str.lower()
   ```

3. **`strip` Method**: Removes leading and trailing whitespace
   ```python
   stripped_str = my_str.strip()
   ```

4. **`split` Method**: Splits the string into a list of substrings
   ```python
   words = my_str.split()
   ```

5. **`join` Method**: Joins a list of strings into a single string
   ```python
   joined_str = " ".join(words)
   ```

6. **`replace` Method**: Replaces occurrences of a substring with another substring
   ```python
   replaced_str = my_str.replace("World", "Python")
   ```

7. **`find` Method**: Returns the index of the first occurrence of a substring
   ```python
   index = my_str.find("World")
   ```

8. **`count` Method**: Returns the number of occurrences of a substring
   ```python
   count = my_str.count("l")
   ```

9. **`startswith` Method**: Checks if the string starts with a specified substring
   ```python
   starts = my_str.startswith("Hello")
   ```

10. **`endswith` Method**: Checks if the string ends with a specified substring
    ```python
    ends = my_str.endswith("```")
    ```

11. **`format` Method**: Formats the string using placeholders
    ```python
    formatted_str = "Hello, {}```".format("Python")
    ```

12. **`isalpha` Method**: Checks if all characters in the string are alphabetic
    ```python
    is_alpha = my_str.isalpha()
    ```

13. **`isdigit` Method**: Checks if all characters in the string are digits
    ```python
    is_digit = my_str.isdigit()
    ```

14. **`isalnum` Method**: Checks if all characters in the string are alphanumeric
    ```python
    is_alnum = my_str.isalnum()
    ```

15. **`title` Method**: Converts the first character of each word to uppercase
    ```python
    title_str = my_str.title()
    ```

16. **`capitalize` Method**: Converts the first character to uppercase and the rest to lowercase
    ```python
    capitalized_str = my_str.capitalize()
    ```

17. **`swapcase` Method**: Swaps the case of all characters
    ```python
    swapped_str = my_str.swapcase()
    ```

#### String Formatting

1. **Using `format` Method**
   ```python
   formatted_str = "Hello, {}```".format("Python")
   ```

2. **Using f-Strings (Python 3.6+)**
   ```python
   name = "Python"
   formatted_str = f"Hello, {name}```"
   ```

3. **Using `%` Operator**
   ```python
   formatted_str = "Hello, %s```" % "Python"
   ```

### String Method Descriptions


- `str.capitalize()` - Converts the first character to upper case.
- `str.casefold()` - Converts string into lower case.
- `str.center(width, fillchar=' ')` - Returns a centered string. `width` specifies the total length of the string, and `fillchar` is the character to fill the padding (default is space).
- `str.count(sub, start=0, end=len(string))` - Returns the number of times a specified value occurs in a string. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.encode(encoding='utf-8', errors='strict')` - Returns an encoded version of the string. `encoding` specifies the encoding to use, and `errors` specifies the error handling scheme.
- `str.endswith(suffix, start=0, end=len(string))` - Returns true if the string ends with the specified value. `suffix` is the substring to check, `start` and `end` specify the range to check within.
- `str.expandtabs(tabsize=8)` - Sets the tab size of the string. `tabsize` specifies the number of spaces to use per tab.
- `str.find(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.format(*args, **kwargs)` - Formats specified values in a string. `args` and `kwargs` are the values to format into the string.
- `str.format_map(mapping)` - Formats specified values in a string. `mapping` is a dictionary-like object with the values to format into the string.
- `str.index(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.isalnum()` - Returns True if all characters in the string are alphanumeric.
- `str.isalpha()` - Returns True if all characters in the string are in the alphabet.
- `str.isascii()` - Returns True if all characters in the string are ASCII characters.
- `str.isdecimal()` - Returns True if all characters in the string are decimals.
- `str.isdigit()` - Returns True if all characters in the string are digits.
- `str.isidentifier()` - Returns True if the string is a valid identifier.
- `str.islower()` - Returns True if all characters in the string are lower case.
- `str.isnumeric()` - Returns True if all characters in the string are numeric.
- `str.isprintable()` - Returns True if all characters in the string are printable.
- `str.isspace()` - Returns True if all characters in the string are whitespaces.
- `str.istitle()` - Returns True if the string follows the rules of a title.
- `str.isupper()` - Returns True if all characters in the string are upper case.
- `str.join(iterable)` - Joins the elements of an iterable to the end of the string. `iterable` is the collection of elements to join.
- `str.ljust(width, fillchar=' ')` - Returns a left justified version of the string. `width` specifies the total length of the string, and `fillchar` is the character to fill the padding (default is space).
- `str.lower()` - Converts a string into lower case.
- `str.lstrip(chars=None)` - Returns a left trim version of the string. `chars` specifies the set of characters to remove (default is whitespace).
- `str.maketrans(x, y=None, z=None)` - Returns a translation table to be used in translations. `x` and `y` are strings of equal length, and `z` is a string with characters to delete.
- `str.partition(sep)` - Returns a tuple where the string is parted into three parts. `sep` is the separator to use.
- `str.replace(old, new, count=-1)` - Returns a string where a specified value is replaced with a specified value. `old` is the substring to replace, `new` is the replacement, and `count` specifies the number of replacements (default is all).
- `str.rfind(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the last position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.rindex(sub, start=0, end=len(string))` - Searches the string for a specified value and returns the last position of where it was found. `sub` is the substring to search for, `start` and `end` specify the range to search within.
- `str.rjust(width, fillchar=' ')` - Returns a right justified version of the string. `width` specifies the total length of the string, and `fillchar` is the character to fill the padding (default is space).
- `str.rpartition(sep)` - Returns a tuple where the string is parted into three parts. `sep` is the separator to use.
- `str.rsplit(sep=None, maxsplit=-1)` - Splits the string at the specified separator, and returns a list. `sep` is the delimiter to split by (default is whitespace), and `maxsplit` specifies the maximum number of splits.
- `str.rstrip(chars=None)` - Returns a right trim version of the string. `chars` specifies the set of characters to remove (default is whitespace).
- `str.split(sep=None, maxsplit=-1)` - Splits the string at the specified separator, and returns a list. `sep` is the delimiter to split by (default is whitespace), and `maxsplit` specifies the maximum number of splits.
- `str.splitlines(keepends=False)` - Splits the string at line breaks and returns a list. `keepends` specifies whether to keep the line breaks (default is False).
- `str.startswith(prefix, start=0, end=len(string))` - Returns true if the string starts with the specified value. `prefix` is the substring to check, `start` and `end` specify the range to check within.
- `str.strip(chars=None)` - Returns a trimmed version of the string. `chars` specifies the set of characters to remove (default is whitespace).
- `str.swapcase()` - Swaps cases, lower case becomes upper case and vice versa.
- `str.title()` - Converts the first character of each word to upper case.
- `str.translate(table)` - Returns a translated string. `table` is a translation table to use.
- `str.upper()` - Converts a string into upper case.
- `str.zfill(width)` - Fills the string with a specified number of 0 values at the beginning. `width` specifies the total length of the string.

## List:

### Properties and usage of Python Lists

#### Properties of Lists

- **Ordered**: Lists maintain the order of elements.
- **Mutable**: Lists can be changed after creation.
- **Indexed**: Lists are indexed by integers, starting from 0.
- **Heterogeneous**: Lists can contain elements of different data types.

#### Creating a List

1. **Empty List**
   ```python
   my_list = []
   ```

2. **List with Initial Values**
   ```python
   my_list = [1, 2, 3, "four", 5.0]
   ```

#### Accessing Elements

1. **Using Index**
   ```python
   element = my_list[0]  # Access the first element
   ```

2. **Using Negative Index**
   ```python
   element = my_list[-1]  # Access the last element
   ```

#### Adding Elements

1. **Using `append` Method** (adds an element to the end)
   ```python
   my_list.append("new_element")
   ```

2. **Using `insert` Method** (inserts an element at a specified position)
   ```python
   my_list.insert(1, "inserted_element")
   ```

3. **Using `extend` Method** (extends the list by appending elements from an iterable)
   ```python
   my_list.extend([6, 7, 8])
   ```

#### Removing Elements

1. **Using `remove` Method** (removes the first occurrence of a value)
   ```python
   my_list.remove("four")
   ```

2. **Using `pop` Method** (removes and returns the element at a specified position)
   ```python
   element = my_list.pop(2)
   ```

3. **Using `clear` Method** (removes all elements)
   ```python
   my_list.clear()
   ```

#### List Operations

1. **Checking if an Element Exists**
   ```python
   if "new_element" in my_list:
       # Code to execute if "new_element" exists in my_list
   ```

2. **Iterating Over Elements**
   ```python
   for element in my_list:
       print(element)
   ```

3. **List Comprehension**
   ```python
   squared_numbers = [x*x for x in range(5)]
   ```

4. **Slicing a List**
   ```python
   sub_list = my_list[1:3]  # Get elements from index 1 to 2
   ```

#### List Methods

1. **`index` Method**: Returns the index of the first occurrence of a value
   ```python
   index = my_list.index("new_element")
   ```

2. **`count` Method**: Returns the number of occurrences of a value
   ```python
   count = my_list.count(2)
   ```

3. **`sort` Method**: Sorts the list in ascending order
   ```python
   my_list.sort()
   ```

4. **`reverse` Method**: Reverses the elements of the list
   ```python
   my_list.reverse()
   ```

5. **`copy` Method**: Returns a shallow copy of the list
   ```python
   new_list = my_list.copy()
   ```

#### List Functions

1. **`len` Function**: Returns the number of elements in the list
   ```python
   length = len(my_list)
   ```

2. **`max` Function**: Returns the largest element in the list
   ```python
   maximum = max(my_list)
   ```

3. **`min` Function**: Returns the smallest element in the list
   ```python
   minimum = min(my_list)
   ```

4. **`sum` Function**: Returns the sum of all elements in the list
   ```python
   total = sum(my_list)
   ```

### List Methods

- `list.append(element)` - Adds an element at the end of the list. `element` is the item to be added.
- `list.clear()` - Removes all the elements from the list.
- `list.copy()` - Returns a copy of the list.
- `list.count(value)` - Returns the number of elements with the specified value. `value` is the item to be counted.
- `list.extend(iterable)` - Adds the elements of a list (or any iterable) to the end of the current list. `iterable` is the collection of elements to be added.
- `list.index(value, start=0, end=len(list))` - Returns the index of the first element with the specified value. `value` is the item to search for, `start` and `end` specify the range to search within.
- `list.insert(index, element)` - Adds an element at the specified position. `index` is the position to insert the element, and `element` is the item to be added.
- `list.pop(index=-1)` - Removes the element at the specified position. `index` is the position of the element to be removed (default is the last item).
- `list.remove(value)` - Removes the item with the specified value. `value` is the item to be removed.
- `list.reverse()` - Reverses the order of the list.
- `list.sort(key=None, reverse=False)` - Sorts the list. `key` is a function that serves as a key for the sort comparison, and `reverse` is a boolean value to sort in descending order.

## Tuples

### In-Depth Note on Python Tuples

#### Properties of Tuples

- **Ordered**: Tuples maintain the order of elements.
- **Immutable**: Once created, the elements of a tuple cannot be changed.
- **Indexed**: Elements can be accessed using indices.
- **Heterogeneous**: Tuples can contain elements of different data types.
- **Hashable**: Tuples can be used as keys in dictionaries if they contain only hashable types.

#### Creating Tuples

1. **Empty Tuple**
   ```python
   my_tuple = ()
   ```

2. **Tuple with One Element** (note the comma)
   ```python
   my_tuple = (element,)
   ```

3. **Tuple with Multiple Elements**
   ```python
   my_tuple = (element1, element2, element3)
   ```

4. **Tuple Without Parentheses** (tuple packing)
   ```python
   my_tuple = element1, element2, element3
   ```

5. **Using `tuple` Constructor**
   ```python
   my_tuple = tuple([element1, element2, element3])
   ```

#### Accessing Elements

1. **Using Indexing**
   ```python
   element = my_tuple[0]
   ```

2. **Using Negative Indexing**
   ```python
   element = my_tuple[-1]
   ```

3. **Using Slicing**
   ```python
   sub_tuple = my_tuple[1:3]
   ```

#### Tuple Operations

1. **Concatenation**
   ```python
   new_tuple = my_tuple1 + my_tuple2
   ```

2. **Repetition**
   ```python
   repeated_tuple = my_tuple * 3
   ```

3. **Membership Test**
   ```python
   if element in my_tuple:
       # Code to execute if element is in my_tuple
   ```

4. **Iterating Over Elements**
   ```python
   for element in my_tuple:
       print(element)
   ```

#### Tuple Methods

1. **`count` Method**: Returns the number of occurrences of a specified value
   ```python
   count = my_tuple.count(element)
   ```

2. **`index` Method**: Returns the index of the first occurrence of a specified value
   ```python
   index = my_tuple.index(element)
   ```

#### Unpacking Tuples

1. **Basic Unpacking**
   ```python
   a, b, c = my_tuple
   ```

2. **Unpacking with `*` Operator**
   ```python
   a, *b, c = my_tuple
   ```

#### Nested Tuples

1. **Creating Nested Tuples**
   ```python
   nested_tuple = (element1, (element2, element3), element4)
   ```

2. **Accessing Elements in Nested Tuples**
   ```python
   inner_element = nested_tuple[1][0]
   ```

#### Tuple Comprehension (Not Directly Supported, Use Generator Expression)

1. **Using Generator Expression**
   ```python
   my_tuple = tuple(x*x for x in range(5))
   ```

### Tuple Methods

- `tuple.count(value)` - Returns the number of times a specified value occurs in a tuple. `value` is the element to count.
- `tuple.index(value, start=0, end=len(tuple))` - Searches the tuple for a specified value and returns the position of where it was found. `value` is the element to search for, `start` and `end` specify the range to search within.

## Sets:

### In-Depth Note on Python Sets

#### Properties of Sets

- **Unordered**: Sets are unordered collections of items.
- **Mutable**: Sets can be changed after creation.
- **Unique Elements**: Sets do not allow duplicate elements.
- **Immutable Elements**: Elements in a set must be immutable (e.g., strings, numbers, tuples).

#### Creating a Set

1. **Empty Set**
   ```python
   my_set = set()
   ```

2. **Set with Initial Values**
   ```python
   my_set = {1, 2, 3, 4, 5}
   ```

#### Adding Elements

1. **Using `add` Method**
   ```python
   my_set.add(6)
   ```

2. **Using `update` Method** (adds multiple elements)
   ```python
   my_set.update([7, 8, 9])
   ```

#### Removing Elements

1. **Using `remove` Method** (raises KeyError if element not found)
   ```python
   my_set.remove(3)
   ```

2. **Using `discard` Method** (does not raise an error if element not found)
   ```python
   my_set.discard(4)
   ```

3. **Using `pop` Method** (removes and returns an arbitrary element)
   ```python
   element = my_set.pop()
   ```

4. **Using `clear` Method** (removes all elements)
   ```python
   my_set.clear()
   ```

#### Set Operations

1. **Union**
   ```python
   set1 = {1, 2, 3}
   set2 = {3, 4, 5}
   union_set = set1 | set2  # or set1.union(set2)
   ```

2. **Intersection**
   ```python
   intersection_set = set1 & set2  # or set1.intersection(set2)
   ```

3. **Difference**
   ```python
   difference_set = set1 - set2  # or set1.difference(set2)
   ```

4. **Symmetric Difference**
   ```python
   sym_diff_set = set1 ^ set2  # or set1.symmetric_difference(set2)
   ```

#### Set Methods

1. **`add` Method**: Adds an element to the set
   ```python
   my_set.add(10)
   ```

2. **`update` Method**: Adds multiple elements to the set
   ```python
   my_set.update([11, 12])
   ```

3. **`remove` Method**: Removes an element from the set (raises KeyError if not found)
   ```python
   my_set.remove(10)
   ```

4. **`discard` Method**: Removes an element from the set (does not raise an error if not found)
   ```python
   my_set.discard(11)
   ```

5. **`pop` Method**: Removes and returns an arbitrary element from the set
   ```python
   element = my_set.pop()
   ```

6. **`clear` Method**: Removes all elements from the set
   ```python
   my_set.clear()
   ```

7. **`union` Method**: Returns the union of sets
   ```python
   union_set = set1.union(set2)
   ```

8. **`intersection` Method**: Returns the intersection of sets
   ```python
   intersection_set = set1.intersection(set2)
   ```

9. **`difference` Method**: Returns the difference of sets
   ```python
   difference_set = set1.difference(set2)
   ```

10. **`symmetric_difference` Method**: Returns the symmetric difference of sets
    ```python
    sym_diff_set = set1.symmetric_difference(set2)
    ```

11. **`issubset` Method**: Checks if one set is a subset of another
    ```python
    is_subset = set1.issubset(set2)
    ```

12. **`issuperset` Method**: Checks if one set is a superset of another
    ```python
    is_superset = set1.issuperset(set2)
    ```

13. **`isdisjoint` Method**: Checks if two sets have no elements in common
    ```python
    is_disjoint = set1.isdisjoint(set2)
    ```

#### Set Comprehension

1. **Basic Set Comprehension**
   ```python
   squared_set = {x*x for x in range(10)}
   ```

### Set Methods

- `set.add(elem)` - Adds an element `elem` to the set.
- `set.clear()` - Removes all the elements from the set.
- `set.copy()` - Returns a copy of the set.
- `set.difference(*others)` - Returns a set containing the difference between this set and `others`.
- `set.difference_update(*others)` - Removes the items in this set that are also included in `others`.
- `set.discard(elem)` - Removes the specified element `elem` from the set if it is present.
- `set.intersection(*others)` - Returns a set that is the intersection of this set and `others`.
- `set.intersection_update(*others)` - Removes the items in this set that are not present in `others`.
- `set.isdisjoint(other)` - Returns whether two sets have an intersection or not.
- `set.issubset(other)` - Returns whether this set is a subset of `other`.
- `set.issuperset(other)` - Returns whether this set is a superset of `other`.
- `set.pop()` - Removes and returns an arbitrary element from the set.
- `set.remove(elem)` - Removes the specified element `elem` from the set. Raises a KeyError if `elem` is not found.
- `set.symmetric_difference(other)` - Returns a set with the symmetric differences of this set and `other`.
- `set.symmetric_difference_update(other)` - Updates this set with the symmetric differences of this set and `other`.
- `set.union(*others)` - Returns a set containing the union of this set and `others`.
- `set.update(*others)` - Updates this set with the union of this set and `others`.

### Method Shortcuts

- `set.difference(*others)` - `-`
- `set.difference_update(*others)` - `-=`
- `set.intersection(*others)` - `&`
- `set.intersection_update(*others)` - `&=`
- `set.issubset(other)` - `<=`
- `set.issuperset(other)` - `>=`
- `set.symmetric_difference(other)` - `^`
- `set.symmetric_difference_update(other)` - `^=`
- `set.union(*others)` - `|`
- `set.update(*others)` - `|=`

## Dictionaries

#### Properties of Dictionaries

- **Unordered**: Dictionaries are unordered collections of items.
- **Mutable**: Dictionaries can be changed after creation.
- **Indexed**: Dictionaries are indexed by keys.
- **Keys**: Keys must be unique and immutable (e.g., strings, numbers, tuples).
- **Values**: Values can be of any data type and can be `duplicated` and `nested`.


#### Creating a Dictionary

1. **Empty Dictionary**
   ```python
   my_dict = {}
   ```

2. **Dictionary with Initial Values**
   ```python
   my_dict = {
       "key1": "value1",
       "key2": "value2",
       "key3": "value3"
   }
   ```

#### Accessing Values

1. **Using Keys**
   ```python
   value = my_dict["key1"]
   ```

2. **Using `get` Method**
   ```python
   value = my_dict.get("key1")
   ```

#### Adding and Updating Values

1. **Adding a New Key-Value Pair**
   ```python
   my_dict["new_key"] = "new_value"
   ```

2. **Updating an Existing Key-Value Pair**
   ```python
   my_dict["key1"] = "updated_value"
   ```

#### Removing Values

1. **Using `del` Statement**
   ```python
   del my_dict["key1"]
   ```

2. **Using `pop` Method**
   ```python
   value = my_dict.pop("key2")
   ```

3. **Using `popitem` Method** (removes the last inserted key-value pair)
   ```python
   key, value = my_dict.popitem()
   ```

4. **Using `clear` Method** (removes all items)
   ```python
   my_dict.clear()
   ```

#### Dictionary Operations

1. **Checking if a Key Exists**
   ```python
   if "key1" in my_dict:
       # Code to execute if key1 exists in my_dict
   ```

2. **Iterating Over Keys**
   ```python
   for key in my_dict:
       print(key)
   ```

3. **Iterating Over Values**
   ```python
   for value in my_dict.values():
       print(value)
   ```

4. **Iterating Over Key-Value Pairs**
   ```python
   for key, value in my_dict.items():
       print(key, value)
   ```

5. **Dictionary Comprehension**
   ```python
   squared_numbers = {x: x*x for x in iterable}
   ```

#### Dictionary Methods

1. **`keys` Method**: Returns a view object of all keys
   ```python
   keys = my_dict.keys()
   ```

2. **`values` Method**: Returns a view object of all values
   ```python
   values = my_dict.values()
   ```

3. **`items` Method**: Returns a view object of all key-value pairs
   ```python
   items = my_dict.items()
   ```

4. **`update` Method**: Updates the dictionary with elements from another dictionary or an iterable of key-value pairs
   ```python
   my_dict.update({"key4": "value4", "key5": "value5"})
   ```

5. **`copy` Method**: Returns a shallow copy of the dictionary
   ```python
   new_dict = my_dict.copy()
   ```

6. **`fromkeys` Method**: Creates a new dictionary with keys from an iterable and values set to a specified value
   ```python
   keys = ["a", "b", "c"]
   new_dict = dict.fromkeys(keys, 0)
   ```

7. **`setdefault` Method**: Returns the value of a key if it is in the dictionary; if not, inserts the key with a specified value
   ```python
   value = my_dict.setdefault("key6", "default_value")
   ```


### Dictionary Methods

- `dict.clear()` - Removes all the elements from the dictionary.
- `dict.copy()` - Returns a copy of the dictionary.
- `dict.fromkeys(keys, value=None)` - Returns a dictionary with the specified keys and value. `keys` is an iterable of keys, and `value` is the value to set for all keys (default is None).
- `dict.get(key, default=None)` - Returns the value of the specified key. `key` is the key to look up, and `default` is the value to return if the key is not found (default is None).
- `dict.items()` - Returns a view object that displays a list of a dictionary's key-value tuple pairs.
- `dict.keys()` - Returns a view object that displays a list of all the keys in the dictionary.
- `dict.pop(key, default=None)` - Removes the element with the specified key. `key` is the key to remove, and `default` is the value to return if the key is not found (default is None).
- `dict.popitem()` - Removes the last inserted key-value pair.
- `dict.setdefault(key, default=None)` - Returns the value of the specified key. If the key does not exist, inserts the key with the specified value. `key` is the key to look up, and `default` is the value to set if the key is not found (default is None).
- `dict.update([other])` - Updates the dictionary with the specified key-value pairs. `other` can be another dictionary or an iterable of key-value pairs.
- `dict.values()` - Returns a view object that displays a list of all the values in the dictionary.

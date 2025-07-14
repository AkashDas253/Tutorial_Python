
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

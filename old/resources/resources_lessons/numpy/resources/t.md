# NumPy 

## Basics

### About

- NumPy is a Python library used for working with arrays.
- The array object in NumPy is called ndarray, it provides a lot of supporting functions that make working with ndarray very easy.
- NumPy arrays are stored at one continuous place in memory unlike lists, so processes can access and manipulate them very efficiently.
- NumPy is a Python library and is written partially in Python, but most of the parts that require fast computation are written in C or C++.

### Importing NumPy and version
- `import numpy as np`
- `print(np.__version__)`

## Array

### Data types

| Symbol | Type            | Description                             |
|--------|-----------------|-----------------------------------------|
| i      | integer         | Integer data type                       |
| b      | boolean         | Boolean data type                       |
| u      | unsigned integer| Unsigned integer data type              |
| f      | float           | Floating-point data type                |
| c      | complex float   | Complex floating-point data type        |
| m      | timedelta       | Time delta data type                    |
| M      | datetime        | Date and time data type                 |
| O      | object          | Object data type                        |
| S      | string          | String data type                        |
| U      | unicode string  | Unicode string data type                |
| V      | fixed chunk     | Fixed chunk of memory for other type (void) |
### Creating
Creating an N-dimensional array (Ndim) involves initializing the array with or without specifying the data type. Once created, you can check its dimensions and data type.

#### Syntax

```python
import numpy as np

# Create Ndim without specifying data type
ndim_array = np.array(object, dtype=None, copy=True, order='K', subok=False, ndmin=0)

# Create Ndim with specifying data type
ndim_array_dtype = np.array(object, dtype=np.float32, copy=True, order='K', subok=False, ndmin=0)
```

#### Parameters
- **`object`**: Any object exposing the array interface method returns an array, or any (nested) sequence.
- **`dtype=None`**: Desired data type of the array. If not given, the data type is inferred from the input data.
- **`copy=True`**: If true (default), then the object is copied. Otherwise, a copy will only be made if `__array__` returns a copy, if `obj` is a nested sequence, or if a copy is needed to satisfy any of the other requirements (dtype, order, etc.).
- **`order='K'`**: Specify the memory layout order of the array. 'C' means C-order, 'F' means Fortran-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise. 'K' means match the layout of `a` as closely as possible.
- **`subok=False`**: If True, then sub-classes will be passed-through, otherwise the returned array will be forced to be a base-class array (default).
- **`ndmin=0`**: Specifies the minimum number of dimensions that the resulting array should have. Ones will be prepended to the shape as needed to meet this requirement.

#### Check Dimensions
To check the dimensions of the Ndim array, use the `ndim` attribute.

```python
dimensions = ndim_array.ndim
```

#### Check Data Type
To check the data type of the Ndim array, use the `dtype` attribute.

```python
data_type = ndim_array.dtype
```
### Access
Accessing elements in an N-dimensional array (Ndim) can be done using standard indexing, negative indexing, and slicing.

#### Syntax

@python
import numpy as np

# Create an example Ndim array
ndim_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Access ndim
element = ndim_array[row_index, col_index]

# Negative indexing
negative_index_element = ndim_array[-row_index, -col_index]

# Slicing
sliced_array = ndim_array[start_row:end_row, start_col:end_col]
@

#### Parameters
- **`row_index`**: Integer index for the row.
- **`col_index`**: Integer index for the column.
- **`-row_index`**: Negative integer index for the row, counting from the end.
- **`-col_index`**: Negative integer index for the column, counting from the end.
- **`start_row`**: Starting index for the row slice.
- **`end_row`**: Ending index for the row slice.
- **`start_col`**: Starting index for the column slice.
- **`end_col`**: Ending index for the column slice.

### Copying
Copying an N-dimensional array (Ndim) involves creating a copy of the array, viewing the array, and understanding data ownership.

#### Syntax

@python
import numpy as np

# Create an example Ndim array
ndim_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Copy
copied_array = ndim_array.copy(order='K')

# View
view_array = ndim_array.view(dtype=None, type=None)

# Data ownership
owns_data = ndim_array.flags['OWNDATA']
@

#### Parameters
- **`order='K'`**: Specify the memory layout order of the copy. 'C' means C-order, 'F' means Fortran-order, 'A' means 'F' if `a` is Fortran contiguous, 'C' otherwise. 'K' means match the layout of `a` as closely as possible.
- **`dtype=None`**: Data-type descriptor of the returned view, e.g., float32 or int16. If None, the data-type of the view is the same as that of the original array.
- **`type=None`**: Type of the view. If None, the type of the view is the same as that of the original array.

### Shape
Getting and reshaping the shape of an N-dimensional array (Ndim) involves retrieving the current shape and changing it to a new shape.

#### Syntax

@python
import numpy as np

# Create an example Ndim array
ndim_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Get shape
shape = ndim_array.shape

# Reshape
reshaped_array = ndim_array.reshape(new_shape, order='C')
@

#### Parameters
- **`new_shape`**: Tuple of integers representing the new shape of the array.
- **`order='C'`**: Read the elements using this index order. 'C' means to read/write the elements using C-like index order, with the last axis index changing fastest, back to the first axis index changing slowest. 'F' means to read/write the elements using Fortran-like index order, with the first index changing fastest, and the last index changing slowest. 'A' means to read/write the elements in Fortran-like index order if `a` is Fortran contiguous in memory, C-like order otherwise. 'K' means to read/write the elements in the order they occur in memory, except for reversing the data when strides are negative.




## 2. Creating Arrays
- `np.array([1, 2, 3])`  # 1D array
- `np.array([[1, 2], [3, 4]])`  # 2D array
- `np.zeros((2, 3))`  # Array of zeros
- `np.ones((2, 3))`  # Array of ones
- `np.empty((2, 3))`  # Empty array
- `np.arange(0, 10, 1)`  # Array with a range
- `np.linspace(0, 1, 5)`  # Linearly spaced array

## 3. Array Properties
- `array.shape`  # Get shape of array
- `array.size`  # Get number of elements
- `array.ndim`  # Get number of dimensions
- `array.dtype`  # Get data type of elements

## 4. Accessing and Modifying Elements
- `array[0]`  # Access first element
- `array[1, 0]`  # Access element in 2D array
- `array[1:4]`  # Slicing
- `array[1, :]`  # All columns in second row
- `array[::2]`  # Every second element
- Allows negative indexing

## 5. Array Operations
- `array + 1`  # Add 1 to each element
- `array * 2`  # Multiply each element by 2
- `np.sqrt(array)`  # Square root of each element
- `np.exp(array)`  # Exponential of each element
- `np.log(array)`  # Natural logarithm of each element

## 6. Statistical Functions
- `np.mean(array)`  # Mean of array
- `np.median(array)`  # Median of array
- `np.std(array)`  # Standard deviation
- `np.var(array)`  # Variance
- `np.sum(array)`  # Sum of elements
- `np.min(array)`  # Minimum value
- `np.max(array)`  # Maximum value

## 7. Reshaping Arrays
- `array.reshape((3, 2))`  # Reshape array to 3x2
- `np.ravel(array)`  # Flatten array
- `np.transpose(array)`  # Transpose array

## 8. Stacking and Splitting Arrays
- `np.vstack((array1, array2))`  # Vertical stacking
- `np.hstack((array1, array2))`  # Horizontal stacking
- `np.split(array, indices)`  # Split array into sub-arrays

## 9. Boolean Indexing
- `array[array > 5]`  # Elements greater than 5
- `np.where(array > 5)`  # Indices where condition is true

## 10. Linear Algebra
- `np.dot(array1, array2)`  # Dot product
- `np.linalg.inv(array)`  # Inverse of matrix
- `np.linalg.eig(array)`  # Eigenvalues and eigenvectors

## 11. Saving and Loading Arrays
- `np.save('array.npy', array)`  # Save array to file
- `loaded_array = np.load('array.npy')`  # Load array from file

## 12. Random Number Generation
- `np.random.rand(3, 2)`  # Random array with uniform distribution
- `np.random.randn(3, 2)`  # Random array with normal distribution
- `np.random.randint(0, 10, size=(3, 2))`  # Random integers

## 13. Broadcasting
- `array + scalar`  # Add scalar to each element
- `array1 + array2`  # Add two arrays element-wise

## 14. Sorting
- `np.sort(array)`  # Return a sorted copy of an array
- `array.sort()`  # Sort an array in-place
- `np.argsort(array)`  # Indices that would sort an array

## 15. Unique Elements
- `np.unique(array)`  # Find unique elements in an array
- `np.unique(array, return_counts=True)`  # Find unique elements and their counts

## 16. Concatenation
- `np.concatenate((array1, array2), axis=0)`  # Concatenate along the first axis
- `np.concatenate((array1, array2), axis=1)`  # Concatenate along the second axis

## 17. Copying Arrays
- `array.copy()`  # Create a copy of an array
- `np.copy(array)`  # Another way to create a copy of an array

## 18. Creating Arrays
- `np.zeros((3, 2))`  # Create an array of zeros
- `np.ones((3, 2))`  # Create an array of ones
- `np.full((3, 2), 7)`  # Create an array filled with a constant value
- `np.eye(3)`  # Create an identity matrix
- `np.linspace(0, 10, 5)`  # Create an array of 5 evenly spaced values between 0 and 10
- `np.arange(0, 10, 2)`  # Create an array with values from 0 to 10 with a step of 2

## 19. Mathematical Functions
- `np.sin(array)`  # Sine of each element
- `np.cos(array)`  # Cosine of each element
- `np.tan(array)`  # Tangent of each element

## 20. Aggregation Functions
- `np.cumsum(array)`  # Cumulative sum of elements
- `np.cumprod(array)`  # Cumulative product of elements

## 21. Handling NaN Values
- `np.isnan(array)`  # Check for NaN values
- `np.nanmean(array)`  # Mean ignoring NaN values
- `np.nanstd(array)`  # Standard deviation ignoring NaN values

## 22. Fancy Indexing
- `array[[0, 2, 4]]`  # Access elements at specific indices
- `array[[0, 2, 4], [1, 3, 5]]`  # Access elements at specific row and column indices

## 23. Memory Layout
- `array.T`  # Transpose of the array
- `array.flatten()`  # Flatten the array
- `array.ravel()`  # Return a flattened array

## 24. Creating Structured Arrays
- `np.zeros(3, dtype=[('x', 'i4'), ('y', 'f4')])`  # Create a structured array

## 25. Advanced Linear Algebra
- `np.linalg.svd(array)`  # Singular Value Decomposition
- `np.linalg.qr(array)`  # QR Decomposition

## 26. Polynomial Operations
- `np.poly1d([1, 2, 3])`  # Create a polynomial
- `np.polyval([1, 2, 3], x)`  # Evaluate a polynomial at specific values

## 27. Set Operations
- `np.intersect1d(array1, array2)`  # Intersection of two arrays
- `np.union1d(array1, array2)`  # Union of two arrays
- `np.setdiff1d(array1, array2)`  # Set difference of two arrays

## 28. Matrix Operations
- `np.matmul(array1, array2)`  # Matrix product of two arrays
- `np.linalg.det(array)`  # Determinant of a matrix

## 29. Creating Meshgrids
- `np.meshgrid(x, y)`  # Create coordinate matrices from coordinate vectors

## 30. Working with Dates
- `np.datetime64('2023-01-01')`  # Create a datetime object
- `np.arange('2023-01', '2023-02', dtype='datetime64[D]')`  # Create an array of dates
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

## Arrays and Properties

### Creating Arrays
- `np.array([1, 2, 3])`  # 1D array
- `np.array([[1, 2], [3, 4]])`  # 2D array
- `np.zeros((2, 3))`  # Array of zeros
- `np.ones((2, 3))`  # Array of ones
- `np.empty((2, 3))`  # Empty array
- `np.arange(0, 10, 1)`  # Array with a range
- `np.linspace(0, 1, 5)`  # Linearly spaced array
- `np.zeros((3, 2))`  # Create an array of zeros
- `np.ones((3, 2))`  # Create an array of ones
- `np.full((3, 2), 7)`  # Create an array filled with a constant value
- `np.eye(3)`  # Create an identity matrix
- `np.linspace(0, 10, 5)`  # Create an array of 5 evenly spaced values between 0 and 10
- `np.arange(0, 10, 2)`  # Create an array with values from 0 to 10 with a step of 2
## 29. Creating Meshgrids
- `np.meshgrid(x, y)`  # Create coordinate matrices from coordinate vectors
- `np.zeros(3, dtype=[('x', 'i4'), ('y', 'f4')])`  # Create a structured array

### Array Properties
- `array.shape`  # Get shape of array
- `array.size`  # Get number of elements
- `array.ndim`  # Get number of dimensions
- `array.dtype`  # Get data type of elements
- `array.T`  # Transpose of the array

### Accessing Elements

#### Indexing

- `array[index]`  # Basic Indexing: Accesses the element at the specified index.
- `array[row_index, col_index]`  # Multi-dimensional Indexing: Accesses the element at the specified row and column indices.
- `array[condition]`  # Boolean Indexing: Selects elements that satisfy the given condition.
- `array[[row_indices], [col_indices]]`  # Fancy Indexing: Accesses elements at the specified row and column indices using arrays of indices.
- `array[start:stop:step]`  # Slicing: Accesses a range of elements from start to stop with the specified step.
- `array[..., index]`  # Ellipsis Indexing: Accesses elements using the ellipsis to represent multiple dimensions.

- `array[start:end:step, start:end:step, ... start:end:step ]` # Access each columns rown in this sequence. All optional and can be negative.


### Datatypes

#### Types
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

#### Operations
- `data_type = ndim_array.dtype` # Check dfatatype
- `array_creation_function( ... , dtype='Datatype Symbol')` # Create funtion of specified type.  Error on incompatable format of data and data type.
- `array.astype(bool)` # Type to convert array to.

#### Working with Dates
- `np.datetime64('2023-01-01')`  # Create a datetime object
- `np.arange('2023-01', '2023-02', dtype='datetime64[D]')`  # Create an array of dates 

### Copying Array
- `x = array.copy()` # COPY: A new copy of old `array` is returned
- `x = arr.view()` # VIEW: A view/reference based on the old `array`
- `status = array.base` # Returns `None` for original array; original array for `VIEW`
- `np.copy(array)`  # Another way to create a copy of an array

### Shape
- `array.shape` # Tuple (nx,ny, ..., nl) representing shape of `array` 
- `array.reshape((n1, n2, ..., nm))`  # Reshape array to m dimension. 
        - Throws error on `incompatable`.
        - The elements required for reshaping are equal in both shapes.
        - Pass `-1` for unknown dimension (only one dimension).
- `np.ravel(array)`  # Flatten array
- `array.flatten()`  # Flatten the array
- `np.transpose(array)`  # Transpose array
- `array.T`  # Transpose of the array

### Broadcasting
- `array + scalar`  # Add scalar to each element
- `array1 + array2`  # Add two arrays element-wise

### Iterating

#### As iterable class
You can iterate over a NumPy array using a simple for loop, treating it as an iterable.

```python
# Iterate using a for loop
for element in ndim_array:
    print(element)
```

#### nditer()
The `np.nditer` function provides an efficient multi-dimensional iterator object to iterate over arrays.

```python
import numpy as np

# Create an example Ndim array
ndim_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# Iterate using np.nditer
with np.nditer(ndim_array, flags=['multi_index'], op_flags=['readwrite']) as it:
    for x in it:
        print(f"Index: {it.multi_index}, Value: {x}")
```

#### Parameters
- **`op`**: The array(s) to iterate over.
- **`flags=[]`**: Flags to control the behavior of the iterator. Common flags include:
  - `'c_index'`: Track a C-order index.
  - `'f_index'`: Track a Fortran-order index.
  - `'multi_index'`: Track a multi-index.
  - `'common_dtype'`: All operands are converted to a common data type.
  - `'copy_if_overlap'`: Create temporary copies to avoid memory overlap.
- **`op_flags=['readwrite']`**: Flags to control the read/write behavior of the operands. Common flags include:
  - `'readonly'`: Read-only operand.
  - `'readwrite'`: Read and write operand.
  - `'writeonly'`: Write-only operand.

#### ndenumerate()
The `np.ndenumerate` function provides an iterator that yields pairs of array coordinates and values.

```python
# Iterate using np.ndenumerate
for index, value in np.ndenumerate(ndim_array):
    print(f"Index: {index}, Value: {value}")
```

#### Parameters
- **`op`**: The array to iterate over.

### Joining Array

Joining NumPy Arrays means putting contents of two or more arrays into a single array. In SQL, we join tables based on a key, whereas in NumPy, we join arrays by axes.

#### Joining Arrays Using `concatenate()`
- `np.concatenate((array1, array2, ...), axis=0)`
- **`axis=0`**: The axis along which the arrays will be joined. Default is 0.

#### Joining Arrays Using `stack()
- `np.hstack((array1, array2, ...))` # Stack arrays along rows
- `np.vstack((array1, array2, ...))` # Stack arrays along columns
- `np.dstack((array1, array2, ...))` # Stack arrays along height (depth)

### Splitting Array

- `np.array_split(array, num_splits)` # Split array into `num_splits` parts. Can handle uneven splits.
- `np.split(array, num_splits)` # Split array into `num_splits` parts. Requires equal division.
- `np.array_split(array, num_splits, axis=0)` # Split 2-D array into `num_splits` parts along specified axis. Default is 0 (rows).
- `np.hsplit(array, num_splits)` # Split array along the horizontal axis (columns).
- `np.vsplit(array, num_splits)` # Split array along the vertical axis (rows).
- `np.dsplit(array, num_splits)` # Split array along the depth axis (third dimension).

### Searching Arrays

- `np.where(condition)` # Return the indices where the condition is True.
  - **condition**: The condition to evaluate.

- `np.searchsorted(array, value)` # Perform a binary search and return the index where `value` should be inserted to maintain order.
  - **array**: The sorted array to search.
  - **value**: The value to insert.

- `np.searchsorted(array, value, side='left')` # Perform a binary search and return the leftmost index where `value` should be inserted.
  - **side='left'**: Default. Search from the left.

- `np.searchsorted(array, value, side='right')` # Perform a binary search and return the rightmost index where `value` should be inserted.
  - **side='right'**: Search from the right.

- `np.searchsorted(array, [value1, value2, ...])` # Perform a binary search for multiple values and return the indices where each value should be inserted.
  - **array**: The sorted array to search.
  - **[value1, value2, ...]**: The values to insert.

### Sorting Arrays

- `np.sort(array)` # Sort the array in ascending order.
  - **array**: The array to be sorted.
  - Returns a sorted copy of the array, leaving the original array unchanged.

- `np.sort(array, axis=-1)` # Sort the array along the specified axis.
  - **axis=-1**: Default. Sort along the last axis.

- `np.sort(array, axis=None)` # Sort the flattened array.
  - **axis=None**: Sort the array as if it is a flattened array.

- `np.sort(array, kind='quicksort')` # Sort the array using the specified algorithm.
  - **kind='quicksort'**: Default. Other options include 'mergesort', 'heapsort', and 'stable'.

- `np.sort(array, order='fieldname')` # Sort a structured array by the specified field.
  - **order='fieldname'**: The field name to sort by.

### Filter Array

- `array[boolean_index_list]` # Filter array using a boolean index list.
  - **boolean_index_list**: A list of booleans corresponding to indexes in the array.

- `array[condition]` # Filter array using a condition directly.
  - **condition**: A condition applied to the array elements.

## Operations

### Array Operations
- `array + 1`  # Add 1 to each element
- `array * 2`  # Multiply each element by 2
- `np.sqrt(array)`  # Square root of each element
- `np.exp(array)`  # Exponential of each element
- `np.log(array)`  # Natural logarithm of each element

### Mathematical Functions
- `np.sin(array)`  # Sine of each element
- `np.cos(array)`  # Cosine of each element
- `np.tan(array)`  # Tangent of each element

### Statistical Functions
- `np.mean(array)`  # Mean of array
- `np.median(array)`  # Median of array
- `np.std(array)`  # Standard deviation
- `np.var(array)`  # Variance
- `np.sum(array)`  # Sum of elements
- `np.min(array)`  # Minimum value
- `np.max(array)`  # Maximum value

### Linear Algebra
- `np.dot(array1, array2)`  # Dot product
- `np.linalg.inv(array)`  # Inverse of matrix
- `np.linalg.eig(array)`  # Eigenvalues and eigenvectors

### Random Number Generation
- `np.random.rand(3, 2)`  # Random array with uniform distribution
- `np.random.randn(3, 2)`  # Random array with normal distribution
- `np.random.randint(0, 10, size=(3, 2))`  # Random integers

### Sorting
- `np.sort(array)`  # Return a sorted copy of an array
- `array.sort()`  # Sort an array in-place
- `np.argsort(array)`  # Indices that would sort an array

### Concatenation
- `np.concatenate((array1, array2), axis=0)`  # Concatenate along the first axis
- `np.concatenate((array1, array2), axis=1)`  # Concatenate along the second axis

### Unique Elements
- `np.unique(array)`  # Find unique elements in an array
- `np.unique(array, return_counts=True)`  # Find unique elements and their counts

### Aggregation Functions
- `np.cumsum(array)`  # Cumulative sum of elements
- `np.cumprod(array)`  # Cumulative product of elements

## 21. Handling NaN Values
- `np.isnan(array)`  # Check for NaN values
- `np.nanmean(array)`  # Mean ignoring NaN values
- `np.nanstd(array)`  # Standard deviation ignoring NaN values

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

## 8. Stacking and Splitting Arrays
- `np.vstack((array1, array2))`  # Vertical stacking
- `np.hstack((array1, array2))`  # Horizontal stacking
- `np.split(array, indices)`  # Split array into sub-arrays


## Load and Store

### Saving and Loading Arrays
- `np.save('array.npy', array)`  # Save array to file
- `loaded_array = np.load('array.npy')`  # Load array from file



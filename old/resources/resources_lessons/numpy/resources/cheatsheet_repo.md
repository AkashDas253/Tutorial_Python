# NumPy Cheatsheet

## 1. Importing NumPy
- `import numpy as np`

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
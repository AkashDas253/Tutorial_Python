## Components of Numpy

### **Core Module**
- `numpy` # Main library for numerical computations.

### **Array Creation**
- `numpy.array` # Create an ndarray from a list or other sequences.
- `numpy.zeros` # Create an array of zeros.
- `numpy.ones` # Create an array of ones.
- `numpy.empty` # Create an uninitialized array.
- `numpy.arange` # Create an array with a range of values.
- `numpy.linspace` # Create evenly spaced values over a specified range.
- `numpy.eye` # Create an identity matrix.

### **Array Manipulation**
- `numpy.reshape` # Change the shape of an array.
- `numpy.ravel` # Flatten an array into 1D.
- `numpy.transpose` # Permute array dimensions.
- `numpy.concatenate` # Join multiple arrays along a specified axis.
- `numpy.stack` # Stack arrays along a new axis.
- `numpy.split` # Split an array into sub-arrays.

### **Mathematical Operations**
- `numpy.add` # Element-wise addition.
- `numpy.subtract` # Element-wise subtraction.
- `numpy.multiply` # Element-wise multiplication.
- `numpy.divide` # Element-wise division.
- `numpy.exp` # Calculate the exponential of all elements.
- `numpy.log` # Natural logarithm of elements.

### **Statistical Operations**
- `numpy.mean` # Compute the arithmetic mean.
- `numpy.median` # Compute the median.
- `numpy.std` # Compute the standard deviation.
- `numpy.var` # Compute the variance.
- `numpy.percentile` # Compute the percentile.

### **Linear Algebra Submodule**
- `numpy.linalg` # Linear algebra functions.
  - `numpy.linalg.inv` # Compute the inverse of a matrix.
  - `numpy.linalg.det` # Compute the determinant of a matrix.
  - `numpy.linalg.eig` # Compute eigenvalues and eigenvectors.
  - `numpy.linalg.svd` # Perform singular value decomposition (SVD).
  - `numpy.linalg.solve` # Solve linear systems.

### **Random Submodule**
- `numpy.random` # Random number generation.
  - `numpy.random.random` # Generate random floats in [0, 1).
  - `numpy.random.randint` # Generate random integers within a range.
  - `numpy.random.normal` # Draw random samples from a normal distribution.
  - `numpy.random.choice` # Randomly choose elements from an array.

### **Fourier Transform Submodule**
- `numpy.fft` # Fourier Transform operations.
  - `numpy.fft.fft` # Compute the one-dimensional discrete Fourier Transform.
  - `numpy.fft.ifft` # Compute the inverse discrete Fourier Transform.

### **Polynomials Submodule**
- `numpy.polynomial` # Polynomial operations.
  - `numpy.polynomial.Polynomial` # Create and manipulate polynomials.
  - `numpy.polynomial.legendre.Legendre` # Legendre polynomial operations.

### **Testing Submodule**
- `numpy.testing` # Test utilities for NumPy.
  - `numpy.testing.assert_array_equal` # Assert that two arrays are equal.

### **Masked Arrays Submodule**
- `numpy.ma` # Masked array operations.
  - `numpy.ma.masked_array` # Create a masked array.
  - `numpy.ma.is_masked` # Check if an array is masked.

### **Data Types Submodule**
- `numpy.dtype` # Define and manipulate data types.
- `numpy.recarray` # Create record arrays.

### **IO Operations**
- `numpy.load` # Load a .npy file.
- `numpy.save` # Save data to a .npy file.
- `numpy.loadtxt` # Load data from a text file.
- `numpy.savetxt` # Save data to a text file.

### **Utilities**
- `numpy.copy` # Create a copy of an array.
- `numpy.view` # Create a new view of the array.
- `numpy.astype` # Cast an array to a different type.

### **Performance Tools**
- `numpy.vectorize` # Vectorize a Python function.
- `numpy.fromfunction` # Construct an array using a function.

### **Distributions Submodule**
- `numpy.random.Generator` # Manage random state and distributions.
  - `numpy.random.Generator.normal` # Normal distribution.
  - `numpy.random.Generator.uniform` # Uniform distribution.

### **FFT and Linear Algebra Helpers**
- `numpy.fft.fftshift` # Shift the zero-frequency component to the center.
- `numpy.linalg.norm` # Compute matrix or vector norms.

This structure reflects both high-level and submodule-specific usage in NumPy.

Here is a comprehensive list of the **NumPy** modules and submodules at the top level with their primary functions:

---

### **`numpy` Modules and Submodules**

- **`numpy`**
  - `numpy.array` # Core function for creating an ndarray (N-dimensional array).
  - `numpy.ndarray` # N-dimensional array object.
  - `numpy.arange` # Function to generate an array with a range of values.
  - `numpy.linspace` # Creates an array of evenly spaced values over a specified range.
  - `numpy.ones` # Creates an array filled with ones.
  - `numpy.zeros` # Creates an array filled with zeros.
  - `numpy.eye` # Creates an identity matrix.
  - `numpy.diag` # Extracts a diagonal or creates a diagonal matrix.
  - `numpy.random` # Random number generation functions (e.g., `rand`, `randn`, `choice`, `shuffle`).
  - `numpy.random.randint` # Generates random integers.
  - `numpy.random.uniform` # Generates random floats in a uniform distribution.
  - `numpy.random.normal` # Generates random samples from a normal distribution.
  - `numpy.random.seed` # Sets the seed for the random number generator.

- **`numpy.linalg`**
  - `numpy.linalg.inv` # Computes the inverse of a matrix.
  - `numpy.linalg.det` # Computes the determinant of a matrix.
  - `numpy.linalg.eig` # Computes eigenvalues and eigenvectors of a matrix.
  - `numpy.linalg.norm` # Computes the norm of a vector or matrix.
  - `numpy.linalg.solve` # Solves linear matrix equations.
  - `numpy.linalg.matrix_rank` # Computes the rank of a matrix.

- **`numpy.fft`**
  - `numpy.fft.fft` # Computes the one-dimensional n-point discrete Fourier Transform (DFT).
  - `numpy.fft.ifft` # Computes the inverse of the one-dimensional n-point DFT.
  - `numpy.fft.fftn` # Computes the N-dimensional DFT.
  - `numpy.fft.ifftn` # Computes the inverse N-dimensional DFT.
  - `numpy.fft.fftfreq` # Computes the sample frequencies for Fourier transform.

- **`numpy.polynomial`**
  - `numpy.polynomial.polynomial` # Polynomial functions such as `polyfit`, `polyval`, and `polyder`.
  - `numpy.polynomial.chebyshev` # Functions for Chebyshev polynomials.
  - `numpy.polynomial.legendre` # Functions for Legendre polynomials.
  - `numpy.polynomial.hermite` # Functions for Hermite polynomials.
  - `numpy.polynomial.hermite_e` # Functions for physicists’ Hermite polynomials.

- **`numpy.matlib`**
  - `numpy.matlib.empty` # Creates an uninitialized matrix.
  - `numpy.matlib.ones` # Creates a matrix filled with ones.
  - `numpy.matlib.zeros` # Creates a matrix filled with zeros.
  - `numpy.matlib.eye` # Creates an identity matrix.
  - `numpy.matlib.rand` # Creates a matrix with random values.

- **`numpy.ma`**
  - `numpy.ma.masked_array` # A class for creating and manipulating arrays with masked elements.
  - `numpy.ma.masked_where` # Masked array where a condition is met.
  - `numpy.ma.masked_equal` # Masked array where elements equal a specified value are masked.
  - `numpy.ma.masked_greater` # Masked array where elements greater than a specified value are masked.
  - `numpy.ma.sum` # Sums masked array elements while ignoring masked elements.
  - `numpy.ma.mean` # Computes the mean while ignoring masked elements.

- **`numpy.lib`**
  - `numpy.lib.index_tricks` # Functions for advanced indexing techniques, like `ix_`, `r_`, `c_`.
  - `numpy.lib.nanfunctions` # Functions for handling NaN values (e.g., `nanmean`, `nanstd`).
  - `numpy.lib.stride_tricks` # Functions for advanced array manipulations with striding.
  - `numpy.lib.scimath` # Mathematical functions for complex numbers (e.g., `scinot` for scientific notation).
  - `numpy.lib.arraysetops` # Functions for set operations on arrays (e.g., `setdiff1d`, `union1d`).
  - `numpy.lib.function_base` # Core functions like `apply_along_axis`, `vectorize`, and `select`.

- **`numpy.core`**
  - `numpy.core.multiarray` # Core functions for manipulating N-dimensional arrays.
  - `numpy.core.records` # Functions for working with structured arrays.
  - `numpy.core.umath` # Universal functions (ufuncs) for element-wise operations.
  - `numpy.core.getlimits` # Provides machine limits (e.g., `finfo`, `iinfo` for float and integer limits).
  - `numpy.core.arrayprint` # Functions for pretty printing arrays.

- **`numpy.testing`**
  - `numpy.testing.assert_array_equal` # Checks if two arrays are equal.
  - `numpy.testing.assert_approx_equal` # Checks if two numbers are approximately equal.
  - `numpy.testing.assert_almost_equal` # Checks if two numbers are almost equal.
  - `numpy.testing.assert_raises` # Asserts that a specified exception is raised.

- **`numpy.ctypeslib`**
  - `numpy.ctypeslib.as_ctypes` # Converts a numpy array into a ctypes object.
  - `numpy.ctypeslib.as_array` # Converts a ctypes object into a numpy array.

- **`numpy.version`**
  - `numpy.version.version` # Displays the version of the NumPy library.
  - `numpy.version.short_version` # Displays a shortened version of the NumPy version.

---

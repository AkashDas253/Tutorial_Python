## NumPy Concepts and Subconcepts  

### **1. NumPy Basics**  
- Importing NumPy (`import numpy as np`)  
- NumPy arrays (`ndarray`)  
- Differences between lists and NumPy arrays  
- NumPy version checking  

### **2. Creating Arrays**  
- **From Lists and Tuples** (`np.array()`)  
- **Using Functions**  
  - `np.zeros()`, `np.ones()`, `np.full()`  
  - `np.arange()`, `np.linspace()`  
  - `np.eye()`, `np.identity()`  
- **Random Arrays** (`np.random`)  
  - `rand()`, `randn()`, `randint()`, `choice()`, `shuffle()`  
- **Uninitialized Arrays** (`np.empty()`)  
- **Structured Arrays** (`np.dtype()`)  

### **3. Array Attributes**  
- Shape (`.shape`)  
- Size (`.size`)  
- Data Type (`.dtype`)  
- Item Size (`.itemsize`)  
- Number of Dimensions (`.ndim`)  

### **4. Indexing and Slicing**  
- **Basic Indexing** (1D, 2D, nD arrays)  
- **Slicing (`:` operator)**  
- **Boolean Indexing**  
- **Fancy Indexing**  
- **Using `np.where()` for conditional indexing**  

### **5. Array Manipulation**  
- **Reshaping (`.reshape()`, `.ravel()`, `.flatten()`)**  
- **Stacking and Splitting**  
  - `np.vstack()`, `np.hstack()`, `np.column_stack()`, `np.row_stack()`  
  - `np.split()`, `np.hsplit()`, `np.vsplit()`  
- **Transposing and Swapping Axes**  
  - `.T`, `np.transpose()`, `np.swapaxes()`  

### **6. Mathematical Operations**  
- **Basic Arithmetic (`+`, `-`, `*`, `/`, `**`, `%`)**  
- **Universal Functions (ufuncs)**  
  - `np.add()`, `np.subtract()`, `np.multiply()`, `np.divide()`  
  - `np.exp()`, `np.log()`, `np.log10()`, `np.sqrt()`  
  - `np.sin()`, `np.cos()`, `np.tan()`, `np.arcsin()`, `np.arccos()`, `np.arctan()`  
- **Aggregation Functions**  
  - `np.sum()`, `np.mean()`, `np.median()`, `np.std()`, `np.var()`  
  - `np.min()`, `np.max()`, `np.argmin()`, `np.argmax()`  
- **Rounding and Clipping**  
  - `np.round()`, `np.floor()`, `np.ceil()`, `np.trunc()`  
  - `np.clip()`  

### **7. Linear Algebra** (`numpy.linalg`)  
- **Matrix Multiplication** (`np.dot()`, `@` operator, `np.matmul()`)  
- **Determinant** (`np.linalg.det()`)  
- **Inverse** (`np.linalg.inv()`)  
- **Eigenvalues and Eigenvectors** (`np.linalg.eig()`)  
- **Singular Value Decomposition (SVD)** (`np.linalg.svd()`)  
- **QR Decomposition** (`np.linalg.qr()`)  
- **Norms** (`np.linalg.norm()`)  
- **Solving Linear Equations** (`np.linalg.solve()`)  

### **8. Random Number Generation (`numpy.random`)**  
- **Generating Random Numbers**  
  - Uniform (`rand()`, `uniform()`)  
  - Normal (`randn()`, `normal()`)  
  - Integer (`randint()`)  
- **Random Choices (`choice()`)**  
- **Shuffling Arrays (`shuffle()`, `permutation()`)**  
- **Seeding Randomness (`seed()`)**  

### **9. Broadcasting**  
- **Scalar and Array Operations**  
- **Broadcasting Rules**  
- **Expanding Dimensions (`np.newaxis`, `np.expand_dims()`)**  

### **10. Sorting and Searching**  
- **Sorting**  
  - `np.sort()`, `np.argsort()`  
- **Searching**  
  - `np.where()`, `np.nonzero()`, `np.argwhere()`, `np.searchsorted()`  

### **11. Structured and Record Arrays**  
- **Custom Data Types (`np.dtype()`)**  
- **Creating Structured Arrays** (`np.array() with dtype`)  
- **Accessing Fields in Structured Arrays**  

### **12. Masked Arrays (`numpy.ma`)**  
- **Creating Masked Arrays (`np.ma.array()`)**  
- **Masking and Unmasking Data (`np.ma.masked_where()`, `np.ma.getmask()`)**  

### **13. Memory and Performance Optimization**  
- **Data Type Selection (`dtype`)**  
- **Memory Views (`np.view()`)**  
- **In-place Operations (`out=` parameter)**  

### **14. Saving and Loading Data**  
- **Binary Files (`.npy`, `.npz`)**  
  - `np.save()`, `np.load()`, `np.savez()`  
- **Text Files (`.txt`, `.csv`)**  
  - `np.savetxt()`, `np.loadtxt()`, `np.genfromtxt()`, `np.recfromcsv()`  

### **15. Advanced NumPy Topics**  
- **Vectorization and Performance** (`np.vectorize()`)  
- **Memory Layout (`C_CONTIGUOUS`, `F_CONTIGUOUS`)**  
- **Multi-threading in NumPy (`np.ufunc.reduce()`)**  
- **NumPy with Pandas (`pd.DataFrame()`)**  

### **16. Compatibility with Other Libraries**  
- **Conversion to/from Pandas** (`pd.DataFrame()`, `.values`)  
- **Interfacing with SciPy (`scipy.sparse`, `scipy.linalg`)**  
- **Working with TensorFlow/PyTorch (`tensor.numpy()`, `np.asarray()`)**


---
---


## NumPy Concepts and Subconcepts  

- **Array Creation**  
  - `array()`, `asarray()`, `zeros()`, `ones()`, `empty()`, `full()`, `arange()`, `linspace()`, `logspace()`, `meshgrid()`, `eye()`, `identity()`  

- **Array Attributes**  
  - `ndim`, `shape`, `size`, `dtype`, `itemsize`, `nbytes`, `T`, `data`  

- **Array Indexing**  
  - Basic indexing (`[]`), Slicing (`:`), Boolean indexing, Fancy indexing (integer arrays)  

- **Array Manipulation**  
  - Reshaping (`reshape()`), Flattening (`ravel()`, `flatten()`), Transposition (`transpose()`, `T`, `swapaxes()`), Concatenation (`concatenate()`, `stack()`, `hstack()`, `vstack()`, `dstack()`), Splitting (`split()`, `hsplit()`, `vsplit()`, `dsplit()`)  

- **Data Types (dtypes)**  
  - `int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`, `float16`, `float32`, `float64`, `complex64`, `complex128`, `bool_`, `object_`, `str_`, `unicode_`  

- **Mathematical Operations**  
  - Element-wise: `add()`, `subtract()`, `multiply()`, `divide()`, `floor_divide()`, `mod()`, `power()`  
  - Rounding: `round()`, `floor()`, `ceil()`, `trunc()`  
  - Exponentials and Logarithms: `exp()`, `exp2()`, `log()`, `log2()`, `log10()`  
  - Trigonometric: `sin()`, `cos()`, `tan()`, `arcsin()`, `arccos()`, `arctan()`, `hypot()`  
  - Hyperbolic: `sinh()`, `cosh()`, `tanh()`, `arcsinh()`, `arccosh()`, `arctanh()`  

- **Statistical Functions**  
  - `min()`, `max()`, `sum()`, `mean()`, `median()`, `std()`, `var()`, `percentile()`, `quantile()`, `cumsum()`, `cumprod()`  

- **Linear Algebra**  
  - Matrix multiplication: `dot()`, `matmul()`  
  - Determinant: `linalg.det()`  
  - Inverse: `linalg.inv()`  
  - Eigenvalues & Eigenvectors: `linalg.eig()`, `linalg.eigh()`  
  - Singular Value Decomposition: `linalg.svd()`  
  - Solving equations: `linalg.solve()`  
  - Norms: `linalg.norm()`  

- **Random Module (`numpy.random`)**  
  - Random numbers: `rand()`, `randn()`, `randint()`, `random()`, `choice()`, `shuffle()`, `permutation()`  
  - Distributions: `normal()`, `uniform()`, `binomial()`, `poisson()`, `gamma()`, `beta()`, `chi2()`, `exponential()`  

- **Broadcasting**  
  - Implicit broadcasting, Expanding dimensions (`newaxis`, `expand_dims()`), `broadcast_to()`  

- **Sorting and Searching**  
  - Sorting: `sort()`, `argsort()`, `lexsort()`  
  - Searching: `where()`, `nonzero()`, `argmax()`, `argmin()`, `searchsorted()`  

- **Set Operations**  
  - `unique()`, `intersect1d()`, `union1d()`, `setdiff1d()`, `setxor1d()`  

- **Bitwise Operations**  
  - `bitwise_and()`, `bitwise_or()`, `bitwise_xor()`, `invert()`, `left_shift()`, `right_shift()`  

- **String Operations (`numpy.char`)**  
  - `char.add()`, `char.multiply()`, `char.center()`, `char.capitalize()`, `char.lower()`, `char.upper()`, `char.strip()`, `char.split()`, `char.replace()`  

- **Structured Arrays**  
  - `dtype()`, `structured array creation`, `view()`, `record arrays`  

- **Memory Management**  
  - `copy()`, `view()`, `base`, `memmap()`  

- **Masked Arrays (`numpy.ma`)**  
  - `masked_array()`, `mask`, `filled()`, `compress()`, `mask_rowcols()`, `masked_where()`  

- **FFT (Fast Fourier Transform) (`numpy.fft`)**  
  - `fft()`, `ifft()`, `rfft()`, `irfft()`, `fftshift()`, `ifftshift()`  

- **Polynomial Module (`numpy.polynomial`)**  
  - `Polynomial()`, `Chebyshev()`, `Legendre()`, `Laguerre()`, `Hermite()`, `hermval()`, `legval()`, `chebval()`  

- **Financial Functions (`numpy.financial`)**  
  - `fv()`, `pv()`, `npv()`, `irr()`, `rate()`, `nper()`, `pmt()`, `ppmt()`, `ipmt()`  

- **I/O Operations**  
  - Saving and loading: `save()`, `load()`, `savez()`, `loadtxt()`, `savetxt()`, `genfromtxt()`, `fromfile()`, `tofile()`  

- **Parallel Computing (`numpy.linalg`, `numpy.ufunc`)**  
  - Vectorization, Universal functions (`ufunc`), Multithreading (`parallelization`)  

- **Compatibility**  
  - `asfarray()`, `astype()`, `copy()`, `view()`, `frombuffer()`, `fromfunction()`, `fromiter()`  


---
---


## Concepts in Numpy

Here is a list of key concepts and sub-concepts in NumPy:

### 1. **Arrays**
   - **ndarray**: Core data structure of NumPy, a multidimensional array.
   - **Array Creation**: 
     - `np.array()`
     - `np.zeros()`
     - `np.ones()`
     - `np.empty()`
     - `np.arange()`
     - `np.linspace()`
     - `np.eye()`
   - **Array Reshaping**:
     - `.reshape()`
     - `.flatten()`
     - `.ravel()`
   - **Array Indexing and Slicing**:
     - Integer and Boolean indexing
     - Fancy indexing
     - Slicing
   - **Broadcasting**: Handling arrays of different shapes during arithmetic operations.

### 2. **Array Operations**
   - **Element-wise Operations**: Addition, subtraction, multiplication, etc.
   - **Universal Functions (ufuncs)**:
     - `np.add()`, `np.multiply()`, etc.
     - Trigonometric functions (`np.sin()`, `np.cos()`)
     - Exponential, logarithmic functions (`np.exp()`, `np.log()`)
   - **Mathematical Functions**: 
     - `np.sum()`, `np.prod()`
     - `np.mean()`, `np.median()`, `np.std()`
   - **Linear Algebra**:
     - Dot product: `np.dot()`
     - Matrix multiplication: `np.matmul()`
     - Determinant, inverse: `np.linalg.det()`, `np.linalg.inv()`
   - **Statistics**:
     - `np.corrcoef()`, `np.var()`, `np.percentile()`

### 3. **Array Manipulation**
   - **Concatenation**:
     - `np.concatenate()`
     - `np.stack()`
     - `np.hstack()`, `np.vstack()`
   - **Splitting**:
     - `np.split()`
     - `np.hsplit()`, `np.vsplit()`
   - **Sorting**:
     - `np.sort()`, `np.argsort()`
   - **Unique Elements**:
     - `np.unique()`

### 4. **Random Module**
   - **Random Sampling**: 
     - `np.random.random()`
     - `np.random.randn()`
     - `np.random.randint()`
   - **Random Permutation**:
     - `np.random.shuffle()`
     - `np.random.permutation()`
   - **Distributions**: 
     - `np.random.normal()`
     - `np.random.binomial()`

### 5. **File Input/Output**
   - **Loading/Saving**:
     - `np.load()`, `np.save()`
     - `np.loadtxt()`, `np.savetxt()`
   - **Binary Format**: 
     - `np.savez()`, `np.loadz()`

### 6. **Linear Algebra**
   - **Eigenvalues and Eigenvectors**:
     - `np.linalg.eig()`
   - **Singular Value Decomposition (SVD)**:
     - `np.linalg.svd()`
   - **Solving Linear Systems**:
     - `np.linalg.solve()`
   - **Matrix Decomposition**:
     - `np.linalg.cholesky()`

### 7. **Advanced Concepts**
   - **Masked Arrays**:
     - `np.ma.masked_array()`
   - **Structured Arrays**:
     - `np.dtype()`, `np.recarray()`
   - **Memory Layout**:
     - `np.copy()`, `.flags`, `.strides`
   - **Array Views**: Understanding views vs. copies (`np.view()`, `.copy()`)

### 8. **Performance and Optimization**
   - **Vectorization**: Using ufuncs and broadcasting for performance.
   - **NumPy C-API**: For creating custom extensions and optimizing performance.
   - **Memory Management**: `np.memmap` for memory-mapped arrays.

### 9. **Compatibility**
   - **Type Casting**:
     - `np.astype()`
   - **Converting to/from Pandas**: Using `pandas.DataFrame` and `pandas.Series` with NumPy.
   - **Interoperability with Other Libraries**: e.g., TensorFlow, PyTorch

Each of these categories covers a wide array of methods and functions to efficiently handle and manipulate numerical data.
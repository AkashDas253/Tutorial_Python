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
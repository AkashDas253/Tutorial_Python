## **Linear Algebra in NumPy**  

NumPy provides a comprehensive set of functions for performing linear algebra operations, including matrix operations, decompositions, and solving linear equations.

---

### **Matrix Operations**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `dot()` | Computes the dot product of two arrays. | `result = np.dot(A, B)` |
| `matmul()` | Computes matrix multiplication. | `result = np.matmul(A, B)` |
| `T` | Returns the transpose of a matrix. | `transposed = A.T` |
| `trace()` | Computes the sum of diagonal elements. | `tr = np.trace(A)` |

---

### **Determinant and Rank**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `linalg.det()` | Computes the determinant of a matrix. | `det_value = np.linalg.det(A)` |
| `linalg.matrix_rank()` | Computes the rank of a matrix. | `rank = np.linalg.matrix_rank(A)` |

---

### **Inverse and Pseudo-Inverse**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `linalg.inv()` | Computes the inverse of a square matrix. | `inverse = np.linalg.inv(A)` |
| `linalg.pinv()` | Computes the Moore-Penrose pseudo-inverse. | `pinv = np.linalg.pinv(A)` |

---

### **Eigenvalues and Eigenvectors**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `linalg.eig()` | Computes eigenvalues and eigenvectors. | `eig_vals, eig_vecs = np.linalg.eig(A)` |
| `linalg.eigh()` | Computes eigenvalues and eigenvectors of a symmetric matrix. | `eig_vals, eig_vecs = np.linalg.eigh(A)` |

---

### **Singular Value Decomposition (SVD)**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `linalg.svd()` | Computes the Singular Value Decomposition. | `U, S, V = np.linalg.svd(A)` |

---

### **Solving Linear Systems**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `linalg.solve()` | Solves a system of linear equations. | `x = np.linalg.solve(A, b)` |
| `linalg.lstsq()` | Computes the least squares solution to Ax = b. | `x, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)` |

---

### **Norms**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `linalg.norm()` | Computes matrix or vector norm. | `norm_value = np.linalg.norm(A, ord)` |

---

### **Summary**  
- **Matrix operations**: `dot()`, `matmul()`, `T`, `trace()`.  
- **Determinant and rank**: `det()`, `matrix_rank()`.  
- **Inverse and pseudo-inverse**: `inv()`, `pinv()`.  
- **Eigenvalues & eigenvectors**: `eig()`, `eigh()`.  
- **SVD**: `svd()`.  
- **Solving equations**: `solve()`, `lstsq()`.  
- **Norms**: `norm()`.
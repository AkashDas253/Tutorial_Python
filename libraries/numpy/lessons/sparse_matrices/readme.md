## **Sparse Matrices (SciPy Integration)**  

---

### **Overview**  
- Sparse matrices store only nonzero values, reducing memory usage.  
- SciPy's `scipy.sparse` module provides efficient sparse matrix operations.  
- Supports various sparse formats like **CSR**, **CSC**, and **DOK** for different use cases.  

---

### **1. Types of Sparse Matrices**  

| **Format** | **Description** | **Best for** |
|-----------|---------------|-------------|
| **CSR (Compressed Sparse Row)** | Stores row indices, column indices, and nonzero values efficiently. | Fast row slicing and matrix-vector products. |
| **CSC (Compressed Sparse Column)** | Similar to CSR but optimized for column operations. | Fast column slicing and matrix-vector products. |
| **DOK (Dictionary of Keys)** | Uses a dictionary to store row-column-value mappings. | Efficient incremental construction. |
| **LIL (List of Lists)** | Stores lists of column indices per row. | Efficient incremental modifications. |
| **COO (Coordinate List)** | Stores `(row, col, value)` tuples. | Easy conversion between formats. |

---

### **2. Creating Sparse Matrices**  

#### **CSR Matrix**  
**Syntax:**  
```python
from scipy.sparse import csr_matrix
import numpy as np

data = np.array([1, 2, 3])
row_indices = np.array([0, 1, 2])
col_indices = np.array([0, 1, 2])

sparse_matrix = csr_matrix((data, (row_indices, col_indices)), shape=(3, 3))
```

#### **COO Matrix**  
**Syntax:**  
```python
from scipy.sparse import coo_matrix

data = np.array([4, 5, 6])
row_indices = np.array([0, 1, 2])
col_indices = np.array([1, 2, 0])

coo_matrix = coo_matrix((data, (row_indices, col_indices)), shape=(3, 3))
```

---

### **3. Converting Between Sparse Formats**  
- **CSR to CSC:**  
  ```python
  csc_matrix = sparse_matrix.tocsc()
  ```
- **Any Sparse Format to Dense:**  
  ```python
  dense_array = sparse_matrix.toarray()
  ```
- **Dense to Sparse:**  
  ```python
  sparse_matrix = csr_matrix(dense_array)
  ```

---

### **4. Sparse Matrix Operations**  

| **Operation** | **Method** |
|--------------|-----------|
| **Matrix Addition** | `A + B` |
| **Matrix Multiplication** | `A.dot(B)` |
| **Transpose** | `A.T` |
| **Element-wise Multiplication** | `A.multiply(B)` |

**Syntax:**  
```python
result = sparse_matrix.dot(sparse_matrix.T)  # Matrix multiplication
```

---

### **5. Memory Efficiency**  
- **Dense matrix storage:** `O(n²)`  
- **Sparse matrix storage:** `O(nnz)` (where `nnz` is the number of nonzero elements)  
- **Compression saves memory**, especially for large sparse datasets.  

---

### **Summary**  
- **Sparse matrices store only nonzero values**, reducing memory usage.  
- **Different formats (CSR, CSC, COO, DOK) optimize operations** based on needs.  
- **Efficient operations available using SciPy’s `scipy.sparse` module**.
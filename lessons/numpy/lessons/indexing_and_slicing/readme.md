## **Indexing and Slicing in NumPy**

NumPy provides powerful indexing and slicing mechanisms to access and manipulate array elements efficiently.

---

### **Indexing in NumPy**  

| Type | Description | Syntax |
|------|------------|--------|
| **Basic Indexing** | Access elements using zero-based indices. | `arr[index]` |
| **Negative Indexing** | Access elements from the end. | `arr[-index]` |
| **Multidimensional Indexing** | Access elements in multi-dimensional arrays. | `arr[row, col]` |
| **Boolean Indexing** | Select elements that satisfy a condition. | `arr[condition]` |

---

### **Slicing in NumPy**  

Slicing extracts portions of an array using the `start:stop:step` notation.

| Type | Description | Syntax |
|------|------------|--------|
| **Basic Slicing** | Extracts a portion of the array. | `arr[start:stop]` |
| **Slicing with Step** | Skips elements based on step size. | `arr[start:stop:step]` |
| **Omitting Indices** | Defaults to full range if `start` or `stop` is omitted. | `arr[:stop]`, `arr[start:]`, `arr[:]` |
| **Multidimensional Slicing** | Extracts subarrays from multi-dimensional arrays. | `arr[start:stop, start:stop]` |

---

### **Advanced Indexing**  

| Type | Description | Syntax |
|------|------------|--------|
| **Integer Indexing** | Selects elements using a list of indices. | `arr[[i, j, k]]` |
| **Boolean Masking** | Filters elements based on conditions. | `arr[arr > value]` |

---

### **Summary**  
- **Indexing** is used to access specific elements.  
- **Slicing** extracts subarrays with `start:stop:step`.  
- **Advanced indexing** allows complex selections using integer lists or Boolean masks.
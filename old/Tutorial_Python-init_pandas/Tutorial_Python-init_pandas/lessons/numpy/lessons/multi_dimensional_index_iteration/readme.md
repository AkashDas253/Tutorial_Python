## **Multi-dimensional Indexing and Iteration in NumPy**  

---

### **Multi-dimensional Indexing**  
NumPy provides efficient ways to access elements in **multi-dimensional arrays** using different indexing techniques.

### **1. Basic Indexing**  
- Uses `[row, column]` notation for 2D arrays.  
- Supports higher dimensions: `[dim1, dim2, dim3, ...]`.  

**Syntax:**  
```python
arr = np.array([[[1, 2], [3, 4]],  
                [[5, 6], [7, 8]]])  

element = arr[1, 0, 1]  # Output: 6  
```
- **Index `[1, 0, 1]` refers to:**  
  - **First index (`1`)** → Second block  
  - **Second index (`0`)** → First row  
  - **Third index (`1`)** → Second column  

---

### **2. Multi-dimensional Slicing**  
- Uses `:` for range selection in multiple dimensions.  

**Syntax:**  
```python
subarray = arr[:, 0, :]  # Extracts first row from each block  
```

| **Indexing Type** | **Description** |
|------------------|---------------|
| `arr[1, :, :]` | All rows, all columns from the second block |
| `arr[:, :, 1]` | Second column from all blocks |
| `arr[0, 1:, :]` | From second row onward in the first block |

---

### **3. Fancy Indexing**  
- Uses lists/arrays to select multiple elements at once.  

**Syntax:**  
```python
rows = [0, 1]  
cols = [1, 0]  
result = arr[rows, cols]  
```
- Selects **specific row-column pairs**.  

---

### **4. Boolean Indexing**  
- Selects elements based on conditions.  

**Syntax:**  
```python
mask = arr > 5  
result = arr[mask]  # Output: [6, 7, 8]  
```

---

### **5. `np.ix_()` for Advanced Indexing**  
- Creates index grids for multi-dimensional selection.  

**Syntax:**  
```python
rows = np.array([0, 1])  
cols = np.array([1])  
result = arr[np.ix_(rows, cols)]  
```

---

### **Multi-dimensional Iteration**  
- NumPy provides different iteration techniques for efficient traversal.

### **1. Using `nditer` for Element-wise Iteration**  
- Works efficiently for multi-dimensional arrays.  

**Syntax:**  
```python
for val in np.nditer(arr):  
    print(val)  # Iterates over all elements  
```

---

### **2. Iterating with Axis Control**  
- Allows specific axis-based iteration.  

**Syntax:**  
```python
for row in arr:  
    print(row)  # Iterates over the first axis  
```

---

### **3. Iterating with `ndenumerate` for Index Tracking**  
- Retrieves both **index and value**.  

**Syntax:**  
```python
for idx, val in np.ndenumerate(arr):  
    print(idx, val)  # Outputs index and value  
```

---

### **4. Broadcasting for Efficient Iteration**  
- Avoids explicit loops for element-wise operations.  

**Syntax:**  
```python
result = arr * 2  # Element-wise multiplication without explicit iteration  
```

---

### **Summary**  
- **Multi-dimensional Indexing:** Supports **basic, fancy, slicing, and boolean indexing**.  
- **`np.ix_()` enables complex selections** across multiple dimensions.  
- **Multi-dimensional Iteration:** Uses `nditer`, `ndenumerate`, and axis-based iteration for efficiency.  
- **Broadcasting avoids explicit iteration**, improving performance.
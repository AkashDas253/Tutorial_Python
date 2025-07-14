## **Multi-dimensional Indexing in NumPy**  

---

### **Overview**  
- NumPy allows **indexing into multi-dimensional arrays** using various techniques.  
- Supports **basic indexing, slicing, fancy indexing, and boolean masking**.  
- Enables efficient extraction and modification of elements in **n-dimensional arrays**.  

---

### **1. Basic Multi-dimensional Indexing**  
- Uses row and column indices to access elements.  
- Follows **zero-based indexing**.  

**Syntax:**  
```python
arr = np.array([[1, 2, 3],  
                [4, 5, 6],  
                [7, 8, 9]])  

element = arr[1, 2]  # Access row index 1, column index 2 → Output: 6  
```

---

### **2. Multi-dimensional Slicing**  
- Uses `:` to extract subarrays.  

**Syntax:**  
```python
subarray = arr[0:2, 1:3]  # Rows 0 to 1, Columns 1 to 2  
```

| **Indexing Type** | **Description** |
|------------------|---------------|
| `arr[1:, :2]` | From row 1 to end, first two columns |
| `arr[:2, 1:]` | First two rows, from column 1 to end |
| `arr[:, 1]` | All rows, column 1 |

---

### **3. Fancy Indexing**  
- Uses **lists or arrays** for non-contiguous indexing.  

**Syntax:**  
```python
rows = [0, 2]  
cols = [1, 2]  
result = arr[rows, cols]  # Selects (0,1) and (2,2) → Output: [2, 9]  
```

---

### **4. Boolean Indexing**  
- Selects elements based on conditions.  

**Syntax:**  
```python
mask = arr > 5  
result = arr[mask]  # Output: [6, 7, 8, 9]  
```

---

### **5. Advanced Indexing with `np.ix_`**  
- Expands indexing for multiple dimensions.  

**Syntax:**  
```python
rows = np.array([0, 2])  
cols = np.array([1, 2])  
result = arr[np.ix_(rows, cols)]  
```
- **Generates a subarray using row and column selections**.  

---

### **Summary**  
- **Basic Indexing:** Uses `[row, col]` notation.  
- **Slicing:** Extracts subarrays with `:` notation.  
- **Fancy Indexing:** Selects specific indices using lists or arrays.  
- **Boolean Indexing:** Extracts elements based on conditions.  
- **`np.ix_` Method:** Used for advanced multi-dimensional indexing.
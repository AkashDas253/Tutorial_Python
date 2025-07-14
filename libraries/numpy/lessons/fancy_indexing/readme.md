## **Fancy Indexing**  

---

### **Overview**  
- Fancy indexing allows selecting elements from a NumPy array using **integer arrays**, **boolean masks**, or **index arrays**.  
- It provides more flexibility than basic slicing and indexing.  

---

### **1. Indexing with Integer Arrays**  
- Use a list or array of integers to select specific elements.  

**Syntax:**  
```python
arr = np.array([10, 20, 30, 40, 50])
result = arr[[0, 2, 4]]  # Selects elements at indices 0, 2, and 4
```

- Works for multi-dimensional arrays.  

**Syntax:**  
```python
arr = np.array([[1, 2], [3, 4], [5, 6]])
result = arr[[0, 2], [1, 0]]  # Selects (0,1) and (2,0)
```

---

### **2. Boolean Mask Indexing**  
- Selects elements that satisfy a condition.  

**Syntax:**  
```python
arr = np.array([10, 20, 30, 40, 50])
mask = arr > 25  # Boolean condition
result = arr[mask]  # Selects elements where condition is True
```

---

### **3. Combining Fancy and Slice Indexing**  
- Can mix fancy indexing with slicing.  

**Syntax:**  
```python
arr = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
result = arr[1:, [0, 2]]  # Selects from rows 1 onward and columns 0, 2
```

---

### **4. Modifying Arrays with Fancy Indexing**  
- Allows assignment of new values.  

**Syntax:**  
```python
arr = np.array([10, 20, 30, 40])
arr[[1, 3]] = [99, 88]  # Updates indices 1 and 3
```

---

### **5. Avoiding Unintended Copies**  
- Fancy indexing creates a new array, unlike slicing, which creates a view.  
- Modifications on the new array do not affect the original.  

**Syntax:**  
```python
arr = np.array([10, 20, 30])
copy_arr = arr[[0, 2]]  # Creates a new copy
copy_arr[0] = 99  # Does not modify 'arr'
```

---

### **Summary**  
- **Fancy indexing enables advanced selection** using integer arrays and boolean masks.  
- **Supports multidimensional selection** and combination with slicing.  
- **Creates a copy instead of a view**, unlike slicing.
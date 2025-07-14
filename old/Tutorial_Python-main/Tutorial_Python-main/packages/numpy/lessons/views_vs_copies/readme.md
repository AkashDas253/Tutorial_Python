## **Views vs. Copies in NumPy**  

---

### **1. Understanding Views and Copies**  
- **View** → A new array object referencing the same memory as the original array.  
- **Copy** → A new array object with its own separate memory allocation.  

| **Aspect** | **View** | **Copy** |
|------------|---------|---------|
| **Memory** | Shares memory with the original array | Has separate memory allocation |
| **Performance** | Faster (no data duplication) | Slower (allocates new memory) |
| **Modifications** | Affects the original array | Does not affect the original array |
| **Use Case** | When memory efficiency is needed | When data integrity is required |

---

### **2. Creating Views in NumPy**  
- Views are created through slicing, `reshape()`, or `view()`.  
- Modifications in a view affect the original array.  

**Syntax:**  
```python
import numpy as np  

arr = np.array([1, 2, 3, 4])  
view_arr = arr[1:3]  # Creates a view  
view_arr[0] = 99  # Modifies both view and original  
```

---

### **3. Creating Copies in NumPy**  
- Copies create a separate array that does not affect the original.  
- Use `copy()` to ensure independence.  

**Syntax:**  
```python
copy_arr = arr.copy()  # Creates a copy  
copy_arr[0] = 100  # Only modifies copy, not original  
```

---

### **4. Checking If an Array is a View or Copy**  
- Use `np.may_share_memory()` to check memory sharing.  

**Syntax:**  
```python
np.may_share_memory(arr, view_arr)  # True (View)  
np.may_share_memory(arr, copy_arr)  # False (Copy)  
```

---

### **5. Summary**  
- **Use Views** for performance optimization and avoiding redundant memory usage.  
- **Use Copies** when modifications should not affect the original data.  
- **Always check memory sharing** when unsure of data independence.
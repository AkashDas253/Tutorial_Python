## **Masked Arrays in NumPy**  

Masked arrays allow handling missing or invalid data in NumPy by masking specific elements. This is useful for computations that should ignore certain values.

---

### **Creating Masked Arrays**  

| Method | Description | Syntax |
|--------|-------------|--------|
| `masked_array()` | Creates a masked array from data and mask. | `masked_arr = np.ma.masked_array(data, mask=condition)` |
| `masked_where()` | Masks elements where a condition is `True`. | `masked_arr = np.ma.masked_where(condition, data)` |
| `masked_equal()` | Masks elements equal to a specific value. | `masked_arr = np.ma.masked_equal(data, value)` |
| `masked_greater()` | Masks elements greater than a threshold. | `masked_arr = np.ma.masked_greater(data, value)` |

---

### **Accessing and Modifying Masks**  

| Operation | Description | Syntax |
|------------|-------------|--------|
| Check mask | View the mask of a masked array. | `masked_arr.mask` |
| Modify mask | Change the mask of an array. | `masked_arr.mask = new_mask` |
| Fill masked values | Replace masked values with a given number. | `filled_arr = masked_arr.filled(value)` |

---

### **Operations on Masked Arrays**  

| Operation | Description | Syntax |
|------------|-------------|--------|
| Apply arithmetic | Operations ignore masked elements. | `result = masked_arr + 10` |
| Aggregate functions | Functions like `sum()`, `mean()` ignore masks. | `masked_arr.sum()` |
| Compress | Remove masked elements. | `compressed = masked_arr.compressed()` |

---

### **Summary**  
- **Masked arrays** handle missing/invalid values by applying masks.  
- **Creation methods** include `masked_array()`, `masked_where()`, `masked_equal()`, etc.  
- **Masks can be accessed, modified, or filled** with new values.  
- **Operations ignore masked elements**, ensuring correct computations.
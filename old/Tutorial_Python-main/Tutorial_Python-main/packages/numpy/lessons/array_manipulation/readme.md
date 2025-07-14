## **Array Manipulation in NumPy**

NumPy provides various functions to modify and transform arrays, including reshaping, joining, splitting, and modifying elements.

---

### **Reshaping Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `reshape()` | Changes the shape of an array without modifying data. | `new_arr = arr.reshape(new_shape)` |
| `ravel()` | Flattens the array into a 1D array (returns a view). | `flat_arr = arr.ravel()` |
| `flatten()` | Flattens the array into a 1D array (returns a copy). | `flat_arr = arr.flatten()` |
| `transpose()` | Transposes the array (rows become columns). | `transposed = arr.transpose()` |
| `T` | Short notation for transposing. | `transposed = arr.T` |

---

### **Joining Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `concatenate()` | Joins multiple arrays along an axis. | `joined = np.concatenate((arr1, arr2), axis=0)` |
| `stack()` | Stacks arrays along a new axis. | `stacked = np.stack((arr1, arr2), axis=0)` |
| `hstack()` | Stacks arrays horizontally. | `h_stacked = np.hstack((arr1, arr2))` |
| `vstack()` | Stacks arrays vertically. | `v_stacked = np.vstack((arr1, arr2))` |
| `dstack()` | Stacks arrays depth-wise. | `d_stacked = np.dstack((arr1, arr2))` |

---

### **Splitting Arrays**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `split()` | Splits an array into multiple subarrays. | `subarrays = np.split(arr, sections, axis=0)` |
| `hsplit()` | Splits an array horizontally. | `subarrays = np.hsplit(arr, sections)` |
| `vsplit()` | Splits an array vertically. | `subarrays = np.vsplit(arr, sections)` |
| `dsplit()` | Splits an array depth-wise. | `subarrays = np.dsplit(arr, sections)` |

---

### **Modifying Elements**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `append()` | Appends values to an array. | `new_arr = np.append(arr, values, axis)` |
| `insert()` | Inserts values at a given index. | `new_arr = np.insert(arr, index, values, axis)` |
| `delete()` | Deletes elements at a given index. | `new_arr = np.delete(arr, index, axis)` |
| `unique()` | Finds unique elements in an array. | `unique_values = np.unique(arr)` |

---

### **Summary**  
- **Reshaping**: `reshape()`, `ravel()`, `flatten()`, `transpose()`.  
- **Joining**: `concatenate()`, `stack()`, `hstack()`, `vstack()`, `dstack()`.  
- **Splitting**: `split()`, `hsplit()`, `vsplit()`, `dsplit()`.  
- **Modifying**: `append()`, `insert()`, `delete()`, `unique()`.
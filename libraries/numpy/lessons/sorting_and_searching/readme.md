## **Sorting and Searching in NumPy**  

NumPy provides efficient functions for sorting and searching operations, which allow for quick manipulation and retrieval of array elements.

---

### **Sorting Functions**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `sort()` | Sorts an array along a specified axis. | `sorted_array = np.sort(arr, axis)` |
| `argsort()` | Returns the indices that would sort an array. | `indices = np.argsort(arr, axis)` |
| `lexsort()` | Performs indirect sorting using multiple keys. | `indices = np.lexsort((key1, key2, ...))` |
| `partition()` | Partially sorts an array, placing the kth element in its correct position. | `part_sorted = np.partition(arr, kth)` |
| `argpartition()` | Returns indices that would partition an array. | `indices = np.argpartition(arr, kth)` |

---

### **Searching Functions**  

| Function | Description | Syntax |
|----------|-------------|--------|
| `where()` | Returns indices of elements satisfying a condition. | `indices = np.where(condition)` |
| `searchsorted()` | Finds indices where elements should be inserted to maintain order. | `indices = np.searchsorted(arr, values, side)` |
| `nonzero()` | Returns indices of nonzero elements. | `indices = np.nonzero(arr)` |
| `argmax()` | Returns the index of the maximum value along an axis. | `index = np.argmax(arr, axis)` |
| `argmin()` | Returns the index of the minimum value along an axis. | `index = np.argmin(arr, axis)` |

---

### **Sorting Order**  
- **Ascending order (default)**: `np.sort(arr)`  
- **Descending order**: `np.sort(arr)[::-1]`  
- **Sorting along a specific axis**: `np.sort(arr, axis=0 or 1)`

---

### **Summary**  
- **Sorting**: `sort()`, `argsort()`, `lexsort()`, `partition()`, `argpartition()`.  
- **Searching**: `where()`, `searchsorted()`, `nonzero()`, `argmax()`, `argmin()`.  
- **Sorting order can be controlled using axis or reversing the result.**
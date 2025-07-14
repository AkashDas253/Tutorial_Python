## **Boolean Masking and Filtering in NumPy**  

### **Overview**  
Boolean masking and filtering in NumPy allow selective access to elements in an array based on conditions, enabling efficient data manipulation.

### **Boolean Masking**  
A Boolean mask is an array of Boolean values (`True` or `False`) that determines which elements of another array should be selected.

#### **Creating a Boolean Mask**  
```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50])
mask = arr > 25  # Create a Boolean mask
print(mask)  
# Output: [False False  True  True  True]
```

#### **Applying a Boolean Mask**  
```python
filtered_arr = arr[mask]  # Apply mask to filter values
print(filtered_arr)  
# Output: [30 40 50]
```

### **Filtering Using Conditions**  
#### **Single Condition**  
```python
arr = np.array([5, 15, 25, 35, 45])
filtered = arr[arr > 20]  # Select values greater than 20
print(filtered)  
# Output: [25 35 45]
```

#### **Multiple Conditions (Logical Operators)**  
Use `&` (AND), `|` (OR), and `~` (NOT) for complex filtering.

```python
arr = np.array([5, 15, 25, 35, 45])
filtered = arr[(arr > 10) & (arr < 40)]  # Select values between 10 and 40
print(filtered)  
# Output: [15 25 35]
```

#### **Filtering Using `np.where()`**  
`np.where(condition, x, y)` returns values from `x` where the condition is `True`, otherwise from `y`.  

```python
arr = np.array([10, 20, 30, 40])
result = np.where(arr > 25, arr, -1)  # Replace values ≤ 25 with -1
print(result)  
# Output: [-1 -1 30 40]
```

### **Filtering NaN (Not a Number) Values**  
Use `np.isnan()` to detect NaN values.

```python
arr = np.array([1.0, np.nan, 2.5, np.nan, 3.0])
filtered = arr[~np.isnan(arr)]  # Remove NaN values
print(filtered)  
# Output: [1.0 2.5 3.0]
```

### **Masking with `np.ma` (Masked Arrays)**  
Masked arrays allow marking specific elements as invalid.

```python
arr = np.array([10, -1, 20, -1, 30])
masked_arr = np.ma.masked_equal(arr, -1)  # Mask values equal to -1
print(masked_arr)  
# Output: [10 -- 20 -- 30]
```

### **Performance Considerations**  
- Boolean masking is vectorized and faster than loops.
- Using `np.where()` can be more memory-efficient than direct Boolean indexing.

### **Conclusion**  
Boolean masking and filtering enable efficient data selection and manipulation, crucial for large datasets and numerical computations.
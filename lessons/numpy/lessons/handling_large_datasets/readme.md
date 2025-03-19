## **Handling Large Datasets in NumPy**  

---

### **Overview**  
- NumPy provides techniques to efficiently work with large datasets.  
- Focuses on **memory optimization**, **efficient computations**, and **incremental processing**.  
- Uses methods like **memory mapping, chunking, broadcasting, and lazy evaluation** to optimize performance.  

---

### **1. Memory Mapping for Large Files**  
- Loads large datasets **without loading the entire file into memory**.  
- Uses `np.memmap` to create a memory-mapped array.  

**Syntax:**  
```python
memmap_array = np.memmap('large_data.dat', dtype='float32', mode='r', shape=(10000, 10000))
```
- **Mode options:**  
  - `'r'` → Read-only  
  - `'r+'` → Read and write  
  - `'w+'` → Create or overwrite  

---

### **2. Using Generators for Incremental Processing**  
- Avoids storing the entire dataset in memory.  

**Syntax:**  
```python
def data_generator(filename, chunk_size=1000):
    for chunk in np.array_split(np.loadtxt(filename), chunk_size):
        yield chunk
```
- Processes data **in chunks**, reducing memory usage.  

---

### **3. Chunking Large Datasets**  
- Reads data **in smaller parts** instead of loading everything at once.  
- Works well with Pandas and NumPy.  

**Syntax:**  
```python
import pandas as pd

chunk_iterator = pd.read_csv('large_file.csv', chunksize=10000)
for chunk in chunk_iterator:
    processed = np.array(chunk)  # Process each chunk
```

---

### **4. Efficient Computations with NumPy Broadcasting**  
- Avoids explicit loops, reducing memory overhead.  

**Syntax:**  
```python
arr = np.random.rand(10000, 10000)
result = arr * 5  # Uses broadcasting instead of looping
```
- **Optimized operations reduce execution time**.  

---

### **5. Optimized Data Types**  
- Use **lower precision** data types to save memory.  

| **Data Type** | **Memory Usage** |
|--------------|----------------|
| `float64` | 8 bytes per element |
| `float32` | 4 bytes per element |
| `int32` | 4 bytes per element |
| `int16` | 2 bytes per element |

**Syntax:**  
```python
arr = np.array([1.2, 2.3, 3.4], dtype=np.float32)  # Saves memory
```

---

### **6. Parallel Processing with NumPy**  
- Uses `numexpr` for faster computations.  

**Syntax:**  
```python
import numexpr as ne

arr1 = np.random.rand(1000000)
arr2 = np.random.rand(1000000)

result = ne.evaluate("arr1 + arr2")  # Faster than np.add
```
- **NumExpr optimizes memory and CPU usage**.  

---

### **7. Lazy Evaluation with Dask**  
- Dask handles large datasets in parallel.  

**Syntax:**  
```python
import dask.array as da

large_array = da.from_array(np.random.rand(1000000), chunks=10000)
result = large_array.mean().compute()  # Computes result lazily
```
- **Works like NumPy but processes data in chunks**.  

---

### **Summary**  
- **Memory mapping (`np.memmap`)** avoids loading full datasets into memory.  
- **Generators and chunking** process data incrementally.  
- **Broadcasting optimizes computations** by reducing loops.  
- **Using lower precision data types** saves memory.  
- **Parallel computation with NumExpr and Dask** improves efficiency.
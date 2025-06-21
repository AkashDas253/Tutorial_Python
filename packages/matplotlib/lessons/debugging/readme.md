# **Debugging in Matplotlib**  

Matplotlib debugging involves identifying and fixing issues related to **syntax errors, logical errors, rendering issues, performance bottlenecks, and compatibility problems**.

---

## **1. Common Errors and Debugging Strategies**  

| **Error Type**        | **Cause**                                          | **Solution**                                  |
|----------------------|--------------------------------------------------|----------------------------------------------|
| `ModuleNotFoundError` | Matplotlib is not installed.                      | `pip install matplotlib`                   |
| `TypeError`          | Incorrect function argument type.                  | Check function signature and data types.   |
| `ValueError`         | Invalid input values for plotting.                 | Validate input data and use `print()` to debug. |
| `AttributeError`     | Using a non-existent attribute or method.          | Check object type and available methods using `dir(obj)`. |
| `OverflowError`      | Exceeding numerical limits (e.g., `inf`).         | Use normalized values and log scaling.    |
| `MemoryError`        | Large data causing excessive memory usage.         | Downsample data and use efficient plotting techniques. |
| `ImportError`        | Version conflicts between dependencies.            | Reinstall Matplotlib and dependencies: `pip install --upgrade matplotlib`. |

---

## **2. Debugging Rendering Issues**  

### **2.1 Figure Not Showing (`plt.show()` Missing)**  
```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()  # Ensure this is called
```
- **Fix**: Always call `plt.show()` for displaying figures.

---

### **2.2 Empty or Incorrect Plot**  
#### **Possible Causes & Fixes**  
| Issue                         | Fix                                   |
|--------------------------------|--------------------------------------|
| Data is not properly passed.   | Ensure valid data is passed to `plot()`. |
| `plt.show()` is missing.       | Add `plt.show()` after plotting.  |
| Wrong axis limits.             | Use `plt.xlim()` and `plt.ylim()`. |
| Conflicting plots.             | Call `plt.clf()` before new plots. |

```python
plt.plot([], [])  # Incorrect: Empty lists
plt.show()
```
✅ **Fix**  
```python
plt.plot([1, 2, 3], [4, 5, 6])  # Ensure valid data
plt.show()
```

---

## **3. Debugging Performance Issues**  

### **3.1 Reducing Memory Usage**  
- **Use Downsampling**: Instead of plotting large datasets, use aggregated data.
- **Optimize Large Plots with `scatter` Instead of `plot`**
  ```python
  import numpy as np

  x = np.random.rand(100000)
  y = np.random.rand(100000)

  plt.scatter(x, y, s=1)  # Use small marker size
  plt.show()
  ```
- **Use `plt.draw()` Instead of `plt.show()` for Live Updates**
  ```python
  plt.plot([1, 2, 3], [4, 5, 6])
  plt.draw()
  ```

---

### **3.2 Fixing Slow Rendering Issues**  
| **Issue**                | **Fix** |
|-------------------------|--------------------------------------------|
| Large datasets.        | Use `scatter()` instead of `plot()`. |
| Multiple subplots.     | Use `fig, ax = plt.subplots()` for efficiency. |
| Inefficient loops.     | Vectorize operations using NumPy. |
| Redundant calls.      | Avoid repeated calls to `show()`. |

✅ **Optimized Plotting with Axes Reuse**  
```python
fig, ax = plt.subplots()
for i in range(3):
    ax.plot([1, 2, 3], [i, i+1, i+2])  # Reuse axes
plt.show()
```

---

## **4. Debugging Compatibility Issues**  

### **4.1 Fixing Backend Issues**  
- Check available backends:  
  ```python
  import matplotlib
  print(matplotlib.get_backend())
  ```
- Set a different backend (e.g., for Jupyter):  
  ```python
  %matplotlib inline  # For Jupyter Notebook
  ```

### **4.2 Fixing `ImportError` or `ModuleNotFoundError`**  
- **Check Installation**  
  ```sh
  pip install --upgrade matplotlib
  ```
- **Check Dependencies**  
  ```python
  import matplotlib
  print(matplotlib.__version__)
  ```
- **Reinstall Dependencies**  
  ```sh
  pip install --upgrade numpy scipy pandas
  ```

---

## **5. Debugging Layout and Appearance Issues**  

### **5.1 Labels and Titles Not Appearing**  
✅ **Ensure `xlabel`, `ylabel`, and `title` are set**  
```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.xlabel("X-Axis")  # Add labels
plt.ylabel("Y-Axis")
plt.title("Title")
plt.show()
```

### **5.2 Legends Not Displaying**  
✅ **Ensure `label` and `legend()` are used**  
```python
plt.plot([1, 2, 3], [4, 5, 6], label="Line A")
plt.legend()
plt.show()
```

---

## **6. Debugging Axis Issues**  

| Issue | Fix |
|---------------------------|----------------------------------|
| Axis labels missing.      | Use `plt.xlabel()`, `plt.ylabel()`. |
| Incorrect axis range.     | Use `plt.xlim()`, `plt.ylim()`. |
| Axis not updating.       | Use `plt.autoscale()`. |

✅ **Fixing Incorrect Axis Scaling**  
```python
plt.plot([1, 2, 3], [400, 500, 600])
plt.yscale("log")  # Apply log scale for better readability
plt.show()
```

---

## **7. Debugging Customization Errors**  

### **7.1 Fixing `TypeError: Invalid color argument`**  
✅ **Ensure Correct Color Formats**
```python
plt.plot([1, 2, 3], [4, 5, 6], color="red")  # Valid color string
plt.show()
```

### **7.2 Fixing `RuntimeError: Invalid Font Configuration`**  
✅ **Check Available Fonts**
```python
import matplotlib.font_manager as fm
print(fm.findSystemFonts(fontpaths=None, fontext='ttf'))
```
✅ **Set a Valid Font**
```python
plt.rcParams["font.family"] = "Arial"
```

---

## **8. Debugging Interactivity Issues**  

### **8.1 Fixing `Figure Freezing` in Interactive Mode**  
✅ **Enable Interactive Mode**  
```python
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

✅ **Properly Handle GUI Events**  
```python
import time
for i in range(5):
    plt.plot([1, 2, 3], [i, i+1, i+2])
    plt.draw()
    time.sleep(1)
```

---

## **9. Debugging Save and Export Issues**  

### **9.1 Fixing `FileNotFoundError` While Saving**  
✅ **Ensure Directory Exists Before Saving**
```python
import os
if not os.path.exists("plots"):
    os.makedirs("plots")
plt.savefig("plots/my_plot.png")
```

### **9.2 Fixing Low-Resolution Output**  
✅ **Increase DPI While Saving**
```python
plt.savefig("high_res_plot.png", dpi=300)
```

---

## **Conclusion**  

Debugging Matplotlib involves **identifying errors, optimizing performance, fixing rendering issues, and ensuring compatibility** with other libraries.  
- **Syntax & Import Errors**: Fix with installation checks and proper imports.  
- **Rendering Issues**: Ensure `plt.show()` is used and valid data is passed.  
- **Performance Optimization**: Use vectorized operations and efficient plotting techniques.  
- **Interactivity & Backend Fixes**: Enable interactive mode and set the correct backend.  
- **Export Issues**: Use `dpi=300` for high-resolution images and verify file paths.
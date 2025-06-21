## **Interpolation and Extrapolation in NumPy**  

## **Overview**  
Interpolation estimates values within a given range of data points, while extrapolation extends values beyond the known data. NumPy supports interpolation through `numpy.interp()`, and SciPy provides advanced interpolation and extrapolation methods.

## **Interpolation in NumPy**  
NumPy provides `numpy.interp()` for linear interpolation.

### **Syntax**  
```python
numpy.interp(x, xp, fp)
```
- `x`: Points where interpolation is required.
- `xp`: Known data points (x-coordinates).
- `fp`: Corresponding function values (y-coordinates).

### **Example**  
```python
import numpy as np

xp = np.array([1, 2, 3, 4, 5])  # Known x-values
fp = np.array([10, 20, 30, 40, 50])  # Known y-values

x_new = np.array([2.5, 3.5, 4.5])  # Points to interpolate
y_new = np.interp(x_new, xp, fp)

print(y_new)
# Output: [25. 35. 45.]
```

### **Handling Out-of-Bounds Values**  
NumPy allows extrapolation using the `left` and `right` parameters.
```python
y_extrapolated = np.interp([0, 6], xp, fp, left=0, right=60)
print(y_extrapolated)  
# Output: [ 0. 60.]
```

## **Interpolation in SciPy**  
SciPy offers more advanced interpolation methods in `scipy.interpolate`.

### **Using `interp1d` for Different Methods**  
```python
from scipy.interpolate import interp1d

xp = np.array([1, 2, 3, 4, 5])
fp = np.array([10, 20, 30, 40, 50])

linear_interp = interp1d(xp, fp, kind='linear')
cubic_interp = interp1d(xp, fp, kind='cubic')

x_new = np.array([2.5, 3.5, 4.5])
print(linear_interp(x_new))  # Linear interpolation
print(cubic_interp(x_new))   # Cubic interpolation
```

### **Interpolation Methods**  
- `'linear'` – Straight-line interpolation.
- `'nearest'` – Nearest neighbor interpolation.
- `'cubic'` – Smooth cubic interpolation.
- `'quadratic'`, `'spline'` – Higher-order smooth curves.

## **Extrapolation in SciPy**  
To extrapolate beyond the known data, set `fill_value="extrapolate"` in `interp1d`.

```python
extrapolate_func = interp1d(xp, fp, kind='linear', fill_value="extrapolate")
print(extrapolate_func([0, 6]))  # Extrapolated values
```

## **Conclusion**  
- **Interpolation** is useful for estimating missing data points.
- **Extrapolation** predicts values beyond available data.
- NumPy’s `interp()` supports simple linear interpolation.
- SciPy’s `interp1d()` provides advanced interpolation and extrapolation.
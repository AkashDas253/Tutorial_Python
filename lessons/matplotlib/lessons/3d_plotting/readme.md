# **3D Plotting in Matplotlib**  

Matplotlib supports **3D plotting** using the `mpl_toolkits.mplot3d` module, enabling visualizations of **surfaces, wireframes, scatter plots, and contours** in three dimensions.

---

## **1. Setting Up a 3D Plot**  

Before plotting in 3D, you need to import the `Axes3D` class and define a 3D figure:

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')  # 3D axis
```

| Parameter | Description |
|-----------|-------------|
| `projection='3d'` | Converts a 2D subplot into a 3D plot. |

---

## **2. 3D Scatter Plot**  

Used to plot points in 3D space.  

```python
x = np.random.rand(50)
y = np.random.rand(50)
z = np.random.rand(50)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

plt.show()
```

| Function | Description |
|----------|-------------|
| `ax.scatter(x, y, z, c, marker)` | Plots 3D points with colors and markers. |

---

## **3. 3D Line Plot**  

Used for plotting curves in 3D.  

```python
t = np.linspace(0, 10, 100)
x = np.sin(t)
y = np.cos(t)
z = t

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x, y, z)

plt.show()
```

| Function | Description |
|----------|-------------|
| `ax.plot(x, y, z)` | Plots a 3D line. |

---

## **4. 3D Surface Plot**  

Creates a continuous surface in 3D.  

```python
X = np.linspace(-5, 5, 50)
Y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(X, Y)
Z = np.sin(np.sqrt(X**2 + Y**2))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')

plt.show()
```

| Function | Description |
|----------|-------------|
| `ax.plot_surface(X, Y, Z, cmap='viridis')` | Creates a 3D surface with colormap. |

---

## **5. 3D Wireframe Plot**  

Creates a **wireframe representation** of a surface.  

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_wireframe(X, Y, Z, color='black')

plt.show()
```

| Function | Description |
|----------|-------------|
| `ax.plot_wireframe(X, Y, Z)` | Plots a wireframe surface. |

---

## **6. 3D Contour Plot**  

Displays **contour lines** in a 3D space.  

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.contour3D(X, Y, Z, 50, cmap='coolwarm')

plt.show()
```

| Function | Description |
|----------|-------------|
| `ax.contour3D(X, Y, Z, levels, cmap)` | Draws 3D contour lines. |

---

## **7. Customizing 3D Plots**  

### **7.1 Setting Labels and Titles**  
```python
ax.set_xlabel("X Axis")
ax.set_ylabel("Y Axis")
ax.set_zlabel("Z Axis")
ax.set_title("3D Plot Example")
```

### **7.2 Adjusting Viewing Angles**  
```python
ax.view_init(elev=30, azim=45)  # Elevation and azimuth
```

| Parameter | Description |
|-----------|-------------|
| `elev` | Controls vertical tilt. |
| `azim` | Controls rotation around Z-axis. |

---

## **8. 3D Bar Plot**  

Used for visualizing categorical data in 3D.  

```python
x = np.array([1, 2, 3, 4, 5])
y = np.array([1, 2, 3, 4, 5])
z = np.zeros(5)
dx = dy = np.ones(5)
dz = [3, 5, 7, 9, 11]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.bar3d(x, y, z, dx, dy, dz, color='b')

plt.show()
```

| Function | Description |
|----------|-------------|
| `ax.bar3d(x, y, z, dx, dy, dz, color)` | Creates 3D bars. |

---

## **9. Exporting 3D Plots**  

### **9.1 Save as PNG**
```python
plt.savefig("3d_plot.png", dpi=300)
```

### **9.2 Save as PDF**
```python
plt.savefig("3d_plot.pdf")
```

| Format | Command |
|--------|----------|
| PNG | `plt.savefig("file.png", dpi=300)` |
| PDF | `plt.savefig("file.pdf")` |

---

## **Conclusion**  
Matplotlib’s `mpl_toolkits.mplot3d` enables **scatter plots, line plots, surfaces, wireframes, contours, and bars** in 3D. It also provides tools for **customization, viewing angles, and exporting** to various formats.
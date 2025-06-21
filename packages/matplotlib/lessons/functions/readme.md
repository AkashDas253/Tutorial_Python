## **Plotting Functions in Matplotlib**

Matplotlib provides a wide variety of plotting functions to visualize data effectively. These functions can be categorized as **basic plots**, **statistical plots**, and **advanced plots**.

---

### **1. Basic Plots**

| Plot Type               | Description                                            | Common Usage Example                              |
|-------------------------|--------------------------------------------------------|--------------------------------------------------|
| **Line Plot (`plot`)**   | Displays data as a continuous line, suitable for trends.| `plt.plot(x, y)`                                  |
| **Scatter Plot (`scatter`)** | Represents data points using markers, showing relationships between variables.| `plt.scatter(x, y)`                              |
| **Bar Plot (`bar`)**     | Displays data as vertical bars, ideal for comparing categories.| `plt.bar(categories, values)`                    |
| **Horizontal Bar Plot (`barh`)** | Horizontal version of the bar plot.               | `plt.barh(categories, values)`                   |
| **Histogram (`hist`)**   | Shows the frequency distribution of a dataset.         | `plt.hist(data, bins=10)`                        |
| **Pie Chart (`pie`)**    | Represents data proportions as segments of a circle.   | `plt.pie(sizes, labels=labels)`                  |

---

### **2. Statistical Plots**

Statistical plots allow for data distribution and variability visualization.

| Plot Type                     | Description                                       | Common Usage Example                                |
|-------------------------------|---------------------------------------------------|----------------------------------------------------|
| **Box Plot (`boxplot`)**       | Shows data distribution using quartiles.         | `plt.boxplot(data)`                                |
| **Violin Plot (`violinplot`)** | Combines boxplot and kernel density estimation for data distribution.| `plt.violinplot(data)`                            |

---

### **3. Advanced Plots**

Advanced plots are used for specialized data visualization, often involving 3D or complex patterns.

| Plot Type                 | Description                                            | Common Usage Example                                |
|---------------------------|--------------------------------------------------------|----------------------------------------------------|
| **3D Plots**              | Displays data in 3D space using scatter, line, or surface plots.| `ax.scatter3D(x, y, z)` (requires `mpl_toolkits.mplot3d`)|
| **Contour Plot (`contour`)**| Shows level curves for 3D data on a 2D plane.         | `plt.contour(X, Y, Z)`                             |
| **Filled Contour (`contourf`)**| Similar to `contour`, but with filled areas.       | `plt.contourf(X, Y, Z, cmap='viridis')`            |
| **Heatmap (`imshow`)**     | Displays data intensity using colors.                 | `plt.imshow(data, cmap='hot')`                     |
| **Quiver Plot (`quiver`)** | Shows vector fields using arrows.                     | `plt.quiver(X, Y, U, V)`                           |
| **Streamline Plot (`streamplot`)** | Plots continuous vector flow as streamlines. | `plt.streamplot(X, Y, U, V)`                       |

---

### **Key Parameters for Plot Customization**

| Parameter        | Description                                                                 | Usage Example                                      |
|------------------|-----------------------------------------------------------------------------|---------------------------------------------------|
| **`color`**      | Sets the color of lines or markers.                                         | `plt.plot(x, y, color='red')`                      |
| **`linewidth`**  | Adjusts the thickness of lines.                                            | `plt.plot(x, y, linewidth=2.0)`                    |
| **`marker`**     | Specifies the marker style for data points.                                | `plt.plot(x, y, marker='o')`                       |
| **`alpha`**      | Sets transparency for lines or markers (0 = transparent, 1 = opaque).      | `plt.scatter(x, y, alpha=0.5)`                     |
| **`label`**      | Adds a label for the legend.                                               | `plt.plot(x, y, label='Line 1')`                   |
| **`cmap`**       | Specifies a colormap for data intensity visualization.                     | `plt.imshow(data, cmap='viridis')`                 |

---

### **4. Examples of Plotting**

#### **Basic Line Plot**
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, color='blue', linewidth=2, label='Sine wave')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Line Plot Example')
plt.legend()
plt.show()
```

#### **Scatter Plot**
```python
plt.scatter(x, y, color='red', marker='o', alpha=0.7)
plt.title('Scatter Plot Example')
plt.show()
```

#### **Histogram**
```python
data = np.random.randn(1000)
plt.hist(data, bins=20, color='green', alpha=0.8)
plt.title('Histogram Example')
plt.show()
```

#### **3D Scatter Plot**
```python
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
z = np.random.rand(50)
ax.scatter3D(x, y, z, c=z, cmap='viridis')
plt.title('3D Scatter Plot Example')
plt.show()
```

---

### **5. Comparison of Plotting Functions**

| Function           | When to Use                                                                 | Example                                              |
|--------------------|-----------------------------------------------------------------------------|-----------------------------------------------------|
| **`plot()`**       | Best for trends and continuous data.                                       | Plotting sine waves or time series.                 |
| **`scatter()`**    | Ideal for individual data points and correlations.                        | Visualizing clusters or relationships.              |
| **`bar()`**        | Useful for comparing discrete categories.                                 | Sales of products over a year.                      |
| **`hist()`**       | Best for frequency or distribution analysis.                              | Examining age groups in a survey.                   |
| **`imshow()`**     | For visualizing 2D data like heatmaps.                                    | Displaying a correlation matrix.                    |
| **`quiver()`**     | To visualize vector fields or directions.                                 | Displaying wind flow patterns.                      |

---

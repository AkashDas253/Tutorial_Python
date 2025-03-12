## **Matplotlib Basics**

**1. Figure and Axes**

- **Figure**: The entire canvas for the plot. It holds one or more axes and can be customized in terms of size, resolution (DPI), and more.
  
- **Axes**: These are individual plots within the figure. Each axis contains the data and visual elements such as lines, markers, and labels.

- **Subplots**: Multiple axes can be placed within a single figure for better data comparison. A subplot can be defined within a grid layout.

- **Figure size and DPI settings**: You can set the size and resolution of the figure for better control over the plot's appearance.

| Concept                  | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **Figure**               | The container that holds one or more axes and other elements.               |
| **Axes**                 | Represents individual plots within a figure.                               |
| **Subplots**             | Multiple axes in a grid layout within a single figure.                      |
| **Figure size**          | Controls the dimensions of the figure (in inches).                          |
| **DPI (Dots Per Inch)**  | Specifies the resolution of the figure, affecting image quality.            |

---

**2. Backends**

Matplotlib can render plots using different backends. A backend is a rendering engine that controls how the figure is displayed or saved.

| Backend Type            | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **Interactive Backends**| These backends allow interaction with the plot (e.g., zooming, panning).    |
| **Non-interactive Backends**| Suitable for saving images and rendering plots without interaction.        |

| Example                   | Usage                                              |
|---------------------------|----------------------------------------------------|
| **Qt5Agg, TkAgg**         | Interactive backends, good for real-time display and interaction.           |
| **Agg, SVG**              | Non-interactive, ideal for saving plots in various file formats.             |

---

**3. Matplotlib Modules**

Matplotlib is organized into several modules that help with different plotting tasks.

| Module Name             | Description                                                                 |
|-------------------------|-----------------------------------------------------------------------------|
| **matplotlib.pyplot**    | A high-level interface for creating static, animated, and interactive plots. |
| **matplotlib.figure**    | Provides tools to control the figure-level attributes like size, DPI, etc.    |
| **matplotlib.axes**      | Allows for detailed control over axes and their properties.                  |
| **matplotlib.animation** | Used for creating animations.                                                |
| **matplotlib.gridspec**  | Offers advanced grid layouts for subplots.                                   |

| Module                  | Common Usage                                              |
|-------------------------|-----------------------------------------------------------|
| **matplotlib.pyplot**    | Quick and easy plot generation. Ideal for simple, fast plots. |
| **matplotlib.animation** | Creating dynamic plots such as animated charts.              |
| **matplotlib.gridspec**  | Creating complex subplot arrangements with control over spacing. |

---

### **4. Example Usage in Code**

```python
import matplotlib.pyplot as plt
import numpy as np

# Create some data
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Create a figure and axes
fig, ax = plt.subplots(figsize=(8, 6), dpi=100)

# Plotting the data
ax.plot(x, y, label='Sine wave', color='b')

# Adding labels
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

# Adding a title
ax.set_title('Sine Wave Plot')

# Show the plot
plt.show()
```

In this example:
- **plt.subplots()** is used to create a figure and axes.
- **ax.plot()** plots the data on the axes.
- **ax.set_xlabel()**, **ax.set_ylabel()**, and **ax.set_title()** are used to customize the labels and title.

---

### **5. Key Functions for Basic Plotting**

| Function                | Description                                                     | Usage Example                                  |
|-------------------------|-----------------------------------------------------------------|------------------------------------------------|
| **plt.plot()**           | Plots a line graph.                                             | `plt.plot(x, y)`                              |
| **plt.scatter()**        | Plots a scatter plot.                                           | `plt.scatter(x, y)`                           |
| **plt.bar()**            | Creates a bar plot (vertical).                                  | `plt.bar(x, y)`                               |
| **plt.barh()**           | Creates a horizontal bar plot.                                  | `plt.barh(x, y)`                              |
| **plt.hist()**           | Plots a histogram.                                              | `plt.hist(data, bins=10)`                      |
| **plt.pie()**            | Creates a pie chart.                                            | `plt.pie(sizes, labels=labels)`                |

---

### **6. Customization in Matplotlib**

- **Line Styles**
  You can customize the appearance of lines in plots, such as their style and width.

| Line Style  | Usage                                  |
|-------------|----------------------------------------|
| **'-'**     | Solid line                             |
| **'--'**    | Dashed line                            |
| **'-.'**    | Dash-dot line                          |
| **':'**     | Dotted line                            |

- **Marker Styles**
  Markers are used to highlight data points.

| Marker     | Usage                                  |
|------------|----------------------------------------|
| **'o'**    | Circle                                  |
| **'s'**    | Square                                  |
| **'^'**    | Triangle up                             |

---

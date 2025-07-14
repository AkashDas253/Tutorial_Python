# **Integration with Other Libraries in Matplotlib**  

Matplotlib can be **integrated with various libraries** to enhance its functionality, making it more efficient for data visualization in different contexts.

---

## **1. Integration with Pandas**  

Pandas provides built-in plotting functions using Matplotlib, allowing direct visualization of DataFrames and Series.

### **1.1 Plotting Directly from Pandas**
```python
import pandas as pd
import matplotlib.pyplot as plt

# Sample DataFrame
data = {'X': [1, 2, 3, 4], 'Y': [10, 20, 25, 30]}
df = pd.DataFrame(data)

df.plot(x='X', y='Y', kind='line', title="Line Plot")
plt.show()
```

| Pandas Function | Description |
|----------------|-------------|
| `df.plot(kind='line')` | Line plot. |
| `df.plot(kind='bar')` | Bar plot. |
| `df.plot(kind='scatter', x='A', y='B')` | Scatter plot. |
| `df.hist(column='A')` | Histogram. |

### **1.2 Using `matplotlib` for More Control**  
```python
df.plot(kind='bar')
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Bar Plot with Pandas")
plt.show()
```

---

## **2. Integration with NumPy**  

Matplotlib works efficiently with NumPy for mathematical operations and large datasets.

### **2.1 Plotting NumPy Arrays**
```python
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.plot(x, y, label="Sine Wave")
plt.legend()
plt.show()
```

| NumPy Function | Description |
|---------------|-------------|
| `np.linspace(start, stop, num)` | Generates evenly spaced numbers. |
| `np.sin(x)`, `np.cos(x)`, `np.exp(x)` | Mathematical functions. |

---

## **3. Integration with Seaborn**  

Seaborn provides **statistical visualization** on top of Matplotlib.

### **3.1 Using Seaborn for Statistical Plots**  
```python
import seaborn as sns

sns.set(style="darkgrid")  # Apply Seaborn style
sns.lineplot(x=[1, 2, 3, 4], y=[10, 20, 25, 30])
plt.show()
```

| Seaborn Function | Description |
|-----------------|-------------|
| `sns.lineplot()` | Line plot with automatic styling. |
| `sns.histplot()` | Histogram with KDE. |
| `sns.boxplot()` | Box plot for distribution. |
| `sns.scatterplot()` | Scatter plot with aesthetics. |

### **3.2 Combining Seaborn and Matplotlib**
```python
sns.set_theme()
plt.figure(figsize=(8, 5))
sns.histplot(df["Y"], kde=True)
plt.title("Histogram with KDE")
plt.show()
```

---

## **4. Integration with SciPy**  

SciPy provides **scientific computing functions**, useful for curve fitting, interpolation, and signal processing.

### **4.1 Curve Fitting Example**
```python
from scipy.optimize import curve_fit

def func(x, a, b):
    return a * np.exp(b * x)

xdata = np.linspace(0, 4, 50)
ydata = func(xdata, 2, -1.5) + 0.2 * np.random.normal(size=len(xdata))

popt, _ = curve_fit(func, xdata, ydata)

plt.scatter(xdata, ydata, label="Data")
plt.plot(xdata, func(xdata, *popt), color='red', label="Fitted Curve")
plt.legend()
plt.show()
```

| SciPy Function | Description |
|---------------|-------------|
| `curve_fit()` | Fits a curve to data. |
| `signal.spectrogram()` | Computes signal spectrogram. |
| `interpolate.spline()` | Interpolates smooth curves. |

---

## **5. Integration with Plotly**  

Plotly enhances Matplotlib with **interactive plots**.

### **5.1 Converting Matplotlib to Plotly**
```python
import plotly.tools as tls

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

plotly_fig = tls.mpl_to_plotly(fig)
plotly_fig.show()
```

| Plotly Function | Description |
|----------------|-------------|
| `mpl_to_plotly(fig)` | Converts Matplotlib figures to Plotly. |
| `plotly.graph_objects` | Create interactive charts. |

---

## **6. Integration with OpenCV**  

OpenCV processes images, which can be visualized using Matplotlib.

### **6.1 Displaying an Image with Matplotlib**
```python
import cv2

img = cv2.imread("image.jpg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

plt.imshow(img)
plt.axis("off")
plt.show()
```

| OpenCV Function | Description |
|---------------|-------------|
| `cv2.imread("file")` | Reads an image. |
| `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` | Converts BGR to RGB for proper display. |

---

## **7. Integration with TensorFlow**  

TensorFlow generates and visualizes deep learning model outputs.

### **7.1 Displaying Model Training Progress**
```python
import tensorflow as tf

history = model.fit(x_train, y_train, epochs=10, validation_data=(x_val, y_val))

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
```

| TensorFlow Function | Description |
|--------------------|-------------|
| `model.fit()` | Trains a model. |
| `history.history` | Stores training progress. |

---

## **8. Integration with NetworkX**  

NetworkX allows visualization of graphs and networks.

### **8.1 Plotting a Graph with Matplotlib**
```python
import networkx as nx

G = nx.erdos_renyi_graph(10, 0.3)
nx.draw(G, with_labels=True)
plt.show()
```

| NetworkX Function | Description |
|------------------|-------------|
| `nx.draw(G)` | Plots a network graph. |
| `nx.spring_layout(G)` | Positions nodes dynamically. |

---

## **9. Integration with Bokeh**  

Bokeh creates **interactive plots** that extend Matplotlib.

### **9.1 Embedding a Matplotlib Figure in Bokeh**
```python
from bokeh.plotting import figure, show
from bokeh.mpl import to_bokeh

plt.plot([1, 2, 3], [4, 5, 6])
show(to_bokeh(plt.gcf()))
```

| Bokeh Function | Description |
|---------------|-------------|
| `to_bokeh(fig)` | Converts Matplotlib figure to Bokeh. |

---

## **10. Integration with LaTeX**  

Matplotlib supports **LaTeX formatting** for mathematical expressions.

### **10.1 Using LaTeX in Titles and Labels**
```python
plt.title(r"$E = mc^2$")
plt.xlabel(r"$\alpha + \beta$ Axis")
plt.show()
```

| LaTeX Syntax | Description |
|-------------|-------------|
| `$E=mc^2$` | Inline equation. |
| `r"\frac{a}{b}"` | Fraction. |
| `r"\sum_{i=1}^{n} x_i$` | Summation. |

---

## **Conclusion**  

Matplotlib integrates with **data science, machine learning, and interactive visualization** libraries.  
- **Pandas & NumPy**: Direct plotting support.  
- **Seaborn & SciPy**: Enhanced statistical visualization.  
- **TensorFlow & OpenCV**: Model performance and image processing.  
- **NetworkX & Plotly**: Network graphs and interactivity.  
- **LaTeX & Bokeh**: Mathematical notation and web-based plotting.
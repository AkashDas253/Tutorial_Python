# **Object-Oriented API in Matplotlib**  

Matplotlib’s **Object-Oriented (OO) API** provides greater control over figures and axes, making it more suitable for complex and customized visualizations. Unlike the `pyplot` interface, which uses implicit figure and axis creation, the OO approach explicitly creates and manages figure elements.

---

## **1. Figure and Axes: Core Objects**  

### **1.1 Figure Object (`Figure`)**  
The **Figure** is the entire canvas where one or more **Axes** (subplots) exist.  

```python
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(6, 4), dpi=100)  # Create a figure
```

| Attribute | Description |
|-----------|-------------|
| `figsize` | Controls the figure width and height in inches |
| `dpi` | Dots per inch (resolution of the figure) |

---

### **1.2 Axes Object (`Axes`)**  
The **Axes** is a specific area inside a figure where data is plotted.

```python
ax = fig.add_subplot(1, 1, 1)  # Add a subplot (1 row, 1 column, 1st position)
ax.plot([1, 2, 3], [4, 5, 6])  # Plot on this specific Axes
plt.show()
```

| Attribute | Description |
|-----------|-------------|
| `add_subplot(rows, cols, index)` | Adds an axes to a figure |
| `plot(x, y)` | Plots a line graph on the axes |

---

## **2. Creating Figures and Axes Explicitly**  

Instead of relying on `plt.plot()`, which implicitly creates figures and axes, the OO approach explicitly manages them.

### **2.1 Using `plt.subplots()` (Preferred Approach)**
```python
fig, ax = plt.subplots(figsize=(6, 4))  # Create figure and axes
ax.plot([1, 2, 3], [4, 5, 6])  # Use ax object for plotting
plt.show()
```

| Function | Description |
|----------|-------------|
| `plt.subplots()` | Creates a figure and axes together |
| `ax.plot()` | Plots on the specific axes |

---

### **2.2 Multiple Subplots**
```python
fig, axes = plt.subplots(2, 2, figsize=(8, 6))  # 2x2 grid of subplots

axes[0, 0].plot([1, 2, 3], [4, 5, 6], color='r')
axes[0, 1].bar([1, 2, 3], [4, 5, 6])
axes[1, 0].scatter([1, 2, 3], [4, 5, 6])
axes[1, 1].hist([1, 2, 3, 3, 3, 4, 5, 6], bins=3)

plt.tight_layout()
plt.show()
```

| Function | Description |
|----------|-------------|
| `plt.subplots(rows, cols)` | Creates multiple subplots |
| `axes[row, col]` | Accesses individual subplots |

---

## **3. Customizing Plots in the OO API**  

### **3.1 Titles and Labels**
```python
fig, ax = plt.subplots()

ax.set_title("Graph Title")
ax.set_xlabel("X-Axis Label")
ax.set_ylabel("Y-Axis Label")

ax.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

| Method | Description |
|--------|-------------|
| `ax.set_title()` | Sets the title of the axes |
| `ax.set_xlabel()` | Labels the x-axis |
| `ax.set_ylabel()` | Labels the y-axis |

---

### **3.2 Axis Limits**
```python
ax.set_xlim(0, 5)  # Set x-axis limits
ax.set_ylim(0, 10)  # Set y-axis limits
```

| Method | Description |
|--------|-------------|
| `set_xlim(min, max)` | Sets x-axis range |
| `set_ylim(min, max)` | Sets y-axis range |

---

### **3.3 Grid, Legends, and Styles**
```python
ax.grid(True)  # Add grid
ax.legend(["Line 1"])  # Add legend
```

| Method | Description |
|--------|-------------|
| `grid(True)` | Enables grid lines |
| `legend()` | Adds a legend |

---

## **4. Combining Multiple Plots on a Single Axes**
```python
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6], label="Line 1", color="blue")
ax.scatter([1, 2, 3], [3, 4, 5], label="Scatter 1", color="red")

ax.legend()
plt.show()
```

| Plot Type | Function |
|-----------|----------|
| Line | `ax.plot()` |
| Scatter | `ax.scatter()` |

---

## **5. Saving Figures in the OO API**
```python
fig.savefig("plot.png", dpi=300)
```

| Parameter | Description |
|-----------|-------------|
| `dpi` | Image resolution |

---

## **Conclusion**  
The **Object-Oriented API** in Matplotlib provides direct control over figures and axes, allowing for more complex and customizable visualizations. It is the recommended approach for multi-plot layouts and advanced customization.
# **Layout and Subplots in Matplotlib**  

Matplotlib provides tools for creating multiple plots within a figure using subplots and layout adjustments to control spacing, alignment, and aspect ratios.  

---

## **1. Creating Subplots**  

Matplotlib provides multiple methods for creating subplots.  

### **Using `plt.subplot()`**  
Creates a single subplot within a figure.  

```python
plt.subplot(rows, cols, index)
```

| Parameter | Description |
|-----------|-------------|
| `rows` | Number of rows in the grid |
| `cols` | Number of columns in the grid |
| `index` | Position of the subplot (1-based index) |

**Example:**  
```python
plt.subplot(2, 2, 1)  # First subplot in a 2x2 grid
plt.plot(x, y)
```

---

### **Using `plt.subplots()`**  
Creates multiple subplots in a single function call.  

```python
fig, axes = plt.subplots(rows, cols, figsize=(width, height))
```

| Parameter | Description |
|-----------|-------------|
| `rows` | Number of rows |
| `cols` | Number of columns |
| `figsize` | Figure size in inches |

**Example:**  
```python
fig, axes = plt.subplots(2, 2, figsize=(8, 6))
axes[0, 0].plot(x, y)
axes[1, 1].plot(x, z)
```

---

## **2. Adjusting Subplot Layout**  

### **Using `plt.tight_layout()`**  
Automatically adjusts spacing between subplots to avoid overlapping.  

```python
plt.tight_layout()
```

### **Using `plt.subplots_adjust()`**  
Manually adjusts subplot spacing.  

```python
plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.4, hspace=0.4)
```

| Parameter | Description |
|-----------|-------------|
| `left` | Left margin (0 to 1) |
| `right` | Right margin (0 to 1) |
| `top` | Top margin (0 to 1) |
| `bottom` | Bottom margin (0 to 1) |
| `wspace` | Width spacing between subplots |
| `hspace` | Height spacing between subplots |

---

## **3. Sharing Axes**  

To ensure multiple subplots share the same axis scale.  

```python
fig, axes = plt.subplots(2, 2, sharex=True, sharey=True)
```

| Parameter | Description |
|-----------|-------------|
| `sharex=True` | Shared x-axis for all subplots |
| `sharey=True` | Shared y-axis for all subplots |

---

## **4. GridSpec for Complex Layouts**  

`GridSpec` allows for advanced layout customization.  

```python
from matplotlib.gridspec import GridSpec
fig = plt.figure(figsize=(8, 6))
gs = GridSpec(3, 3, figure=fig)
ax1 = fig.add_subplot(gs[0, :])  # Spans all columns
ax2 = fig.add_subplot(gs[1:, 0])  # Spans two rows
ax3 = fig.add_subplot(gs[1:, 1:])  # Spans two columns
```

| GridSpec Index | Description |
|---------------|-------------|
| `[0, :]` | First row spanning all columns |
| `[1:, 0]` | Second and third rows, first column |
| `[1:, 1:]` | Second and third rows, second and third columns |

---

## **5. Inset and Twin Axes**  

### **Inset Axes (Zoomed View)**  
```python
ax_inset = ax.inset_axes([x, y, width, height])
```

| Parameter | Description |
|-----------|-------------|
| `x, y` | Bottom-left corner coordinates (relative to axes) |
| `width, height` | Size of the inset |

### **Twin Axes (Dual Y-Axis)**  
```python
ax2 = ax.twinx()
ax2.plot(x, y2, 'r')
```

| Function | Description |
|----------|-------------|
| `twinx()` | Creates a secondary y-axis |
| `twiny()` | Creates a secondary x-axis |

---

## **6. Saving Figures with Layouts**  

```python
plt.savefig('subplots.png', dpi=300, bbox_inches='tight')
```

| Parameter | Description |
|-----------|-------------|
| `dpi=300` | High-resolution output |
| `bbox_inches='tight'` | Crops extra white space |

---

## **Conclusion**  

Matplotlib provides flexible layout options using subplots, axis sharing, and advanced grid configurations. Using these techniques, multiple plots can be displayed efficiently within a figure.
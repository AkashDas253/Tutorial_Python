# **Output and Export in Matplotlib**  

Matplotlib provides multiple ways to **display, save, and export** plots in various formats, ensuring flexibility in reporting and presentation.

---

## **1. Displaying Plots**  

### **1.1 Using `plt.show()`**  
- Displays the plot in an interactive window.
- Used when working in **scripts**.

```python
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
plt.show()
```

| Function | Description |
|----------|-------------|
| `plt.show()` | Opens an interactive window displaying the plot. |

---

### **1.2 Using `%matplotlib inline` (Jupyter Notebooks)**
- Embeds the plot **inside the notebook**.
- Used when working in **Jupyter Notebook**.

```python
%matplotlib inline
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [4, 5, 6])
```

| Command | Description |
|---------|-------------|
| `%matplotlib inline` | Displays the plot inside a Jupyter Notebook. |

---

## **2. Saving Plots**  

### **2.1 Using `savefig()`**  
- Saves the figure in various formats.
- Can specify **DPI, quality, and background color**.

```python
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("plot.png", dpi=300, bbox_inches='tight')
```

| Parameter | Description |
|-----------|-------------|
| `"plot.png"` | Filename and format. |
| `dpi=300` | Sets resolution (dots per inch). |
| `bbox_inches='tight'` | Removes extra whitespace. |

---

## **3. Supported File Formats**  

Matplotlib supports multiple file formats for exporting plots:

| Format | Extension | Description |
|--------|----------|-------------|
| PNG | `.png` | Lossless image format (default). |
| JPEG | `.jpg` | Compressed format, good for photos. |
| PDF | `.pdf` | Vector-based, suitable for printing. |
| SVG | `.svg` | Scalable vector format, best for web graphics. |
| EPS | `.eps` | Used for high-quality printing. |
| TIFF | `.tiff` | High-resolution image format. |

Example for different formats:

```python
plt.savefig("plot.pdf")  # Saves as PDF
plt.savefig("plot.svg")  # Saves as SVG
plt.savefig("plot.jpg", quality=95)  # Saves as JPEG with high quality
```

---

## **4. Adjusting Export Quality**  

### **4.1 Setting DPI for High-Resolution Images**
```python
plt.savefig("high_res_plot.png", dpi=600)
```

### **4.2 Setting Transparent Background**
```python
plt.savefig("transparent_plot.png", transparent=True)
```

| Parameter | Description |
|-----------|-------------|
| `dpi=600` | Increases image resolution. |
| `transparent=True` | Removes background color. |

---

## **5. Exporting with Custom Size**  

Before saving, set figure size using `figsize`:

```python
plt.figure(figsize=(8, 6))  # 8x6 inches
plt.plot([1, 2, 3], [4, 5, 6])
plt.savefig("custom_size_plot.png", dpi=300)
```

| Parameter | Description |
|-----------|-------------|
| `figsize=(width, height)` | Sets figure dimensions in inches. |

---

## **6. Exporting Multiple Plots**  

```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for ax in axes.flat:
    ax.plot([1, 2, 3], [4, 5, 6])

plt.savefig("multiple_plots.png", dpi=300)
```

---

## **7. Exporting Vector Graphics**  

Use **SVG, PDF, or EPS** for scalable graphics.

```python
plt.savefig("vector_plot.svg")  # Best for web
plt.savefig("vector_plot.pdf")  # Best for print
plt.savefig("vector_plot.eps")  # Best for professional publishing
```

| Format | Use Case |
|--------|----------|
| SVG | Scalable, ideal for web graphics. |
| PDF | High quality, used for printing. |
| EPS | Suitable for LaTeX and publishing. |

---

## **8. Exporting to a Bytes Buffer**  

Useful for embedding images in web applications.

```python
import io
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
```

---

## **9. Exporting with Metadata**  

You can embed metadata such as **title, author, and description** in the file.

```python
plt.savefig("plot_with_metadata.png", dpi=300, metadata={"Title": "My Plot", "Author": "User"})
```

| Parameter | Description |
|-----------|-------------|
| `metadata={"Title": "...", "Author": "..."}` | Adds metadata to the exported file. |

---

## **10. Exporting Interactive Plots**  

For **interactive visualizations**, Matplotlib integrates with Plotly, Bokeh, and Dash.

Example: **Convert Matplotlib to Plotly**  
```python
import plotly.tools as tls
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])

plotly_fig = tls.mpl_to_plotly(fig)
plotly_fig.show()
```

---

## **Conclusion**  

Matplotlib allows flexible **display, saving, and exporting** of plots in multiple formats.  
- **`plt.show()`** for displaying plots.  
- **`savefig()`** for saving images in various formats.  
- **High-DPI, metadata, and transparency options** improve output quality.  
- **Vector formats (SVG, PDF, EPS)** ensure scalability.
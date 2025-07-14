## Saving Plots in Seaborn  

Seaborn uses **Matplotlib's `savefig` function** to save visualizations in various formats.  

---

### 1. **Basic Saving (`savefig`)**  
Saves the current figure in an image file.  

**Example – Save as PNG**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
tips = sns.load_dataset("tips")

sns.boxplot(x="day", y="total_bill", data=tips)

plt.savefig("plot.png")  # Saves as PNG
```

---

### 2. **Choosing File Format**  
Seaborn supports multiple formats like PNG, PDF, SVG, and JPEG.  

| Format | Usage |
|--------|-------|
| PNG | `plt.savefig("plot.png")` |
| JPEG | `plt.savefig("plot.jpg")` |
| PDF | `plt.savefig("plot.pdf")` |
| SVG | `plt.savefig("plot.svg")` |

---

### 3. **Adjusting Resolution (`dpi`)**  
Controls image quality (dots per inch).  

**Example – High-Resolution Image**  
```python
plt.savefig("high_res_plot.png", dpi=300)
```

---

### 4. **Removing Extra Borders (`bbox_inches`)**  
Trims unnecessary white spaces.  

**Example – Save Without Borders**  
```python
plt.savefig("trimmed_plot.png", bbox_inches="tight")
```

---

### 5. **Saving Transparent Images (`transparent`)**  
Removes the background for overlays.  

**Example – Save Transparent Image**  
```python
plt.savefig("transparent_plot.png", transparent=True)
```

---

### Summary  

| Parameter | Usage |
|-----------|---------|
| `savefig("file.png")` | Save as PNG |
| `savefig("file.pdf")` | Save as PDF |
| `dpi=300` | High-resolution output |
| `bbox_inches="tight"` | Remove extra white space |
| `transparent=True` | Save without background |

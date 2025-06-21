## Customization and Styling in Seaborn  

Seaborn allows customization of plots using **themes, colors, scales, and style settings** to improve visualization.  

---

### 1. **Setting Themes (`set_theme`)**  
Seaborn provides built-in themes to style plots consistently.  

**Available Themes:**  
| Theme | Description |
|-------|-------------|
| `darkgrid` | Default, grid background |
| `whitegrid` | White background with grid |
| `dark` | Dark background without grid |
| `white` | White background without grid |
| `ticks` | Minimalist with ticks |

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")  
sns.histplot([1, 2, 3, 4, 5])
plt.show()
```

---

### 2. **Changing Color Palettes (`set_palette`)**  
Seaborn provides predefined and custom color palettes.  

**Predefined Palettes:**  
| Palette | Description |
|---------|-------------|
| `deep` | Default colors |
| `muted` | Softer shades |
| `bright` | Vivid colors |
| `dark` | Darker shades |
| `colorblind` | Colorblind-friendly |

**Example:**  
```python
sns.set_palette("muted")
sns.scatterplot(x=[1, 2, 3], y=[4, 5, 6])
plt.show()
```

**Custom Colors:**  
```python
custom_palette = ["#FF5733", "#33FF57", "#3357FF"]
sns.set_palette(custom_palette)
```

---

### 3. **Context Settings (`set_context`)**  
Controls scale for different visualization purposes.  

| Context | Usage |
|---------|-------|
| `paper` | Smallest, for documents |
| `notebook` | Default, for Jupyter notebooks |
| `talk` | Slightly larger for presentations |
| `poster` | Largest for big displays |

**Example:**  
```python
sns.set_context("talk")
sns.lineplot(x=[1, 2, 3], y=[3, 2, 1])
plt.show()
```

---

### 4. **Modifying Figure Size (`set_rc`)**  
**Example:**  
```python
sns.set(rc={"figure.figsize": (8, 6)})
```

---

### 5. **Adding Despine (`despine`)**  
Removes plot borders.  

**Example:**  
```python
sns.histplot([1, 2, 3, 4, 5])
sns.despine()
plt.show()
```

---

### Summary  
| Feature | Function |
|---------|------------|
| **Themes** | `set_theme(style="whitegrid")` |
| **Color Palettes** | `set_palette("muted")` |
| **Context Scaling** | `set_context("talk")` |
| **Figure Size** | `set(rc={"figure.figsize": (8, 6)})` |
| **Remove Borders** | `despine()` |

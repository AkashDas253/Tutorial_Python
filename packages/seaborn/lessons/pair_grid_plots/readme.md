## Pair and Grid Plots in Seaborn  

Pair and grid plots in Seaborn allow visualization of relationships between multiple numerical variables in a dataset.  

---

### Pair Plot (`pairplot`)  
A **pair plot** creates scatter plots for variable pairs and histograms for distributions.  

**Syntax:**  
```python
sns.pairplot(data, hue, diag_kind)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

iris = sns.load_dataset("iris")
sns.pairplot(iris, hue="species")
plt.show()
```

- `hue`: Colors plots by category.  
- `diag_kind`: Type of diagonal plots (`"hist"` or `"kde"`).  

**Example with KDE on Diagonal:**  
```python
sns.pairplot(iris, hue="species", diag_kind="kde")
plt.show()
```

---

### Joint Plot (`jointplot`)  
A **joint plot** combines scatter plots with marginal histograms or KDE plots.  

**Syntax:**  
```python
sns.jointplot(x, y, data, kind)
```

**Example:**  
```python
tips = sns.load_dataset("tips")
sns.jointplot(x="total_bill", y="tip", data=tips, kind="scatter")
plt.show()
```

- `kind="scatter"`: Default scatter plot.  
- `kind="kde"`: Kernel density estimate.  
- `kind="hex"`: Hexagonal binning for large datasets.  

**Example with KDE:**  
```python
sns.jointplot(x="total_bill", y="tip", data=tips, kind="kde")
plt.show()
```

---

### Pair Grid (`PairGrid`)  
A **PairGrid** gives more customization than `pairplot()`.  

**Syntax:**  
```python
g = sns.PairGrid(data)
g.map_diag(func)
g.map_offdiag(func)
```

**Example:**  
```python
g = sns.PairGrid(iris)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
plt.show()
```

- `map_diag()`: Sets diagonal plots.  
- `map_offdiag()`: Sets off-diagonal plots.  

---

### Facet Grid (`FacetGrid`)  
A **Facet Grid** creates subplots based on categorical variables.  

**Syntax:**  
```python
g = sns.FacetGrid(data, col, row)
g.map(func, x, y)
```

**Example:**  
```python
g = sns.FacetGrid(tips, col="sex", row="time")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()
```

- `col`: Creates subplots in columns.  
- `row`: Creates subplots in rows.  

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `pairplot` | Visualizes relationships between multiple variables |
| `jointplot` | Combines scatter plot with marginal histograms |
| `PairGrid` | Customizable alternative to `pairplot` |
| `FacetGrid` | Creates subplots based on categories |

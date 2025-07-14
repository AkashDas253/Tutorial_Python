## Multi-Plot Grids in Seaborn  

Seaborn provides **multi-plot grids** to visualize multiple subplots based on categorical variables. These grids allow easy comparison of different groups.  

---

### 1. **FacetGrid**  
A **FacetGrid** is used for plotting multiple subplots based on categorical variables.  

**Syntax:**  
```python
g = sns.FacetGrid(data, col, row, hue)
g.map(func, x, y)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
g = sns.FacetGrid(tips, col="sex", row="time")
g.map(sns.scatterplot, "total_bill", "tip")
plt.show()
```

- `col`: Creates subplots in columns.  
- `row`: Creates subplots in rows.  
- `hue`: Colors plots by category.  

---

### 2. **PairGrid**  
A **PairGrid** provides a customizable version of `pairplot()`.  

**Syntax:**  
```python
g = sns.PairGrid(data)
g.map_diag(func)
g.map_offdiag(func)
```

**Example:**  
```python
iris = sns.load_dataset("iris")
g = sns.PairGrid(iris)
g.map_diag(sns.histplot)
g.map_offdiag(sns.scatterplot)
plt.show()
```

- `map_diag()`: Sets plots for diagonal elements.  
- `map_offdiag()`: Sets plots for off-diagonal elements.  

---

### 3. **JointGrid**  
A **JointGrid** is a customizable version of `jointplot()`.  

**Syntax:**  
```python
g = sns.JointGrid(x, y, data)
g.plot(func1, func2)
```

**Example:**  
```python
g = sns.JointGrid(x="total_bill", y="tip", data=tips)
g.plot(sns.scatterplot, sns.histplot)
plt.show()
```

- `plot()`: Sets main and marginal plots.  

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `FacetGrid` | Creates subplots based on categories |
| `PairGrid` | Customizes pairwise relationships between variables |
| `JointGrid` | Customizes scatter plots with marginal distributions |

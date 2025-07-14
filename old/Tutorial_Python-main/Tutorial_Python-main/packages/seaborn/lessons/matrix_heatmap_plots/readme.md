## Matrix and Heatmap Plots in Seaborn  

Matrix plots in Seaborn are used to visualize relationships in tabular data, particularly correlations between numerical variables.  

---

### Heatmap (`heatmap`)  
A **heatmap** represents matrix values using color intensity.  

**Syntax:**  
```python
sns.heatmap(data, annot, cmap, linewidths)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset("flights").pivot("month", "year", "passengers")
sns.heatmap(flights, annot=True, cmap="coolwarm")
plt.show()
```

- `annot=True`: Displays numerical values.  
- `cmap`: Sets color scheme (`coolwarm`, `viridis`, etc.).  
- `linewidths`: Adds grid lines between cells.  

---

### Correlation Heatmap  
A **correlation heatmap** shows relationships between numerical variables.  

**Example:**  
```python
tips = sns.load_dataset("tips")
sns.heatmap(tips.corr(), annot=True, cmap="coolwarm")
plt.show()
```

- `tips.corr()`: Computes Pearson correlation between numerical columns.  

---

### Cluster Map (`clustermap`)  
A **cluster map** groups similar data points together using hierarchical clustering.  

**Syntax:**  
```python
sns.clustermap(data, cmap, method, metric)
```

**Example:**  
```python
sns.clustermap(flights, cmap="coolwarm", method="ward")
plt.show()
```

- `method`: Clustering algorithm (`ward`, `single`, `complete`).  
- `metric`: Distance calculation method (`euclidean`, `correlation`).  

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `heatmap` | Visualizes data intensity using colors |
| Correlation heatmap | Displays relationships between numerical variables |
| `clustermap` | Groups similar data points using clustering |

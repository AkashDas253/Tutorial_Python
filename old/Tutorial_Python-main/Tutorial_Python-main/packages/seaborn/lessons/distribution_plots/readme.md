## Distribution Plots in Seaborn  

Distribution plots are used to understand the spread and shape of numerical data. Seaborn provides several functions for visualizing distributions.  

---

### Histogram (`histplot`)  
A **histogram** shows the frequency of data points in bins.  

**Syntax:**  
```python
sns.histplot(data, x, bins, kde, hue)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.histplot(tips["total_bill"], bins=20, kde=True)
plt.show()
```

- `bins`: Number of bins for grouping data.  
- `kde=True`: Adds a kernel density estimate (KDE) curve.  
- `hue`: Colors histogram by category.  

---

### Kernel Density Estimate Plot (`kdeplot`)  
A **KDE plot** estimates the probability density function of a dataset.  

**Syntax:**  
```python
sns.kdeplot(data, x, shade, bw_adjust)
```

**Example:**  
```python
sns.kdeplot(tips["total_bill"], shade=True)
plt.show()
```

- `shade=True`: Fills the area under the curve.  
- `bw_adjust`: Adjusts smoothness of the curve.  

---

### Rug Plot (`rugplot`)  
A **rug plot** shows individual data points along an axis.  

**Syntax:**  
```python
sns.rugplot(data, x)
```

**Example:**  
```python
sns.rugplot(tips["total_bill"])
plt.show()
```

- Useful for **small datasets** to show exact data distribution.  

---

### Empirical Cumulative Distribution Function Plot (`ecdfplot`)  
An **ECDF plot** shows the cumulative proportion of data points.  

**Syntax:**  
```python
sns.ecdfplot(data, x, hue)
```

**Example:**  
```python
sns.ecdfplot(tips["total_bill"])
plt.show()
```

- Useful for comparing distributions.  

---

### Pair Plot (`pairplot`)  
A **pair plot** shows distributions and relationships between numerical variables.  

**Syntax:**  
```python
sns.pairplot(data, hue)
```

**Example:**  
```python
sns.pairplot(tips, hue="sex")
plt.show()
```

- `hue`: Colors scatter plots by category.  

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `histplot` | Shows frequency distribution of data |
| `kdeplot` | Estimates the probability density function |
| `rugplot` | Displays exact data points along an axis |
| `ecdfplot` | Shows cumulative distribution of data |
| `pairplot` | Visualizes pairwise relationships between variables |


## Categorical Plots in Seaborn  

Categorical plots are used to visualize relationships between categorical and numerical data. Seaborn provides several high-level functions for different categorical plot types.  

---

### Strip Plot  
A **strip plot** shows individual data points along a categorical axis.  

**Syntax:**  
```python
sns.stripplot(x, y, data, jitter=True, hue, palette)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.show()
```

- `jitter=True`: Spreads overlapping points for better visibility.  
- `hue`: Colors points based on a categorical variable.  

---

### Swarm Plot  
A **swarm plot** arranges points to avoid overlap while maintaining category separation.  

**Syntax:**  
```python
sns.swarmplot(x, y, data, hue, palette)
```

**Example:**  
```python
sns.swarmplot(x="day", y="total_bill", data=tips)
plt.show()
```

- **Similar to `stripplot` but prevents overlap.**  
- **Avoid using it on large datasets** due to high computational cost.  

---

### Box Plot  
A **box plot** (box-and-whisker plot) shows the distribution, median, quartiles, and outliers.  

**Syntax:**  
```python
sns.boxplot(x, y, data, hue, palette, notch)
```

**Example:**  
```python
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()
```

- `notch=True`: Adds notches to indicate confidence intervals.  
- `hue`: Groups data using color categories.  

---

### Violin Plot  
A **violin plot** combines a box plot and a KDE plot to show data distribution density.  

**Syntax:**  
```python
sns.violinplot(x, y, data, hue, split, scale)
```

**Example:**  
```python
sns.violinplot(x="day", y="total_bill", data=tips)
plt.show()
```

- `split=True`: Splits violins when using `hue`.  
- `scale="count"`: Adjusts width based on sample size.  

---

### Boxen Plot  
A **boxen plot** (extended box plot) is optimized for large datasets with many outliers.  

**Syntax:**  
```python
sns.boxenplot(x, y, data, hue, palette)
```

**Example:**  
```python
sns.boxenplot(x="day", y="total_bill", data=tips)
plt.show()
```

- **Works well with large datasets compared to `boxplot`**.  

---

### Point Plot  
A **point plot** shows mean values of a numerical variable across categories, with error bars.  

**Syntax:**  
```python
sns.pointplot(x, y, data, hue, palette, ci)
```

**Example:**  
```python
sns.pointplot(x="day", y="total_bill", data=tips)
plt.show()
```

- `ci=None`: Removes confidence interval error bars.  

---

### Bar Plot  
A **bar plot** shows mean values with confidence intervals as bars.  

**Syntax:**  
```python
sns.barplot(x, y, data, hue, palette, ci)
```

**Example:**  
```python
sns.barplot(x="day", y="total_bill", data=tips)
plt.show()
```

- Similar to `pointplot`, but **displays bars instead of points**.  

---

### Count Plot  
A **count plot** shows the number of observations in each category.  

**Syntax:**  
```python
sns.countplot(x, data, hue, palette)
```

**Example:**  
```python
sns.countplot(x="day", data=tips)
plt.show()
```

- **Useful for counting categorical occurrences**.  

---

### Summary  
| Plot Type  | Purpose |
|------------|---------|
| `stripplot` | Individual data points (jittered for visibility) |
| `swarmplot` | Individual points arranged to avoid overlap |
| `boxplot` | Box-and-whisker plot for distribution and outliers |
| `violinplot` | KDE + boxplot to show density and distribution |
| `boxenplot` | Extended box plot for large datasets |
| `pointplot` | Mean values with error bars |
| `barplot` | Mean values as bars |
| `countplot` | Frequency of categorical values |

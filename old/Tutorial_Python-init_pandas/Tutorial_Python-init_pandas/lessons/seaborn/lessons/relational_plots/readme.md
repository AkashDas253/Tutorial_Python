## Relational Plots in Seaborn  

Relational plots are used to visualize relationships between numerical variables. Seaborn provides two primary functions for relational plotting:  

- `scatterplot()`: Displays individual data points.  
- `lineplot()`: Connects data points to show trends over time or continuous values.  

---

### Scatter Plot  
A **scatter plot** shows the relationship between two numerical variables.  

**Syntax:**  
```python
sns.scatterplot(x, y, data, hue, style, size)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()
```

- `hue`: Colors points based on a categorical variable.  
- `style`: Uses different markers for categories.  
- `size`: Varies point size based on another variable.  

**Example with Customization:**  
```python
sns.scatterplot(x="total_bill", y="tip", data=tips, hue="day", style="sex", size="size")
plt.show()
```

---

### Line Plot  
A **line plot** shows trends over time or continuous variables.  

**Syntax:**  
```python
sns.lineplot(x, y, data, hue, style, size)
```

**Example:**  
```python
flights = sns.load_dataset("flights")
sns.lineplot(x="year", y="passengers", data=flights)
plt.show()
```

- `hue`: Colors lines based on a categorical variable.  
- `style`: Uses different line styles for categories.  
- `size`: Varies line thickness.  

**Example with Multiple Categories:**  
```python
sns.lineplot(x="month", y="passengers", data=flights, hue="year")
plt.show()
```

---

### Relational Plot (`relplot()`)  
Seaborn’s `relplot()` is a high-level function that creates either scatter or line plots based on the `kind` parameter.  

**Syntax:**  
```python
sns.relplot(x, y, data, kind, hue, col, row)
```

**Example (Scatter Plot):**  
```python
sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter")
plt.show()
```

**Example (Line Plot):**  
```python
sns.relplot(x="year", y="passengers", data=flights, kind="line")
plt.show()
```

- `col`: Creates subplots in columns.  
- `row`: Creates subplots in rows.  

**Example with Multiple Subplots:**  
```python
sns.relplot(x="total_bill", y="tip", data=tips, kind="scatter", col="sex", row="time")
plt.show()
```

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `scatterplot` | Shows individual data points and relationships |
| `lineplot` | Displays trends over time or continuous variables |
| `relplot` | High-level function for scatter and line plots |

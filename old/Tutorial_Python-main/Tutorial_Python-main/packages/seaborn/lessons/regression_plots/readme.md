## Regression Plots in Seaborn  

Regression plots in Seaborn visualize relationships between variables and fit regression models.  

---

### Regression Plot (`regplot`)  
A **regression plot** fits and visualizes a linear regression model between two numerical variables.  

**Syntax:**  
```python
sns.regplot(x, y, data, ci, scatter_kws, line_kws)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips)
plt.show()
```

- `ci`: Confidence interval for regression line (default 95%).  
- `scatter_kws`: Styles scatter points.  
- `line_kws`: Styles regression line.  

**Example with Customization:**  
```python
sns.regplot(x="total_bill", y="tip", data=tips, ci=None, scatter_kws={"color": "blue"}, line_kws={"color": "red"})
plt.show()
```

---

### LM Plot (`lmplot`)  
An **LM plot** (`lmplot`) extends `regplot()` by adding multiple categories and facets.  

**Syntax:**  
```python
sns.lmplot(x, y, data, hue, col, row)
```

**Example:**  
```python
sns.lmplot(x="total_bill", y="tip", data=tips, hue="sex")
plt.show()
```

- `hue`: Colors lines based on categories.  
- `col`: Creates subplots in columns.  
- `row`: Creates subplots in rows.  

**Example with Multiple Facets:**  
```python
sns.lmplot(x="total_bill", y="tip", data=tips, col="day", hue="sex")
plt.show()
```

---

### Residual Plot (`residplot`)  
A **residual plot** (`residplot`) shows the difference between observed and predicted values.  

**Syntax:**  
```python
sns.residplot(x, y, data)
```

**Example:**  
```python
sns.residplot(x="total_bill", y="tip", data=tips)
plt.show()
```

- Helps check if residuals are randomly distributed (good fit).  

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `regplot` | Fits a regression line with a scatter plot |
| `lmplot` | Extends `regplot` with multiple categories and facets |
| `residplot` | Displays residuals to assess model fit |

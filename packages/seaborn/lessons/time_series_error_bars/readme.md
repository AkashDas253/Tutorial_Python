## Time Series and Error Bars in Seaborn  

Seaborn provides tools for visualizing **time series data** and handling **error bars** to display confidence intervals.  

---

### Time Series Plots (`lineplot`)  
A **line plot** (`lineplot`) is used for visualizing time series data.  

**Syntax:**  
```python
sns.lineplot(x, y, data, hue, style, markers)
```

**Example:**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

flights = sns.load_dataset("flights")
sns.lineplot(x="year", y="passengers", data=flights)
plt.show()
```

- `hue`: Colors lines based on category.  
- `style`: Differentiates lines by pattern.  
- `markers`: Adds markers for data points.  

**Example with Multiple Categories:**  
```python
sns.lineplot(x="year", y="passengers", data=flights, hue="month")
plt.show()
```

---

### Error Bars in Seaborn  

#### 1. Confidence Interval in `lineplot`  
Seaborn’s `lineplot()` automatically includes a confidence interval (CI).  

**Example:**  
```python
sns.lineplot(x="year", y="passengers", data=flights, ci=95)
plt.show()
```

- `ci`: Confidence interval (default 95%).  
- `ci=None`: Disables error bars.  

---

#### 2. Error Bars in `barplot`  
`barplot()` includes error bars by default.  

**Syntax:**  
```python
sns.barplot(x, y, data, ci)
```

**Example:**  
```python
tips = sns.load_dataset("tips")
sns.barplot(x="day", y="total_bill", data=tips, ci="sd")
plt.show()
```

- `ci="sd"`: Shows standard deviation.  
- `ci=None`: Disables error bars.  

---

#### 3. Error Bars in `pointplot`  
A **point plot** (`pointplot()`) shows mean values with confidence intervals.  

**Example:**  
```python
sns.pointplot(x="day", y="total_bill", data=tips, capsize=0.2)
plt.show()
```

- `capsize`: Adds caps to error bars.  

---

### Summary  
| Plot Type       | Purpose |
|----------------|---------|
| `lineplot` | Visualizes time series data |
| `barplot` | Displays categorical data with error bars |
| `pointplot` | Shows mean values with confidence intervals |

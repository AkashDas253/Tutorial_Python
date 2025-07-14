## Statistical Functions in Seaborn  

Seaborn provides built-in **statistical functions** to visualize data relationships, distributions, and confidence intervals.  

---

### 1. **Estimating Central Tendency (`barplot`, `pointplot`)**  
Seaborn calculates **mean** and **median** automatically in categorical plots.  

**Example – Mean with Confidence Intervals (`barplot`)**  
```python
import seaborn as sns
import matplotlib.pyplot as plt

tips = sns.load_dataset("tips")
sns.barplot(x="day", y="total_bill", data=tips, ci="sd")  # Shows mean ± standard deviation
plt.show()
```

- `ci="sd"`: Displays standard deviation as error bars.  
- `ci=None`: Disables confidence intervals.  

**Example – Mean with Error Bars (`pointplot`)**  
```python
sns.pointplot(x="day", y="total_bill", data=tips, capsize=0.2)
plt.show()
```

- `capsize=0.2`: Adds error bar caps.  

---

### 2. **Regression Analysis (`regplot`, `lmplot`)**  
Seaborn performs **linear regression** with confidence intervals.  

**Example – Regression with `regplot`**  
```python
sns.regplot(x="total_bill", y="tip", data=tips)
plt.show()
```

- `ci=None`: Removes confidence interval.  

**Example – Regression with `lmplot` (Multiple Categories)**  
```python
sns.lmplot(x="total_bill", y="tip", hue="sex", data=tips)
plt.show()
```

- `hue`: Colors by category.  

---

### 3. **Kernel Density Estimation (`kdeplot`)**  
**Example – KDE Plot**  
```python
sns.kdeplot(x=tips["total_bill"], fill=True)
plt.show()
```

- `fill=True`: Fills the density plot.  

---

### 4. **Box Plots and Percentiles (`boxplot`)**  
Box plots show **median**, **quartiles**, and **outliers**.  

**Example – Box Plot**  
```python
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()
```

---

### Summary  

| Function | Purpose |
|----------|---------|
| `barplot` | Shows mean ± confidence interval |
| `pointplot` | Displays mean with error bars |
| `regplot` | Performs linear regression |
| `lmplot` | Regression with categorical grouping |
| `kdeplot` | Kernel density estimation |
| `boxplot` | Median, quartiles, and outliers |

## Basic Usage in Seaborn  

### Importing Seaborn  
Seaborn is typically used alongside Matplotlib and Pandas. To use Seaborn, it must be imported first:  

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
```

### Setting the Default Theme  
Seaborn provides a default theme for better aesthetics in plots. The default theme can be set using:  

```python
sns.set_theme()
```

This function applies a default style, which can be customized using theme parameters.  

### Loading Built-in Datasets  
Seaborn includes several built-in datasets that can be loaded using the `sns.load_dataset()` function. These datasets are useful for testing and visualization:  

```python
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
```

Each dataset is returned as a Pandas DataFrame, which can be explored using:  

```python
print(tips.head())  # Display first few rows
print(iris.info())  # Show dataset structure
```

### Plotting with Seaborn  
Seaborn provides various high-level functions for visualization. Basic plots can be created using built-in datasets.  

**Example: Scatter Plot using `scatterplot`**  

```python
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()
```

### Customizing Plots  
Seaborn allows customization of plots by modifying elements such as color palettes, themes, and axes labels.  

**Example: Setting a Style**  

```python
sns.set_style("whitegrid")  # Apply grid background
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()
```

**Example: Changing Color Palette**  

```python
sns.set_palette("pastel")
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()
```

### Integration with Matplotlib  
Seaborn works seamlessly with Matplotlib. Plots can be customized using Matplotlib functions.  

**Example: Adding a Title with Matplotlib**  

```python
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.title("Total Bill vs Tip Amount")
plt.show()
```

### Summary  
- **Import Seaborn** before use.  
- **Set themes** using `sns.set_theme()` for better aesthetics.  
- **Load built-in datasets** using `sns.load_dataset()`.  
- **Use high-level functions** like `sns.scatterplot()`.  
- **Customize plots** using styles and palettes.  
- **Combine Seaborn with Matplotlib** for enhanced control.  

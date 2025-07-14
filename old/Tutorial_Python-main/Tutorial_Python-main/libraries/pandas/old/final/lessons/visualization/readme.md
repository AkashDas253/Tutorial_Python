## **Visualization in Pandas**

### **Basic Plotting**  
| Method | Description |  
|--------|-------------|  
| `df.plot()` | Plots the entire DataFrame using its index as the x-axis and all columns as y-axis |  
| `df.plot(x='column', y='value_column')` | Plots specific columns, with one as the x-axis and another as the y-axis |  
| `df.plot(kind='line')` | Plots the data as a line chart (default) |  
| `df.plot(kind='bar')` | Plots the data as a bar chart |  
| `df.plot(kind='hist')` | Plots the data as a histogram |  
| `df.plot(kind='box')` | Plots the data as a box plot (useful for visualizing the distribution) |  
| `df.plot(kind='scatter', x='column', y='value_column')` | Plots a scatter plot between two columns |  
| `df.plot(kind='pie', y='column')` | Plots a pie chart of the given column |  

---

### **Customizing Plots**  
| Method | Description |  
|--------|-------------|  
| `df.plot(title='Title')` | Adds a title to the plot |  
| `df.plot(xlabel='X Axis Label', ylabel='Y Axis Label')` | Adds labels to the x and y axes |  
| `df.plot(grid=True)` | Adds grid lines to the plot |  
| `df.plot(style='--')` | Changes the style of the plot (e.g., dashed line) |  
| `df.plot(color='red')` | Sets the color of the plot |  
| `df.plot(marker='o')` | Adds markers at each data point in the plot |  

---

### **Subplots and Axes**  
| Method | Description |  
|--------|-------------|  
| `df.plot(subplots=True)` | Plots each column in a separate subplot |  
| `df.plot(subplots=True, layout=(2, 2))` | Plots each column in a grid layout (2x2 in this example) |  
| `df.plot(ax=axes)` | Plots on an existing axis object (for custom layout with `matplotlib`) |  
| `fig, ax = plt.subplots()` | Creates a custom plot with Matplotlib axes for more flexibility |  

---

### **Matplotlib Integration**  
| Method | Description |  
|--------|-------------|  
| `import matplotlib.pyplot as plt` | Imports Matplotlib for advanced plotting features |  
| `df.plot(figsize=(10, 6))` | Adjusts the figure size using Matplotlib |  
| `plt.savefig('plot.png')` | Saves the plot as an image file |  
| `plt.show()` | Displays the plot in an interactive window |  
| `plt.subplot()` | Allows further customization of the layout of multiple plots |  

---

### **Advanced Plot Types**  
| Method | Description |  
|--------|-------------|  
| `df.plot(kind='hexbin', gridsize=20)` | Plots a hexbin plot, useful for large data sets with continuous variables |  
| `df.plot(kind='area')` | Plots an area plot, useful for visualizing stacked data |  
| `df.plot(kind='kde')` | Plots a Kernel Density Estimate (KDE) for distribution estimation |  

---

### **Handling Missing Data in Plots**  
| Method | Description |  
|--------|-------------|  
| `df.plot(linewidth=2)` | Plots with thicker lines to help handle sparse or missing data |  
| `df.dropna().plot()` | Drops missing values before plotting |  
| `df.fillna(method='ffill').plot()` | Fills missing values using forward fill before plotting |  

---

### **Seaborn Integration**  
| Method | Description |  
|--------|-------------|  
| `import seaborn as sns` | Imports Seaborn for advanced statistical plots |  
| `sns.lineplot(data=df, x='x_column', y='y_column')` | Plots a line plot using Seaborn |  
| `sns.boxplot(data=df)` | Plots a box plot using Seaborn |  
| `sns.heatmap(df.corr())` | Plots a heatmap of the correlation matrix using Seaborn |  
| `sns.scatterplot(data=df, x='x_column', y='y_column')` | Plots a scatter plot using Seaborn |  

---

### **Visualization with DateTime Index**  
| Method | Description |  
|--------|-------------|  
| `df.plot(x=df.index)` | Plots with DateTime index on the x-axis |  
| `df.resample('M').sum().plot()` | Resamples the data by month and plots the sum of values |  
| `df.plot(kind='line', x_compat=True)` | Plots with DateTime index for compatibility with earlier versions of Pandas |  

---

### **Visualizing Categorical Data**  
| Method | Description |  
|--------|-------------|  
| `df['category'].value_counts().plot(kind='bar')` | Visualizes the frequency distribution of categorical data |  
| `df['category'].value_counts().plot(kind='pie')` | Visualizes the frequency distribution of categorical data in a pie chart |  

---

### **3D Plotting**  
| Method | Description |  
|--------|-------------|  
| `from mpl_toolkits.mplot3d import Axes3D` | Imports 3D plotting tools from Matplotlib |  
| `fig = plt.figure()` | Creates a new figure for 3D plotting |  
| `ax = fig.add_subplot(111, projection='3d')` | Adds a 3D subplot to the figure |  
| `ax.scatter(df['x'], df['y'], df['z'])` | Creates a 3D scatter plot using Matplotlib |  

---

### **Plot Customization with `matplotlib`**  
| Method | Description |  
|--------|-------------|  
| `plt.title('Title')` | Sets the title for the plot |  
| `plt.xlabel('X Axis Label')` | Sets the x-axis label |  
| `plt.ylabel('Y Axis Label')` | Sets the y-axis label |  
| `plt.legend()` | Adds a legend to the plot |  
| `plt.grid(True)` | Enables the grid in the plot |  
| `plt.xlim([xmin, xmax])` | Sets the limits for the x-axis |  
| `plt.ylim([ymin, ymax])` | Sets the limits for the y-axis |  

---

### **Plotting Multiple Figures**  
| Method | Description |  
|--------|-------------|  
| `fig, ax = plt.subplots(nrows=2, ncols=2)` | Creates multiple subplots in a grid |  
| `ax[0, 0].plot(df['column1'])` | Plots on the first subplot |  
| `ax[0, 1].bar(df['category'], df['value'])` | Plots on the second subplot |  

---

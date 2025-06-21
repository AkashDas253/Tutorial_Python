## Seaborn Cheatsheet  

Seaborn is a powerful data visualization library built on top of Matplotlib. This cheatsheet covers the most common plotting functions, customization options, and styling methods in Seaborn.

---

### 1. **Basic Plotting Functions**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.scatterplot`  | Creates a scatter plot.                               | `sns.scatterplot(x="x", y="y", data=df)`                 |
| `sns.lineplot`     | Creates a line plot.                                  | `sns.lineplot(x="x", y="y", data=df)`                    |
| `sns.barplot`      | Creates a bar plot with confidence intervals.         | `sns.barplot(x="category", y="value", data=df)`           |
| `sns.histplot`     | Creates a histogram plot.                             | `sns.histplot(df["column"])`                              |
| `sns.boxplot`      | Creates a box plot.                                   | `sns.boxplot(x="category", y="value", data=df)`           |
| `sns.violinplot`   | Creates a violin plot.                                | `sns.violinplot(x="category", y="value", data=df)`        |
| `sns.kdeplot`      | Creates a kernel density estimate plot.               | `sns.kdeplot(df["column"])`                               |
| `sns.pairplot`     | Creates pairwise scatter plots for all numeric variables. | `sns.pairplot(df)`                                        |
| `sns.heatmap`      | Creates a heatmap for matrix-like data.               | `sns.heatmap(df.corr())`                                  |

---

### 2. **Categorical Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.countplot`    | Counts the occurrences of categorical data.           | `sns.countplot(x="category", data=df)`                    |
| `sns.barplot`      | Creates a bar plot with summary statistics (mean by default). | `sns.barplot(x="category", y="value", data=df)`           |
| `sns.pointplot`    | Creates a point plot showing means and error bars.    | `sns.pointplot(x="category", y="value", data=df)`         |
| `sns.catplot`      | Creates various types of categorical plots in a facet grid. | `sns.catplot(x="category", y="value", data=df, kind="box")` |

---

### 3. **Relational Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.scatterplot`  | Creates a scatter plot (shown above in Basic Plots).  | `sns.scatterplot(x="x", y="y", data=df)`                  |
| `sns.lineplot`     | Creates a line plot (shown above in Basic Plots).     | `sns.lineplot(x="x", y="y", data=df)`                     |

---

### 4. **Regression Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.regplot`      | Creates a scatter plot with a linear regression line. | `sns.regplot(x="x", y="y", data=df)`                      |
| `sns.lmplot`       | Creates a scatter plot with a linear regression line, can handle facets. | `sns.lmplot(x="x", y="y", data=df, hue="category")`       |

---

### 5. **Matrix Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.heatmap`      | Creates a heatmap, typically for correlation matrices. | `sns.heatmap(df.corr(), annot=True)`                       |
| `sns.clustermap`   | Creates a hierarchical clustering heatmap.            | `sns.clustermap(df.corr())`                               |

---

### 6. **Time Series Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.lineplot`     | Can handle time series plots by passing dates to the `x` variable. | `sns.lineplot(x="date", y="value", data=df)`              |
| `sns.relplot`      | Creates relational plots (like `scatterplot`) and supports time series by specifying a `kind`. | `sns.relplot(x="date", y="value", data=df, kind="line")`  |

---

### 7. **FacetGrid & Multi-Plot Grids**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.FacetGrid`    | Allows you to plot multiple subplots based on categorical variables. | `g = sns.FacetGrid(df, col="category", row="type")`        |
| `sns.PairGrid`     | Creates pairwise plots for different variables in a grid. | `g = sns.PairGrid(df)`                                    |
| `sns.catplot`      | Creates categorical plots within facets (shown above). | `sns.catplot(x="category", y="value", data=df)`            |

---

### 8. **Customizing Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.set_theme`    | Sets the overall aesthetic style.                     | `sns.set_theme(style="whitegrid")`                         |
| `sns.set_palette`  | Customizes the color palette.                         | `sns.set_palette("pastel")`                               |
| `sns.set_context`  | Adjusts the scale of elements in the plot.            | `sns.set_context("talk")`                                 |
| `sns.despine`      | Removes the top and right spines (axes).              | `sns.despine()`                                            |
| `sns.set(rc={"figure.figsize": (8, 6)})` | Sets the figure size.                     | `sns.set(rc={"figure.figsize": (10, 6)})`                  |

---

### 9. **Saving Plots**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `plt.savefig`      | Saves the current plot to a file.                     | `plt.savefig("plot.png", dpi=300, bbox_inches="tight")`   |

---

### 10. **Advanced Statistical Functions**

| Function          | Description                                           | Example                                                   |
|-------------------|-------------------------------------------------------|-----------------------------------------------------------|
| `sns.barplot`      | Calculates and plots the mean with confidence intervals. | `sns.barplot(x="category", y="value", data=df, ci="sd")`  |
| `sns.regplot`      | Performs linear regression with confidence intervals. | `sns.regplot(x="x", y="y", data=df)`                      |
| `sns.kdeplot`      | Plots a kernel density estimate.                      | `sns.kdeplot(df["value"], shade=True)`                     |

---

### Summary  
Seaborn integrates easily with **Matplotlib**, **Pandas**, and other libraries for complex visualizations, statistical functions, and customization. This cheatsheet provides an overview of the most common functions and how they can be applied to create various types of plots.
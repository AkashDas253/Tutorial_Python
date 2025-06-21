## Handling Data in Seaborn  

Seaborn provides functions to **load, manipulate, and preprocess** datasets before visualization.  

---

### 1. **Loading Built-in Datasets (`load_dataset`)**  
Seaborn includes built-in datasets that can be loaded easily.  

**Example – Loading a Dataset**  
```python
import seaborn as sns

tips = sns.load_dataset("tips")
print(tips.head())
```

---

### 2. **Working with External Data (`pandas.read_csv`)**  
Seaborn works with Pandas DataFrames, so external CSV files can be loaded.  

**Example – Loading External Data**  
```python
import pandas as pd

df = pd.read_csv("data.csv")
sns.scatterplot(x="column1", y="column2", data=df)
```

---

### 3. **Filtering and Selecting Data**  
Seaborn can visualize filtered subsets of data.  

**Example – Filtering Data**  
```python
filtered_tips = tips[tips["day"] == "Sun"]
sns.boxplot(x="time", y="total_bill", data=filtered_tips)
```

---

### 4. **Handling Missing Data (`dropna`, `fillna`)**  
Missing values can be **removed** or **replaced** before visualization.  

**Example – Removing Missing Data**  
```python
cleaned_data = tips.dropna()
```

**Example – Filling Missing Data**  
```python
tips["tip"] = tips["tip"].fillna(tips["tip"].median())
```

---

### 5. **Grouping and Aggregation (`groupby`)**  
Seaborn works with grouped data for advanced visualizations.  

**Example – Grouping Data**  
```python
import matplotlib.pyplot as plt

grouped_data = tips.groupby("day")["total_bill"].mean()
sns.barplot(x=grouped_data.index, y=grouped_data.values)
plt.show()
```

---

### 6. **Merging and Combining Data (`merge`, `concat`)**  
Multiple datasets can be merged for visualization.  

**Example – Merging Two DataFrames**  
```python
df1 = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
df2 = pd.DataFrame({"A": [1, 2, 3], "C": [7, 8, 9]})

merged_df = pd.merge(df1, df2, on="A")
```

**Example – Concatenating DataFrames**  
```python
combined_df = pd.concat([df1, df2])
```

---

### Summary  

| Function | Purpose |
|----------|---------|
| `load_dataset` | Load built-in Seaborn datasets |
| `pd.read_csv` | Load external datasets |
| `dropna` | Remove missing data |
| `fillna` | Fill missing data with a value |
| `groupby` | Aggregate data for visualization |
| `merge` | Combine datasets based on keys |
| `concat` | Stack datasets together |


## **Data Transformation in Pandas**  
Transform and reshape data for better analysis, reporting, or modeling.

---

### **Apply Functions to Data**

```python
df.apply(np.sqrt)                        # Apply function element-wise to each column
df['col'].apply(lambda x: x * 2)         # Apply custom function to a Series
df.applymap(str)                         # Apply function to every element (DataFrame only)
```

---

### **Mapping and Replacing Values**

```python
df['grade'].map({'A': 4, 'B': 3})        # Replace using mapping
df['col'].replace([1, 2], [10, 20])      # Replace values
```

---

### **Renaming Columns or Index**

```python
df.rename(columns={'old': 'new'})        # Rename columns
df.rename(index={0: 'row1'})             # Rename index
df.columns = df.columns.str.upper()      # Bulk rename using string methods
```

---

### **Sorting**

```python
df.sort_values(by='col')                 # Sort by a column
df.sort_values(by=['col1', 'col2'], ascending=[True, False])  # Multi-column sort
df.sort_index()                          # Sort by index
```

---

### **Changing Data Types**

```python
df['col'].astype('int')                  # Convert column to int
df.astype({'A': 'float', 'B': 'str'})    # Multiple column conversion
```

---

### **Discretization / Binning**

```python
pd.cut(df['col'], bins=3)                        # Bin into 3 equal-width bins
pd.cut(df['col'], bins=[0, 5, 10], labels=['Low', 'High'])  # Custom bins
pd.qcut(df['col'], q=4)                          # Quantile-based binning
```

---

### **One-Hot Encoding (Dummies)**

```python
pd.get_dummies(df['category'])                  # Convert category to dummies
pd.get_dummies(df, columns=['cat_col'])         # Dummy encoding for full DataFrame
```

---

### **Function Pipelines (Chaining)**

```python
(df.assign(new_col = df['col'] * 2)             # Add new column
   .query('new_col > 10')                       # Filter based on it
   .sort_values('new_col')                      # Sort
)
```

---

### **Log / Exp / Power Transformations**

```python
np.log1p(df['col'])                             # Log(x + 1) transformation
np.exp(df['col'])                               # Exponential transformation
df['col'] ** 2                                  # Power transformation
```

---

### **Z-Score Standardization**

```python
(df['col'] - df['col'].mean()) / df['col'].std()   # Z-score normalization
```

---

### **Min-Max Scaling**

```python
(df['col'] - df['col'].min()) / (df['col'].max() - df['col'].min())  # Scale to [0, 1]
```

---

### **Aggregation + Transformation (`groupby`)**

```python
df.groupby('group')['col'].transform('mean')     # Transform with group mean
df.groupby('group').transform(lambda x: x - x.mean())  # Group-wise normalization
```

---

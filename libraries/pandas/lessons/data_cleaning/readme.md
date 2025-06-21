## **Data Cleaning in Pandas**  
A complete guide to handling missing, inconsistent, duplicate, or incorrect data.

---

### **Detecting Missing Data**

```python
df.isna()                      # Detect NaN values (returns boolean DataFrame)
df.isnull()                    # Same as isna()

df.notna()                     # Detect non-NaN values
df.notnull()                   # Same as notna()
```

---

### **Removing Missing Data**

```python
df.dropna()                                # Drop rows with any NaN
df.dropna(axis=1)                          # Drop columns with any NaN
df.dropna(how='all')                       # Drop rows where all values are NaN
df.dropna(subset=['A', 'B'])               # Drop rows where 'A' or 'B' is NaN
df.dropna(thresh=2)                        # Keep rows with at least 2 non-NaNs
```

---

### **Filling Missing Data**

```python
df.fillna(0)                               # Replace all NaNs with 0
df.fillna(method='ffill')                 # Forward fill (propagate previous value)
df.fillna(method='bfill')                 # Backward fill (use next value)
df.fillna(value={'A': 0, 'B': 'NA'})      # Fill different values per column
df.fillna(df.mean())                      # Fill NaNs with column mean
```

---

### **Replacing Values**

```python
df.replace(999, np.nan)                   # Replace 999 with NaN
df.replace([1, 2], [100, 200])            # Replace multiple values
df.replace({'A': {'old': 'new'}})         # Column-specific replacements
```

---

### **Renaming Columns or Index**

```python
df.rename(columns={'old': 'new'})         # Rename column
df.rename(index={0: 'row1'})              # Rename index
df.columns = ['a', 'b', 'c']              # Rename all columns directly
```

---

### **Changing Data Types**

```python
df['col'].astype('int')                   # Convert data type
df.astype({'A': 'int', 'B': 'float'})     # Convert multiple columns
```

---

### **Detecting and Removing Duplicates**

```python
df.duplicated()                           # Returns True for duplicated rows
df.duplicated(subset=['A'])               # Check duplicates based on column 'A'
df.drop_duplicates()                      # Remove duplicated rows
df.drop_duplicates(subset=['A'], keep='last')  # Keep last occurrence
```

---

### **String Cleaning**

```python
df['col'].str.strip()                     # Remove leading/trailing whitespace
df['col'].str.lower()                     # Convert to lowercase
df['col'].str.upper()                     # Convert to uppercase
df['col'].str.replace('$', '', regex=False)  # Replace characters
```

---

### **Outlier Handling (Basic)**

```python
df[df['col'] < df['col'].quantile(0.95)]  # Filter out top 5% values
df.clip(lower=0, upper=100)               # Clip all values within range
```

---

### **Standardizing Column Names**

```python
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')  # Clean column names
```

---

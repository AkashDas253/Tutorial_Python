## **Handling Missing Data in Pandas**  

### **Identifying Missing Data**  
| Method | Description |  
|--------|-------------|  
| `df.isnull()` | Returns a DataFrame of same shape with `True` for missing values and `False` otherwise |  
| `df.notnull()` | Returns `True` for non-missing values and `False` otherwise |  
| `df.isna()` | Alias for `isnull()` |  
| `df.notna()` | Alias for `notnull()` |  
| `df.isnull().sum()` | Returns the count of missing values per column |  

---

### **Removing Missing Data**  
| Method | Description |  
|--------|-------------|  
| `df.dropna()` | Removes rows containing any missing value |  
| `df.dropna(how='all')` | Removes rows only if all values are missing |  
| `df.dropna(axis=1)` | Removes columns with any missing value |  
| `df.dropna(thresh=n)` | Keeps rows with at least `n` non-null values |  

---

### **Filling Missing Data**  
| Method | Description |  
|--------|-------------|  
| `df.fillna(value)` | Replaces missing values with a specific `value` |  
| `df.fillna(method='ffill')` | Forward fills missing values using previous row values |  
| `df.fillna(method='bfill')` | Backward fills missing values using next row values |  
| `df.fillna(df.mean())` | Replaces missing values with column mean |  
| `df.fillna(df.median())` | Replaces missing values with column median |  
| `df.fillna(df.mode().iloc[0])` | Replaces missing values with most frequent value |  

---

### **Replacing Missing Data**  
| Method | Description |  
|--------|-------------|  
| `df.replace(to_replace=np.nan, value=0)` | Replaces NaN values with `0` |  
| `df.replace({np.nan: 'Unknown'})` | Replaces NaN with a custom value |  

---

### **Interpolating Missing Data**  
| Method | Description |  
|--------|-------------|  
| `df.interpolate()` | Fills missing values by interpolating between available data |  
| `df.interpolate(method='linear')` | Linear interpolation |  
| `df.interpolate(method='polynomial', order=2)` | Polynomial interpolation of order 2 |  
| `df.interpolate(method='time')` | Interpolates time-series data |  

---

### **Handling Missing Data in Specific Data Types**  
#### **Numerical Data**  
- Use **mean, median, mode** or **interpolation** to fill missing values.  
- Drop rows if missing values are significant.  

#### **Categorical Data**  
- Use **mode (most frequent value)** to fill missing categories.  
- Replace with **"Unknown"** or a specific label.  

#### **Time-Series Data**  
- Use **forward fill (`ffill`)** or **backward fill (`bfill`)**.  
- Use **interpolation** for smoother transitions.  

---


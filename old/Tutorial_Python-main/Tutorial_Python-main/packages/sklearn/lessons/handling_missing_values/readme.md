## Handling Missing Values in Scikit-Learn  

#### Overview  
Handling missing values is crucial to ensure data consistency and model performance. Scikit-Learn provides different techniques for imputation, including **mean, median, mode, k-nearest neighbors, and iterative approaches**.

---

### **Techniques for Handling Missing Values**  

| Method                         | Type           | Description |
|--------------------------------|--------------|-------------|
| **SimpleImputer**              | Imputation   | Replaces missing values using a constant, mean, median, or mode. |
| **KNNImputer**                 | Imputation   | Uses k-nearest neighbors to impute missing values. |
| **IterativeImputer**           | Imputation   | Predicts missing values using regression models. |
| **Drop Missing Values**        | Removal      | Removes rows or columns with missing values. |
| **Custom Imputation**          | Transformation | Uses domain-specific rules for filling missing values. |

---

### **Syntax with Parameters**  

#### **1. Checking Missing Values**
```python
import pandas as pd

df = pd.read_csv('data.csv')

# Check missing values
print(df.isnull().sum())  # Count missing values per column
print(df.isnull().mean()) # Percentage of missing values
```

#### **2. Removing Missing Values**
```python
df_cleaned = df.dropna(    # Drop rows or columns with NaNs
    axis=0,  # 0 = drop rows, 1 = drop columns
    how='any'  # 'any' = drop if any value is missing, 'all' = drop only if all values are missing
)
```

---

### **Imputation Methods**  

#### **3. Simple Imputer (Mean, Median, Mode, Constant)**
```python
from sklearn.impute import SimpleImputer
import numpy as np

imputer = SimpleImputer(
    missing_values=np.nan, # Define missing value
    strategy='mean',       # Options: 'mean', 'median', 'most_frequent', 'constant'
    fill_value=None        # Value to use if strategy='constant'
)
X_imputed = imputer.fit_transform(X)
```

#### **4. K-Nearest Neighbors Imputer**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5,      # Number of neighbors
    weights='uniform',  # 'uniform' or 'distance'
    metric='nan_euclidean'
)
X_imputed = imputer.fit_transform(X)
```

#### **5. Iterative Imputer (Predictive Imputation)**
```python
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    estimator=None,   # Regression model for imputation (None uses BayesianRidge)
    max_iter=10,      # Maximum iterations
    random_state=None
)
X_imputed = imputer.fit_transform(X)
```

#### **6. Custom Imputation (Filling with Domain-Specific Rules)**
```python
df['column'].fillna(
    value=df['column'].median(),  # Replace with median
    inplace=True
)
```

---

### **Choosing the Right Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| Few missing values | **SimpleImputer (mean/median/mode)** |
| Many missing values | **KNNImputer, IterativeImputer** |
| Data follows normal distribution | **Mean Imputation** |
| Data is skewed | **Median Imputation** |
| Categorical data | **Most Frequent Imputation (Mode)** |
| Missing values depend on other features | **IterativeImputer, KNNImputer** |
| Too many missing values (over 50%) | **Drop Column or Use Domain Knowledge** |

---
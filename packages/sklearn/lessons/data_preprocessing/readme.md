## **Data Preprocessing in Scikit-Learn**

Data preprocessing is essential before model training to handle missing data, scale values, encode categories, create interaction features, and bin or transform inputs.

---

### **1. Missing Value Handling**

| Method           | Class                  | Strategy                          | Use Case                     |
| ---------------- | ---------------------- | --------------------------------- | ---------------------------- |
| Mean/Median/Mode | `SimpleImputer`        | `mean`, `median`, `most_frequent` | Numeric or categorical       |
| KNN Imputation   | `KNNImputer`           | Based on KNN distances            | Small to medium datasets     |
| Drop Rows/Cols   | `dropna()` from Pandas | N/A                               | When missing data is minimal |

#### **Syntax**

```python
# SimpleImputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    strategy='mean'  # Options: 'mean', 'median', 'most_frequent', 'constant'
)
X_imputed = imputer.fit_transform(X)

# KNNImputer
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5,    # Number of neighbors to use
    weights='uniform' # or 'distance'
)
X_imputed = imputer.fit_transform(X)
```

---

### **2. Categorical Encoding**

| Encoder | Class            | Output            | Use Case          |
| ------- | ---------------- | ----------------- | ----------------- |
| Label   | `LabelEncoder`   | Integer labels    | Target variable   |
| One-hot | `OneHotEncoder`  | Binary matrix     | Nominal features  |
| Ordinal | `OrdinalEncoder` | Ordered integers  | Ordinal features  |
| Hashing | `FeatureHasher`  | Fixed-length hash | Large cardinality |

#### **Syntax**

```python
# Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# OneHotEncoder
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_encoded = encoder.fit_transform(X)

# OrdinalEncoder
from sklearn.preprocessing import OrdinalEncoder
encoder = OrdinalEncoder()
X_encoded = encoder.fit_transform(X)
```

---

### **3. Feature Scaling**

| Scaler         | Class            | Robust to Outliers | Range             |
| -------------- | ---------------- | ------------------ | ----------------- |
| StandardScaler | `StandardScaler` | ✖                  | Mean=0, Var=1     |
| MinMaxScaler   | `MinMaxScaler`   | ✖                  | \[0, 1] or custom |
| RobustScaler   | `RobustScaler`   | ✔                  | Scales by IQR     |

#### **Syntax**

```python
# Standard Scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Min-Max Scaler
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Robust Scaler
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)
```

---

### **4. Normalization**

| Normalizer       | Class        | Effect                              |
| ---------------- | ------------ | ----------------------------------- |
| L2 Normalization | `Normalizer` | Transforms each sample to unit norm |

#### **Syntax**

```python
from sklearn.preprocessing import Normalizer
normalizer = Normalizer(norm='l2')
X_normalized = normalizer.fit_transform(X)
```

---

### **5. Binning (Discretization)**

| Method                | Class              | Binning Type                    | Use Case                                     |
| --------------------- | ------------------ | ------------------------------- | -------------------------------------------- |
| Equal-width, quantile | `KBinsDiscretizer` | 'uniform', 'quantile', 'kmeans' | Creating categories from continuous features |

#### **Syntax**

```python
from sklearn.preprocessing import KBinsDiscretizer

binner = KBinsDiscretizer(
    n_bins=5,               # Number of bins
    encode='ordinal',       # or 'onehot'
    strategy='quantile'     # or 'uniform', 'kmeans'
)
X_binned = binner.fit_transform(X)
```

---

### **6. Thresholding (Binarization)**

#### **Syntax**

```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(threshold=0.0)  # Values > threshold become 1
X_binary = binarizer.fit_transform(X)
```

---

### **7. Polynomial and Interaction Features**

| Method     | Class                | Feature Type    | Use Case                        |
| ---------- | -------------------- | --------------- | ------------------------------- |
| Polynomial | `PolynomialFeatures` | x², x₁·x₂, etc. | Capture non-linear interactions |

#### **Syntax**

```python
from sklearn.preprocessing import PolynomialFeatures

# Initialize PolynomialFeatures
poly = PolynomialFeatures(
    degree=2,             # Maximum polynomial degree
    interaction_only=True,  # Only interaction terms
    include_bias=False      # No constant term
)
X_poly = poly.fit_transform(X)
```

---

### **8. Custom Transformations**

| Method          | Class                 | Description                  |
| --------------- | --------------------- | ---------------------------- |
| NumPy Functions | `FunctionTransformer` | Apply custom transformations |

#### **Syntax**

```python
from sklearn.preprocessing import FunctionTransformer
import numpy as np

transformer = FunctionTransformer(func=np.log1p)
X_log = transformer.fit_transform(X)
```

---

### **9. Pipeline Integration**

| Class      | Use                                            |
| ---------- | ---------------------------------------------- |
| `Pipeline` | Sequentially chain transformers and estimators |

#### **Syntax**

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression())
])
pipe.fit(X_train, y_train)
```

---

## **Which Preprocessing Method to Use?**

| Scenario                    | Recommended Technique           |
| --------------------------- | ------------------------------- |
| Missing values (numeric)    | `SimpleImputer` (mean/median)   |
| Missing values (structured) | `KNNImputer`                    |
| Nominal categories          | `OneHotEncoder`                 |
| Ordinal categories          | `OrdinalEncoder`                |
| Outliers                    | `RobustScaler`                  |
| Continuous to categorical   | `KBinsDiscretizer`              |
| Highly skewed features      | `FunctionTransformer(np.log1p)` |
| Feature interactions        | `PolynomialFeatures`            |

---

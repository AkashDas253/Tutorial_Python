## Data Transformation in Scikit-Learn  

#### Overview  
Data transformation techniques in Scikit-Learn modify the features to improve model performance. These include **scaling, normalization, encoding, imputation, and polynomial features**.

---

### Techniques  

| Method                          | Type                  | Description |
|---------------------------------|----------------------|-------------|
| **StandardScaler**              | Scaling             | Standardizes features to zero mean and unit variance. |
| **MinMaxScaler**                | Scaling             | Scales features to a given range (default: [0,1]). |
| **MaxAbsScaler**                | Scaling             | Scales features to [−1,1] based on absolute maximum. |
| **RobustScaler**                | Scaling             | Scales using median and IQR, robust to outliers. |
| **Normalizer**                  | Normalization       | Normalizes samples (rows) to unit norm. |
| **Binarizer**                   | Thresholding        | Converts values to binary (0 or 1) based on a threshold. |
| **QuantileTransformer**         | Nonlinear Scaling   | Transforms data to a uniform or normal distribution. |
| **PowerTransformer**            | Nonlinear Scaling   | Applies power transformations (Box-Cox, Yeo-Johnson) for normality. |
| **OneHotEncoder**               | Encoding           | Converts categorical features to binary vectors. |
| **LabelEncoder**                | Encoding           | Encodes labels as integers. |
| **OrdinalEncoder**              | Encoding           | Encodes categorical features with ordinal relationships. |
| **SimpleImputer**               | Imputation         | Fills missing values with a constant, mean, median, or mode. |
| **KNNImputer**                  | Imputation         | Fills missing values based on k-nearest neighbors. |
| **IterativeImputer**            | Imputation         | Predicts missing values using other features iteratively. |
| **PolynomialFeatures**          | Feature Expansion  | Generates polynomial and interaction terms. |
| **FunctionTransformer**         | Custom Transformation | Applies a user-defined function to transform data. |

---

### **Syntax with Parameters**  

#### **1. Standard Scaling**
```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(
    copy=True,        # Whether to copy the data
    with_mean=True,   # Center data to zero mean
    with_std=True     # Scale to unit variance
)
X_scaled = scaler.fit_transform(X)
```

#### **2. Min-Max Scaling**
```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(
    feature_range=(0, 1), # Desired range
    copy=True             # Whether to copy the data
)
X_scaled = scaler.fit_transform(X)
```

#### **3. Max-Abs Scaling**
```python
from sklearn.preprocessing import MaxAbsScaler

scaler = MaxAbsScaler(copy=True)
X_scaled = scaler.fit_transform(X)
```

#### **4. Robust Scaling (Outlier Resistant)**
```python
from sklearn.preprocessing import RobustScaler

scaler = RobustScaler(
    with_centering=True, # Center using median
    with_scaling=True,   # Scale using IQR
    quantile_range=(25.0, 75.0), # Range for IQR calculation
    copy=True
)
X_scaled = scaler.fit_transform(X)
```

#### **5. Normalization (Row-Wise)**
```python
from sklearn.preprocessing import Normalizer

scaler = Normalizer(
    norm='l2',  # Type of normalization ('l1', 'l2', 'max')
    copy=True
)
X_normalized = scaler.fit_transform(X)
```

#### **6. Binarization**
```python
from sklearn.preprocessing import Binarizer

binarizer = Binarizer(
    threshold=0.5,  # Values above become 1, others become 0
    copy=True
)
X_binarized = binarizer.fit_transform(X)
```

#### **7. Quantile Transformation**
```python
from sklearn.preprocessing import QuantileTransformer

qt = QuantileTransformer(
    n_quantiles=1000, # Number of quantiles to use
    output_distribution='uniform', # 'uniform' or 'normal'
    random_state=None,
    subsample=1e5
)
X_transformed = qt.fit_transform(X)
```

#### **8. Power Transformation (Box-Cox / Yeo-Johnson)**
```python
from sklearn.preprocessing import PowerTransformer

pt = PowerTransformer(
    method='yeo-johnson', # 'yeo-johnson' or 'box-cox' (Box-Cox requires positive values)
    standardize=True,     # Whether to standardize output
    copy=True
)
X_transformed = pt.fit_transform(X)
```

#### **9. One-Hot Encoding**
```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(
    categories='auto',   # Categories can be inferred or manually defined
    drop=None,           # Drop a category (for avoiding multicollinearity)
    sparse_output=False, # Return dense numpy array
    handle_unknown='ignore' # Handle unseen categories
)
X_encoded = encoder.fit_transform(X_categorical)
```

#### **10. Label Encoding**
```python
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)
```

#### **11. Ordinal Encoding**
```python
from sklearn.preprocessing import OrdinalEncoder

encoder = OrdinalEncoder(
    categories=[['low', 'medium', 'high']], # Define category order
    dtype=int
)
X_encoded = encoder.fit_transform(X_categorical)
```

#### **12. Simple Imputation**
```python
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(
    missing_values=np.nan, # Define missing value
    strategy='mean',       # Options: 'mean', 'median', 'most_frequent', 'constant'
    fill_value=None        # Value to use if strategy='constant'
)
X_imputed = imputer.fit_transform(X)
```

#### **13. KNN Imputation**
```python
from sklearn.impute import KNNImputer

imputer = KNNImputer(
    n_neighbors=5,      # Number of neighbors
    weights='uniform',  # 'uniform' or 'distance'
    metric='nan_euclidean'
)
X_imputed = imputer.fit_transform(X)
```

#### **14. Iterative Imputation**
```python
from sklearn.impute import IterativeImputer

imputer = IterativeImputer(
    estimator=None,   # Regression model for imputation (None uses BayesianRidge)
    max_iter=10,      # Maximum iterations
    random_state=None
)
X_imputed = imputer.fit_transform(X)
```

#### **15. Polynomial Feature Expansion**
```python
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(
    degree=2,       # Polynomial degree
    interaction_only=False, # If True, only interaction terms are used
    include_bias=True      # Include bias (constant term)
)
X_poly = poly.fit_transform(X)
```

#### **16. Custom Function Transformation**
```python
from sklearn.preprocessing import FunctionTransformer

transformer = FunctionTransformer(
    func=np.log1p,   # Apply log(1 + x)
    inverse_func=np.expm1, # Inverse transformation
    validate=True,
    check_inverse=True
)
X_transformed = transformer.fit_transform(X)
```

---

### **Choosing the Right Method**  

| Scenario | Recommended Method |
|----------|--------------------|
| Features have different ranges | **StandardScaler, MinMaxScaler** |
| Data has outliers | **RobustScaler** |
| Normalize each row to unit length | **Normalizer** |
| Convert features to binary values | **Binarizer** |
| Convert categorical features to numerical | **OneHotEncoder, OrdinalEncoder, LabelEncoder** |
| Data needs a normal distribution | **PowerTransformer, QuantileTransformer** |
| Handle missing values | **SimpleImputer, KNNImputer, IterativeImputer** |
| Generate polynomial features | **PolynomialFeatures** |
| Apply a custom transformation | **FunctionTransformer** |

---
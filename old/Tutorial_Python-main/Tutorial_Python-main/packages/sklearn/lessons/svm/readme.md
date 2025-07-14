## Support Vector Machine (SVM) in Scikit-Learn  

#### Overview  
Support Vector Machine (SVM) is a supervised learning algorithm used for classification, regression, and outlier detection. Scikit-Learn provides multiple classes for SVM with different functionalities.

---

### Classes in Scikit-Learn for SVM  

| Class | Description |
|---|---|
| `SVC` | Support Vector Classification. |
| `NuSVC` | SVM classification with a different parameterization (`nu`). |
| `LinearSVC` | SVM with a linear kernel optimized for large datasets. |
| `SVR` | Support Vector Regression. |
| `NuSVR` | Regression using `nu` parameter instead of `C`. |
| `LinearSVR` | SVR with a linear kernel optimized for efficiency. |
| `OneClassSVM` | Unsupervised outlier detection using SVM. |

---

### Parameters of SVM Classes  

| Parameter | Description |
|---|---|
| `C` | Regularization parameter (higher = less regularization). |
| `kernel` | Kernel type (`linear`, `poly`, `rbf`, `sigmoid`, or callable). |
| `degree` | Degree of polynomial kernel (used when `kernel='poly'`). |
| `gamma` | Kernel coefficient (`scale`, `auto`, or float value). |
| `coef0` | Independent term in `poly` and `sigmoid` kernels. |
| `shrinking` | Whether to use shrinking heuristic (`True` or `False`). |
| `probability` | Enable probability estimates (`True` or `False`). |
| `tol` | Tolerance for stopping criterion. |
| `cache_size` | Size of the kernel cache (in MB). |
| `class_weight` | Weighting for classes (`balanced` or dict). |
| `verbose` | Verbose output during training. |
| `max_iter` | Maximum number of iterations (-1 for no limit). |
| `decision_function_shape` | `ovr` (one-vs-rest) or `ovo` (one-vs-one). |
| `break_ties` | Enable tie-breaking in classification. |

---

### Methods in SVM Classes  

| Method | Description |
|---|---|
| `fit(X, y)` | Trains the SVM model on input data. |
| `predict(X)` | Predicts labels for classification or values for regression. |
| `predict_proba(X)` | Returns class probability estimates (only when `probability=True`). |
| `decision_function(X)` | Returns confidence scores for predictions. |
| `score(X, y)` | Computes the accuracy or R² score. |
| `support_vectors_` | Returns the support vectors. |
| `dual_coef_` | Coefficients of support vectors in dual form. |
| `intercept_` | Intercept (bias) term of the decision function. |

---

### Functions Related to SVM  

| Function | Description |
|---|---|
| `sklearn.svm.l1_min_c(X, y, loss='squared_hinge')` | Computes the minimum `C` for L1 regularization in `LinearSVC`. |

---

### Attributes of SVM Classes  

| Attribute | Description |
|---|---|
| `support_` | Indices of support vectors. |
| `support_vectors_` | The support vectors. |
| `n_support_` | Number of support vectors per class. |
| `dual_coef_` | Coefficients of support vectors in decision function. |
| `intercept_` | Intercept term in the decision function. |

---
---

## **Comprehensive Note on Support Vector Machines (SVM) in Scikit-Learn**  

### **Overview**  
Support Vector Machines (SVM) are powerful supervised learning algorithms used for classification, regression, and outlier detection. They work by finding the optimal hyperplane that best separates data points in a high-dimensional space.  

---

## **Classes in Scikit-Learn for SVM**  

| **Class** | **Description** |  
|-----------|----------------|  
| `SVC` | Support Vector Classification (SVM for classification). |  
| `SVR` | Support Vector Regression (SVM for regression). |  
| `LinearSVC` | Linear Support Vector Classification (optimized for large datasets). |  
| `LinearSVR` | Linear Support Vector Regression (optimized for large datasets). |  
| `NuSVC` | Nu-Support Vector Classification (alternative to `SVC` with different parameterization). |  
| `NuSVR` | Nu-Support Vector Regression (alternative to `SVR` with different parameterization). |  
| `OneClassSVM` | Unsupervised outlier detection using SVM. |  

---

## **1. Support Vector Classification (SVC)**  
**Usage**: Classifies data by finding the optimal hyperplane that maximizes margin.  

### **Syntax**  
```python
from sklearn.svm import SVC

# Initialize SVM classifier
clf = SVC(
    C=1.0,              # Regularization parameter
    kernel='rbf',       # Kernel type: 'linear', 'poly', 'rbf', 'sigmoid'
    degree=3,           # Degree for polynomial kernel
    gamma='scale',      # Kernel coefficient: 'scale', 'auto', or float
    coef0=0.0,          # Independent term in 'poly' and 'sigmoid' kernels
    probability=False,  # Enable probability estimates
    shrinking=True,     # Use shrinking heuristic
    tol=1e-3,           # Tolerance for stopping criterion
    cache_size=200,     # Size of kernel cache (in MB)
    class_weight=None,  # Handle imbalanced classes
    verbose=False,      # Output logs
    max_iter=-1,        # Max iterations (-1 for no limit)
    decision_function_shape='ovr',  # One-vs-Rest or 'ovo' (One-vs-One)
    random_state=42     # Random seed for reproducibility
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### **Kernel Selection**  
| **Kernel** | **Description** |  
|-----------|----------------|  
| `linear` | Best for linearly separable data. |  
| `poly` | Uses polynomial decision boundaries. |  
| `rbf` | Maps to high-dimensional space for non-linear classification. |  
| `sigmoid` | Similar to a neural network activation function. |  

---

## **2. Support Vector Regression (SVR)**  
**Usage**: Predicts continuous values using SVM.  

### **Syntax**  
```python
from sklearn.svm import SVR

# Initialize SVM regressor
reg = SVR(
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    tol=1e-3,
    epsilon=0.1,  # Defines margin of tolerance
    shrinking=True,
    cache_size=200,
    verbose=False,
    max_iter=-1
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### **Use Case**  
- Works well when the relationship between input and output is highly non-linear.  
- Less sensitive to outliers than standard regression models.  

---

## **3. Linear Support Vector Classification (LinearSVC)**  
**Usage**: Optimized for large datasets using a linear kernel.  

### **Syntax**  
```python
from sklearn.svm import LinearSVC

# Initialize Linear SVM Classifier
clf = LinearSVC(
    C=1.0,
    penalty='l2',  # Regularization ('l1' or 'l2')
    loss='squared_hinge',
    dual=True,     # 'True' for n_samples > n_features
    tol=1e-4,
    max_iter=1000
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### **Use Case**  
- Preferred when dataset is large and `kernel='linear'` is needed.  
- More efficient than `SVC(kernel='linear')`.  

---

## **4. Linear Support Vector Regression (LinearSVR)**  
**Usage**: Optimized for large datasets requiring linear regression.  

### **Syntax**  
```python
from sklearn.svm import LinearSVR

# Initialize Linear SVM Regressor
reg = LinearSVR(
    C=1.0,
    epsilon=0.1,
    tol=1e-4,
    max_iter=1000
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### **Use Case**  
- Faster alternative to `SVR(kernel='linear')`.  
- Used for high-dimensional regression problems.  

---

## **5. Nu-Support Vector Classification (NuSVC)**  
**Usage**: Alternative to `SVC`, where `nu` controls margin and support vectors.  

### **Syntax**  
```python
from sklearn.svm import NuSVC

# Initialize Nu-SVC Classifier
clf = NuSVC(
    nu=0.5,          # Upper bound on training errors
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    shrinking=True,
    probability=False,
    tol=1e-3,
    cache_size=200,
    verbose=False,
    max_iter=-1,
    random_state=42
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### **Use Case**  
- More control over margin than `SVC`.  
- Often performs better in small datasets.  

---

## **6. Nu-Support Vector Regression (NuSVR)**  
**Usage**: Alternative to `SVR`, controlling the number of support vectors.  

### **Syntax**  
```python
from sklearn.svm import NuSVR

# Initialize Nu-SVR Regressor
reg = NuSVR(
    nu=0.5,
    C=1.0,
    kernel='rbf',
    degree=3,
    gamma='scale',
    coef0=0.0,
    tol=1e-3,
    shrinking=True,
    cache_size=200,
    verbose=False,
    max_iter=-1
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### **Use Case**  
- Provides flexibility in controlling support vectors.  
- Suitable for small datasets requiring fine control.  

---

## **7. One-Class SVM (Anomaly Detection)**  
**Usage**: Detects outliers in unsupervised settings.  

### **Syntax**  
```python
from sklearn.svm import OneClassSVM

# Initialize One-Class SVM
model = OneClassSVM(
    kernel='rbf',
    gamma='scale',
    nu=0.1  # Proportion of outliers expected
)

model.fit(X_train)
outliers = model.predict(X_test)  # -1 for anomalies, 1 for normal data
```

### **Use Case**  
- Used for fraud detection, novelty detection, and anomaly detection.  

---

## **Choosing the Right SVM Model**  

| **Scenario** | **Recommended Class** |  
|-------------|-----------------------|  
| Classification with non-linear decision boundary | `SVC` |  
| Regression with non-linear relationships | `SVR` |  
| Large-scale classification | `LinearSVC` |  
| Large-scale regression | `LinearSVR` |  
| Alternative to `SVC` with different margin control | `NuSVC` |  
| Alternative to `SVR` with different support vector control | `NuSVR` |  
| Anomaly detection | `OneClassSVM` |  

---
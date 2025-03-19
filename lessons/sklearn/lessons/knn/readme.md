## K-Nearest Neighbors (KNN) in Scikit-Learn  

#### Overview  
K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm used for classification and regression. Scikit-Learn provides multiple classes and functions to implement KNN efficiently.

---

### Classes in Scikit-Learn for KNN  

| Class | Description |
|---|---|
| `KNeighborsClassifier` | KNN for classification tasks. |
| `KNeighborsRegressor` | KNN for regression tasks. |
| `RadiusNeighborsClassifier` | Classification using neighbors within a fixed radius. |
| `RadiusNeighborsRegressor` | Regression using neighbors within a fixed radius. |
| `NearestNeighbors` | Unsupervised nearest neighbor learning. |

---

### Parameters of KNN Classes  

| Parameter | Description |
|---|---|
| `n_neighbors` | Number of neighbors to consider (default: 5). |
| `weights` | Weight function for neighbors (`uniform`, `distance`, or callable). |
| `algorithm` | Algorithm used to compute neighbors (`auto`, `ball_tree`, `kd_tree`, `brute`). |
| `leaf_size` | Leaf size for BallTree/KDTree (default: 30). |
| `p` | Power parameter for Minkowski distance (1: Manhattan, 2: Euclidean). |
| `metric` | Distance metric (default: `minkowski`). |
| `metric_params` | Additional keyword arguments for metric function. |
| `n_jobs` | Number of parallel jobs (-1 for all cores). |

---

### Methods in KNN Classes  

| Method | Description |
|---|---|
| `fit(X, y)` | Fits the model using training data. |
| `predict(X)` | Predicts class labels or regression values for input data. |
| `predict_proba(X)` | Returns probability estimates for classification. |
| `kneighbors(X, n_neighbors, return_distance)` | Finds nearest neighbors for input data. |
| `kneighbors_graph(X, n_neighbors, mode)` | Computes connectivity graph of neighbors. |
| `score(X, y)` | Returns mean accuracy of classification/regression. |

---

### Functions Related to KNN  

| Function | Description |
|---|---|
| `sklearn.neighbors.kneighbors_graph(X, n_neighbors, mode)` | Computes neighbor graph for input data. |
| `sklearn.neighbors.radius_neighbors_graph(X, radius, mode)` | Computes radius-based neighbor graph. |

---

### Attributes of KNN Classes  

| Attribute | Description |
|---|---|
| `classes_` | Unique class labels in classification. |
| `effective_metric_` | The metric used for distance calculation. |
| `effective_metric_params_` | Parameters for distance metric. |
| `n_features_in_` | Number of features in input data. |
| `outputs_2d_` | Boolean indicating if output is 2D. |

---
---

# **Comprehensive Note on K-Nearest Neighbors (KNN) in Scikit-Learn**  

### **Overview**  
K-Nearest Neighbors (KNN) is a non-parametric, instance-based learning algorithm used for classification and regression. It determines the output based on the majority vote (classification) or average (regression) of the `k` nearest data points.  

---

## **Classes in Scikit-Learn for KNN**  

| **Class** | **Description** |  
|-----------|----------------|  
| `KNeighborsClassifier` | KNN for classification tasks. |  
| `KNeighborsRegressor` | KNN for regression tasks. |  
| `RadiusNeighborsClassifier` | Classification using neighbors within a fixed radius. |  
| `RadiusNeighborsRegressor` | Regression using neighbors within a fixed radius. |  
| `NearestNeighbors` | Unsupervised nearest neighbor learning. |  

---

## **1. K-Nearest Neighbors Classifier**  
**Usage**: Classifies data points based on the majority class among `k` nearest neighbors.  

### **Syntax**  
```python
from sklearn.neighbors import KNeighborsClassifier

# Initialize KNN classifier
clf = KNeighborsClassifier(
    n_neighbors=5,      # Number of neighbors to consider
    weights='uniform',  # Weighting function: 'uniform' or 'distance'
    algorithm='auto',   # Algorithm: 'auto', 'ball_tree', 'kd_tree', 'brute'
    leaf_size=30,       # Leaf size for tree-based methods
    p=2,                # Distance metric: 1 (Manhattan), 2 (Euclidean)
    metric='minkowski', # Distance function used
    n_jobs=-1           # Number of parallel jobs (-1 uses all cores)
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### **Choosing `k`**  
- **Small `k`**: High variance, sensitive to noise.  
- **Large `k`**: Smoother decision boundary but may underfit.  
- **Optimal `k`**: Found using cross-validation.  

---

## **2. K-Nearest Neighbors Regressor**  
**Usage**: Predicts a value based on the average of `k` nearest neighbors.  

### **Syntax**  
```python
from sklearn.neighbors import KNeighborsRegressor

# Initialize KNN regressor
reg = KNeighborsRegressor(
    n_neighbors=5,      
    weights='distance',  # Weighted averaging
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### **Advantages**  
- Simple and easy to interpret.  
- No assumptions about data distribution.  

### **Limitations**  
- Computationally expensive for large datasets.  
- Sensitive to irrelevant features and feature scaling.  

---

## **3. Radius Neighbors Classifier**  
**Usage**: Classifies data based on neighbors within a fixed radius.  

### **Syntax**  
```python
from sklearn.neighbors import RadiusNeighborsClassifier

# Initialize Radius Neighbors Classifier
clf = RadiusNeighborsClassifier(
    radius=1.0,        # Fixed radius for neighbors
    weights='uniform',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski'
)

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

### **Use Case**  
- When data points have varying densities and `k` may not be optimal.  

---

## **4. Radius Neighbors Regressor**  
**Usage**: Predicts based on neighbors within a fixed radius.  

### **Syntax**  
```python
from sklearn.neighbors import RadiusNeighborsRegressor

# Initialize Radius Neighbors Regressor
reg = RadiusNeighborsRegressor(
    radius=1.0,      
    weights='distance',
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski'
)

reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
```

### **Use Case**  
- Works well when a fixed neighborhood is more meaningful than a fixed `k`.  

---

## **5. Nearest Neighbors (Unsupervised Learning)**  
**Usage**: Finds nearest points for a given query without classification/regression.  

### **Syntax**  
```python
from sklearn.neighbors import NearestNeighbors

# Initialize Nearest Neighbors
nn = NearestNeighbors(
    n_neighbors=5,      
    algorithm='auto',
    leaf_size=30,
    p=2,
    metric='minkowski',
    n_jobs=-1
)

nn.fit(X_train)
distances, indices = nn.kneighbors(X_test)
```

### **Use Case**  
- Used in recommendation systems, anomaly detection, and density estimation.  

---

## **Choosing the Right KNN Algorithm**  

| **Scenario** | **Recommended Class** |  
|-------------|-----------------------|  
| Classification problems | `KNeighborsClassifier` |  
| Regression problems | `KNeighborsRegressor` |  
| Fixed radius classification | `RadiusNeighborsClassifier` |  
| Fixed radius regression | `RadiusNeighborsRegressor` |  
| Finding nearest points without prediction | `NearestNeighbors` |  

---
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

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

## Decision Tree in Scikit-Learn  

#### Overview  
A Decision Tree is a non-parametric supervised learning algorithm used for classification and regression. It splits data based on feature values to create a tree-like structure.  

---

### Classes for Decision Trees  

| Class | Type | Description |
|---|---|---|
| `DecisionTreeClassifier` | Classification | Decision tree for classification tasks. |
| `DecisionTreeRegressor` | Regression | Decision tree for regression tasks. |

---

### Parameters of Decision Tree Models  

| Parameter | Description |
|---|---|
| `criterion` | Split quality measure: `'gini'`, `'entropy'` (classification) or `'squared_error'`, `'friedman_mse'`, `'absolute_error'`, `'poisson'` (regression). |
| `splitter` | Splitting strategy: `'best'` (default) or `'random'`. |
| `max_depth` | Maximum tree depth (limits overfitting). |
| `min_samples_split` | Minimum samples to split a node (`default=2`). |
| `min_samples_leaf` | Minimum samples per leaf node (`default=1`). |
| `min_weight_fraction_leaf` | Minimum weighted fraction of total samples per leaf. |
| `max_features` | Number of features to consider at each split. |
| `max_leaf_nodes` | Maximum number of leaf nodes. |
| `min_impurity_decrease` | Minimum impurity reduction for a split to occur. |
| `class_weight` | Weights assigned to classes (classification only). |
| `random_state` | Controls randomness in splits. |

---

### Methods in Decision Tree Models  

| Method | Description |
|---|---|
| `fit(X, y)` | Fits the model to training data. |
| `predict(X)` | Predicts class labels (classification) or values (regression). |
| `score(X, y)` | Returns accuracy (classification) or R² score (regression). |
| `get_params()` | Returns model hyperparameters. |
| `set_params(**params)` | Sets model hyperparameters. |
| `feature_importances_` | Returns importance scores for features. |
| `decision_path(X)` | Returns the tree traversal path for samples. |

---

### Attributes of Decision Tree Models  

| Attribute | Description |
|---|---|
| `tree_` | The underlying tree structure. |
| `feature_importances_` | Importance scores for features. |
| `max_features_` | Number of features considered at each split. |
| `n_classes_` | Number of classes (classification only). |
| `n_outputs_` | Number of outputs (1 for single-output trees). |

---

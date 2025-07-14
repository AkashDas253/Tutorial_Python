## Ensemble Methods
Ensemble methods are techniques that create multiple models and then combine them to produce improved results.

#### Syntax

```python
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

# Random Forest
rf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None)
rf.fit(X_train, y_train)

# Gradient Boosting
gb = GradientBoostingClassifier(loss='deviance', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse', min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3, min_impurity_decrease=0.0, init=None, random_state=None, max_features=None, verbose=0, max_leaf_nodes=None, warm_start=False, validation_fraction=0.1, n_iter_no_change=None, tol=1e-4, ccp_alpha=0.0)
gb.fit(X_train, y_train)

# AdaBoost
ab = AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', random_state=None)
ab.fit(X_train, y_train)
```

#### Parameters
- **`n_estimators`**: int, The number of trees in the forest.
- **`criterion`**: str, The function to measure the quality of a split. {'gini', 'entropy'}
- **`max_depth`**: int, The maximum depth of the tree.
- **`min_samples_split`**: int or float, The minimum number of samples required to split an internal node.
- **`min_samples_leaf`**: int or float, The minimum number of samples required to be at a leaf node.
- **`min_weight_fraction_leaf`**: float, The minimum weighted fraction of the sum total of weights required to be at a leaf node.
- **`max_features`**: int, float, str or None, The number of features to consider when looking for the best split.
- **`max_leaf_nodes`**: int, Grow trees with `max_leaf_nodes` in best-first fashion.
- **`min_impurity_decrease`**: float, A node will be split if this split induces a decrease of the impurity greater than or equal to this value.
- **`bootstrap`**: bool, Whether bootstrap samples are used when building trees.
- **`oob_score`**: bool, Whether to use out-of-bag samples to estimate the generalization accuracy.
- **`n_jobs`**: int, The number of jobs to run in parallel.
- **`random_state`**: int, Controls the randomness of the estimator.
- **`verbose`**: int, Controls the verbosity when fitting and predicting.
- **`warm_start`**: bool, When set to `True`, reuse the solution of the previous call to fit and add more estimators to the ensemble.
- **`class_weight`**: dict, list of dicts, "balanced", "balanced_subsample" or None, Weights associated with classes.
- **`loss`**: str, Loss function to be optimized. {'deviance', 'exponential'}
- **`learning_rate`**: float, Learning rate shrinks the contribution of each tree by `learning_rate`.
- **`subsample`**: float, The fraction of samples to be used for fitting the individual base learners.
- **`init`**: estimator or None, An estimator object that is used to compute the initial predictions.
- **`validation_fraction`**: float, The proportion of training data to set aside as validation set for early stopping.
- **`n_iter_no_change`**: int, Number of iterations with no improvement to wait before stopping fitting.
- **`tol`**: float, Tolerance for the early stopping.
- **`ccp_alpha`**: float, Complexity parameter used for Minimal Cost-Complexity Pruning.
- **`base_estimator`**: estimator or None, The base estimator from which the boosted ensemble is built.
- **`algorithm`**: str, If 'SAMME.R' then use the SAMME.R real boosting algorithm. {'SAMME', 'SAMME.R'}

#### Attributes
- **`estimators_`**: list of classifiers, The collection of fitted sub-estimators.
- **`classes_`**: array of shape (n_classes,), The classes labels.
- **`n_classes_`**: int, The number of classes.
- **`n_features_`**: int, The number of features when `fit` is performed.
- **`feature_importances_`**: array of shape (n_features,), The feature importances.
- **`oob_score_`**: float, Score of the training dataset obtained using an out-of-bag estimate.
- **`oob_decision_function_`**: array of shape (n_samples, n_classes), Decision function computed with out-of-bag estimate.

#### Functions
- **`fit(X, y=None, sample_weight=None)`**: Fit the model according to the given training data.
  - **Parameters**:
    - `X`: array-like of shape (n_samples, n_features), The training input samples.
    - `y`: array-like of shape (n_samples,) or (n_samples, n_outputs), The target values (class labels).
    - `sample_weight`: array-like of shape (n_samples,), Optional, Sample weights.

- **`predict(X)`**: Predict class for X.
  - **Parameters**:
    - `X`: array-like of shape (n_samples, n_features), The input samples.
  - **Returns**:
    - `y`: array of shape (n_samples,), The predicted classes.

- **`predict_proba(X)`**: Predict class probabilities for X.
  - **Parameters**:
    - `X`: array-like of shape (n_samples, n_features), The input samples.
  - **Returns**:
    - `p`: array of shape (n_samples, n_classes), The class probabilities of the input samples.

- **`score(X, y, sample_weight=None)`**: Return the mean accuracy on the given test data and labels.
  - **Parameters**:
    - `X`: array-like of shape (n_samples, n_features), Test samples.
    - `y`: array-like of shape (n_samples,) or (n_samples, n_outputs), True labels for X.
    - `sample_weight`: array-like of shape (n_samples,), Optional, Sample weights.
  - **Returns**:
    - `score`: float, Mean accuracy of self.predict(X) wrt. y.

- **`set_params(**params)`**: Set the parameters of this estimator.
  - **Parameters**:
    - `params`: dict, Estimator parameters.
  - **Returns**:
    - `self`: object, Estimator instance.

- **`get_params(deep=True)`**: Get parameters for this estimator.
  - **Parameters**:
    - `deep`: bool, If True, will return the parameters for this estimator and contained subobjects that are estimators.
  - **Returns**:
    - `params`: dict, Parameter names mapped to their values.


## Supervised Learning

### Classification
Classification is a supervised learning approach where the goal is to predict the categorical class labels of new instances, based on past observations.

#### Syntax

```python
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Logistic Regression
log_reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs', max_iter=100, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
log_reg.fit(X_train, y_train)

# Decision Tree
tree_clf = DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, class_weight=None, ccp_alpha=0.0)
tree_clf.fit(X_train, y_train)

# Random Forest
forest_clf = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)
forest_clf.fit(X_train, y_train)
```

#### Parameters
- **`penalty='l2'`**: Used to specify the norm used in the penalization. {‘l1’, ‘l2’, ‘elasticnet’, ‘none’}
- **`dual=False`**: Dual or primal formulation. {True, False}
- **`tol=0.0001`**: Tolerance for stopping criteria. {float}
- **`C=1.0`**: Inverse of regularization strength. {float}
- **`fit_intercept=True`**: Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function. {True, False}
- **`intercept_scaling=1`**: Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True. {float}
- **`class_weight=None`**: Weights associated with classes in the form {class_label: weight}. {None, ‘balanced’, dict}
- **`random_state=None`**: Controls the randomness of the estimator. {None, integer}
- **`solver='lbfgs'`**: Algorithm to use in the optimization problem. {‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’}
- **`max_iter=100`**: Maximum number of iterations taken for the solvers to converge. {integer}
- **`multi_class='auto'`**: If the option chosen is ‘ovr’, then a binary problem is fit for each label. {‘auto’, ‘ovr’, ‘multinomial’}
- **`verbose=0`**: For the liblinear and lbfgs solvers set verbose to any positive number for verbosity. {integer}
- **`warm_start=False`**: When set to True, reuse the solution of the previous call to fit as initialization. {True, False}
- **`n_jobs=None`**: Number of CPU cores used when parallelizing over classes if multi_class=’ovr’. {None, integer}
- **`l1_ratio=None`**: The Elastic-Net mixing parameter, with 0 <= l1_ratio <= 1. {None, float}

#### Attributes
- **`coef_`**: Coefficient of the features in the decision function.
- **`intercept_`**: Intercept (a.k.a. bias) added to the decision function.
- **`n_iter_`**: Actual number of iterations for all classes.
- **`classes_`**: A list of class labels known to the classifier.
- **`feature_importances_`**: The feature importances (for tree-based classifiers).
- **`n_features_in_`**: Number of features seen during fit.
- **`oob_score_`**: Score of the training dataset obtained using an out-of-bag estimate (for RandomForestClassifier).

#### Functions
- **`fit(X, y)`**: Fit the model according to the given training data.
- **`predict(X)`**: Predict class labels for samples in X.
- **`predict_proba(X)`**: Probability estimates.
- **`decision_function(X)`**: Decision function.
- **`score(X, y)`**: Returns the mean accuracy on the given test data and labels.
- **`get_params(deep=True)`**: Get parameters for this estimator.
- **`set_params(**params)`**: Set the parameters of this estimator.

### Regression
Regression analysis is a set of statistical processes for estimating the relationships among variables.

#### Syntax

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Linear Regression
linear_reg = LinearRegression(fit_intercept=True, normalize='deprecated', copy_X=True, n_jobs=None, positive=False)
linear_reg.fit(X_train, y_train)

# Ridge Regression
ridge_reg = Ridge(alpha=1.0, fit_intercept=True, normalize='deprecated', copy_X=True, max_iter=None, tol=0.001, solver='auto', random_state=None)
ridge_reg.fit(X_train, y_train)

# Lasso Regression
lasso_reg = Lasso(alpha=1.0, fit_intercept=True, normalize='deprecated', precompute=False, copy_X=True, max_iter=1000, tol=0.0001, warm_start=False, positive=False, random_state=None, selection='cyclic')
lasso_reg.fit(X_train, y_train)
```

#### Parameters
- **`fit_intercept=True`**: Whether to calculate the intercept for this model. {True, False}
- **`normalize='deprecated'`**: This parameter is ignored when `fit_intercept` is set to False. {True, False}
- **`copy_X=True`**: If True, X will be copied; else, it may be overwritten. {True, False}
- **`n_jobs=None`**: The number of jobs to use for the computation. {None, integer}
- **`positive=False`**: When set to True, forces the coefficients to be positive. {True, False}
- **`alpha=1.0`**: Regularization strength; must be a positive float. {float}
- **`max_iter=None`**: Maximum number of iterations for conjugate gradient solver. {None, integer}
- **`tol=0.001`**: Precision of the solution. {float}
- **`solver='auto'`**: Solver to use in the computational routines. {‘auto’, ‘svd’, ‘cholesky’, ‘lsqr’, ‘sparse_cg’, ‘sag’, ‘saga’}
- **`precompute=False`**: Whether to use a precomputed Gram matrix to speed up calculations. {True, False, array-like}
- **`warm_start=False`**: When set to True, reuse the solution of the previous call to fit as initialization. {True, False}
- **`selection='cyclic'`**: If set to ‘random’, a random coefficient is updated every iteration rather than looping over features sequentially by default. {‘cyclic’, ‘random’}

#### Attributes
- **`coef_`**: Estimated coefficients for the linear regression problem.
- **`intercept_`**: Independent term in the linear model.
- **`n_iter_`**: Actual number of iterations performed by the solver to reach the specified tolerance.
- **`alpha_`**: The amount of regularization used.
- **`dual_gap_`**: The dual gap at the end of the optimization.
- **`sparse_coef_`**: Sparse representation of the fitted coefficients.

#### Functions
- **`fit(X, y)`**: Fit linear model.
- **`predict(X)`**: Predict using the linear model.
- **`score(X, y)`**: Return the coefficient of determination R^2 of the prediction.
- **`get_params(deep=True)`**: Get parameters for this estimator.
- **`set_params(**params)`**: Set the parameters of this estimator.
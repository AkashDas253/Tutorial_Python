## Classification
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

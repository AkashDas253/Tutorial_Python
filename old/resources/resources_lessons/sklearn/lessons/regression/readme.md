
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
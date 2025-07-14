## Hyperparameter Tuning
Hyperparameter tuning is the process of optimizing the hyperparameters of a machine learning model to improve its performance.

#### Syntax

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Grid Search
grid_search = GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False)
grid_search.fit(X_train, y_train)

# Randomized Search
random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=np.nan, return_train_score=False)
random_search.fit(X_train, y_train)
```

#### Parameters
- **`estimator`**: The object type that implements the "fit" and "predict" methods.
- **`param_grid`**: Dictionary with parameters names (string) as keys and lists of parameter settings to try as values.
- **`param_distributions`**: Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
- **`scoring=None`**: Strategy to evaluate the performance of the cross-validated model on the test set. {None, 'accuracy', 'f1', 'roc_auc', ...}
- **`n_iter=10`**: Number of parameter settings that are sampled (only for RandomizedSearchCV).
- **`n_jobs=None`**: Number of jobs to run in parallel. {None, -1, 1, 2, ...}
- **`iid='deprecated'`**: Deprecated, use `False`. {True, False}
- **`refit=True`**: Refit an estimator using the best found parameters on the whole dataset. {True, False}
- **`cv=None`**: Determines the cross-validation splitting strategy. {None, integer, cross-validation generator}
- **`verbose=0`**: Controls the verbosity. {0, 1, 2, ...}
- **`pre_dispatch='2*n_jobs'`**: Controls the number of jobs that get dispatched during parallel execution. {string, integer}
- **`random_state=None`**: Controls the randomness of the estimator (only for RandomizedSearchCV). {None, integer}
- **`error_score=np.nan`**: Value to assign to the score if an error occurs in estimator fitting. {numeric, 'raise'}
- **`return_train_score=False`**: If `False`, the `cv_results_` attribute will not include training scores. {True, False}

#### Attributes
- **`cv_results_`**: A dictionary with keys as column headers and values as columns, that can be imported into a pandas DataFrame.
- **`best_estimator_`**: Estimator that was chosen by the search, i.e. estimator which gave highest score (or smallest loss if specified) on the left out data.
- **`best_score_`**: Mean cross-validated score of the best_estimator.
- **`best_params_`**: Parameter setting that gave the best results on the hold out data.
- **`best_index_`**: The index (of the `cv_results_` arrays) which corresponds to the best candidate parameter setting.
- **`scorer_`**: Scorer function used on the held out data to choose the best parameters for the model.
- **`n_splits_`**: The number of cross-validation splits (folds/iterations).

#### Functions
- **`fit(X, y=None, **fit_params)`**: Run fit with all sets of parameters.
- **`score(X, y=None)`**: Return the score on the given data, if the estimator has been refit.
- **`predict(X)`**: Call predict on the estimator with the best found parameters.
- **`predict_proba(X)`**: Call predict_proba on the estimator with the best found parameters.
- **`decision_function(X)`**: Call decision_function on the estimator with the best found parameters.
- **`transform(X)`**: Call transform on the estimator with the best found parameters.
- **`inverse_transform(X)`**: Call inverse_transform on the estimator with the best found parameters.
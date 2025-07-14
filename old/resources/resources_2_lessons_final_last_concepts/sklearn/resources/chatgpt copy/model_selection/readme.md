## Model Selection

### Train-Test Split
Train-test split is a technique for evaluating the performance of a machine learning model by splitting the dataset into a training set and a testing set.

#### Syntax

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None, shuffle=True, stratify=None)
```

#### Parameters
- **`test_size=0.25`**: The proportion of the dataset to include in the test split.
- **`random_state=None`**: Controls the shuffling applied to the data before applying the split.
- **`shuffle=True`**: Whether or not to shuffle the data before splitting.
- **`stratify=None`**: If not None, data is split in a stratified fashion, using this as the class labels.


### Cross-Validation
Cross-validation is a technique for assessing how the results of a statistical analysis will generalize to an independent dataset.

#### Syntax

```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(estimator, X, y, scoring=None, cv=None, n_jobs=None, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', error_score=np.nan)
```

#### Parameters
- **`estimator`**: The object to use to fit the data.
- **`X`**: The data to fit.
- **`y`**: The target variable to try to predict.
- **`scoring=None`**: A string or callable to evaluate the predictions on the test set.
- **`cv=None`**: Determines the cross-validation splitting strategy.
- **`n_jobs=None`**: Number of jobs to run in parallel.
- **`verbose=0`**: The verbosity level.
- **`fit_params=None`**: Parameters to pass to the fit method of the estimator.
- **`pre_dispatch='2*n_jobs'`**: Controls the number of jobs that get dispatched during parallel execution.
- **`error_score=np.nan`**: Value to assign to the score if an error occurs in estimator fitting.

#### Attributes
- **`scores`**: Array of scores of the estimator for each run of the cross-validation.

#### Functions
- **`cross_val_score`**: Evaluate a score by cross-validation.

### Grid Search
Grid Search is a technique to perform hyperparameter tuning by exhaustively searching through a specified parameter grid.

#### Syntax

```python
from sklearn.model_selection import GridSearchCV
grid_search = GridSearchCV(estimator, param_grid, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', error_score=np.nan, return_train_score=False)
grid_search.fit(X_train, y_train)
```

#### Parameters
- **`estimator`**: The object type that implements the "fit" and "predict" methods.
- **`param_grid`**: Dictionary or list of dictionaries with parameters names (`str`) as keys and lists of parameter settings to try as values.
- **`scoring=None`**: A single string or a callable to evaluate the predictions on the test set.
- **`n_jobs=None`**: Number of jobs to run in parallel.
- **`iid='deprecated'`**: Deprecated parameter, ignored.
- **`refit=True`**: Refit an estimator using the best found parameters on the whole dataset.
- **`cv=None`**: Determines the cross-validation splitting strategy.
- **`verbose=0`**: Controls the verbosity.
- **`pre_dispatch='2*n_jobs'`**: Controls the number of jobs that get dispatched during parallel execution.
- **`error_score=np.nan`**: Value to assign to the score if an error occurs in estimator fitting.
- **`return_train_score=False`**: If False, the `cv_results_` attribute will not include training scores.

#### Attributes
- **`best_estimator_`**: Estimator that was chosen by the search, i.e., estimator which gave highest score (or smallest loss if specified) on the left out data.
- **`best_score_`**: Mean cross-validated score of the best_estimator.
- **`best_params_`**: Parameter setting that gave the best results on the hold out data.
- **`cv_results_`**: A dictionary with keys as column headers and values as columns, that can be imported into a pandas DataFrame.

#### Functions
- **`fit(X, y=None, **fit_params)`**: Run fit with all sets of parameters.
- **`predict(X)`**: Call predict on the estimator with the best found parameters.
- **`score(X, y=None)`**: Returns the score on the given data, if the estimator has been refit.
- **`get_params(deep=True)`**: Get parameters for this estimator.
- **`set_params(**params)`**: Set the parameters of this estimator.

### Randomized Search
Randomized search is a technique for hyperparameter optimization that samples a fixed number of parameter settings from specified distributions.

#### Syntax

```python
from sklearn.model_selection import RandomizedSearchCV
search = RandomizedSearchCV(estimator, param_distributions, n_iter=10, scoring=None, n_jobs=None, iid='deprecated', refit=True, cv=None, verbose=0, pre_dispatch='2*n_jobs', random_state=None, error_score=np.nan, return_train_score=False)
search.fit(X_train, y_train)
```

#### Parameters
- **`estimator`**: The object type that implements the "fit" and "predict" methods.
- **`param_distributions`**: Dictionary with parameters names (string) as keys and distributions or lists of parameters to try.
- **`n_iter=10`**: Number of parameter settings that are sampled.
- **`scoring=None`**: Strategy to evaluate the performance of the cross-validated model on the test set.
- **`n_jobs=None`**: Number of jobs to run in parallel.
- **`iid='deprecated'`**: Deprecated, use `False`.
- **`refit=True`**: Refit an estimator using the best found parameters on the whole dataset.
- **`cv=None`**: Determines the cross-validation splitting strategy.
- **`verbose=0`**: Controls the verbosity.
- **`pre_dispatch='2*n_jobs'`**: Controls the number of jobs that get dispatched during parallel execution.
- **`random_state=None`**: Controls the randomness of the estimator.
- **`error_score=np.nan`**: Value to assign to the score if an error occurs in estimator fitting.
- **`return_train_score=False`**: If `False`, the `cv_results_` attribute will not include training scores.

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


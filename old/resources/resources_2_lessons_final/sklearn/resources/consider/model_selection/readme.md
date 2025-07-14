## Model Selection

Model selection involves choosing the best model and hyperparameters for a given dataset. 

### 1. Cross-Validation

Cross-validation is a technique for assessing how the results of a statistical analysis will generalize to an independent dataset.

#### Syntax

```python
from sklearn.model_selection import Validation_Strategy_Class, cross_val_score
validation_strategy = Validation_Strategy_Class(n_splits=number_of_splits)
scores = cross_val_score(model, X, y, cv=validation_strategy)
```

#### Parameters
- **`Validation_Strategy_Class`**: The class used for cross-validation strategy. Possible classes:
    - `KFold`: K-Fold Cross-Validation
    - `StratifiedKFold`: Stratified K-Fold Cross-Validation
- **`n_splits`**: The number of folds. This parameter is used to specify how many folds the dataset should be split into.
  - Type: `int`
- **`cv`**: The cross-validation splitting strategy. This parameter is set to the instance of the cross-validation class.
  - Type: `cross-validation generator` or `an iterable`

By using these parameters, you can configure the cross-validation strategy for evaluating your model.

#### Attributes

- **`n_splits_`**: The number of splits (folds/iterations).
  - Type: `int`
- **`split`**: Method to generate indices to split data into training and test set.
  - Type: `generator`

#### Functions

- **`split(X, y=None, groups=None)`**: Generate indices to split data into training and test set.
  - **Parameters**:
    - `X`: Training data.
    - `y` (optional): Target variable.
    - `groups` (optional): Group labels for the samples used while splitting the dataset into train/test set.
- **`get_n_splits(X=None, y=None, groups=None)`**: Returns the number of splitting iterations in the cross-validator.
  - **Parameters**:
    - `X` (optional): Always ignored, exists for compatibility.
    - `y` (optional): Always ignored, exists for compatibility.
    - `groups` (optional): Always ignored, exists for compatibility.

### 2. Grid Search

Grid search is used to find the optimal hyperparameters for a model by exhaustively searching over a specified parameter grid.

#### Syntax

```python
from sklearn.model_selection import GridSearchCV

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=number_of_folds)

# Fit the grid search to the data
grid_search.fit(X, y)

# Best parameters and score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

# Display the results
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)
```

### Parameters

- **`estimator`**: The model or estimator for which you want to tune the hyperparameters.
  - Type: `estimator object`

- **`param_grid`**: The dictionary specifying the parameter grid to search over.
  - Type: `dict` or `list of dictionaries`

- **`cv`**: The cross-validation splitting strategy. This can be an integer specifying the number of folds or an instance of a cross-validation class.
  - Type: `int`, `cross-validation generator`, or `an iterable`

- **`scoring`** (optional): A string or callable to evaluate the predictions on the test set.
  - Type: `str` or `callable`

- **`n_jobs`** (optional): The number of jobs to run in parallel.
  - Type: `int`

- **`pre_dispatch`** (optional): Controls the number of jobs that get dispatched during parallel execution.
  - Type: `int` or `str`

- **`iid`** (optional): If `True`, the data is assumed to be identically distributed across the folds.
  - Type: `bool` (deprecated in version 0.22)

- **`refit`** (optional): Refit an estimator using the best found parameters on the whole dataset.
  - Type: `bool` or `str`

- **`verbose`** (optional): Controls the verbosity.
  - Type: `int`

- **`return_train_score`** (optional): If `False`, the `cv_results_` attribute will not include training scores.
  - Type: `bool`

### Attributes

- **`cv_results_`**: A dictionary containing cross-validation results.
  - Type: `dict`

- **`best_estimator_`**: The estimator that was chosen by the search, i.e., the estimator which gave the highest score (or smallest loss if specified) on the left-out data.
  - Type: `estimator object`

- **`best_score_`**: Mean cross-validated score of the best_estimator.
  - Type: `float`

- **`best_params_`**: Parameter setting that gave the best results on the hold-out data.
  - Type: `dict`

- **`best_index_`**: The index (of the `cv_results_` arrays) which corresponds to the best candidate parameter setting.
  - Type: `int`

- **`scorer_`**: Scorer function used on the held-out data to choose the best parameters for the model.
  - Type: `callable`

- **`n_splits_`**: The number of cross-validation splits (folds/iterations).
  - Type: `int`

- **`refit_time_`**: The time (in seconds) it took to refit the best estimator on the whole dataset.
  - Type: `float`

- **`multimetric_`**: Whether or not the scorers compute several metrics.
  - Type: `bool`

### Functions

- **`fit(X, y=None, groups=None, **fit_params)`**: Run fit with all sets of parameters.
  - **Parameters**:
    - `X`: Training data.
    - `y` (optional): Target variable.
    - `groups` (optional): Group labels for the samples used while splitting the dataset into train/test set.
    - `**fit_params` (optional): Additional parameters passed to the `fit` method of the estimator.

- **`predict(X)`**: Call predict on the estimator with the best found parameters.
  - **Parameters**:
    - `X`: Data to predict on.

- **`score(X, y=None)`**: Returns the score on the given data, if the estimator has been refit.
  - **Parameters**:
    - `X`: Test data.
    - `y` (optional): True labels for `X`.

- **`get_params(deep=True)`**: Get parameters for this estimator.
  - **Parameters**:
    - `deep` (optional): If `True`, will return the parameters for this estimator and contained subobjects that are estimators.

- **`set_params(**params)`**: Set the parameters of this estimator.
  - **Parameters**:
    - `**params`: Estimator parameters.

### 3. Randomized Search 

Randomized search is used to find the optimal hyperparameters for a model by sampling a given number of candidates from a parameter space with a specified distribution.

#### Syntax 

```python
from sklearn.model_selection import RandomizedSearchCV

# Create the RandomizedSearchCV object
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=number_of_iterations, cv=number_of_folds)

# Fit the randomized search to the data
random_search.fit(X, y)

# Best parameters and score
best_params = random_search.best_params_
best_score = random_search.best_score_

# Display the results
print("Best Parameters:", best_params)
print("Best Cross-Validation Score:", best_score)
```

### Parameters

- **`estimator`**: The model or estimator for which you want to tune the hyperparameters.
  - Type: `estimator object`

- **`param_distributions`**: The dictionary specifying the parameter distributions to sample from.
  - Type: `dict` or `list of dictionaries`

- **`n_iter`**: Number of parameter settings that are sampled.
  - Type: `int`

- **`cv`**: The cross-validation splitting strategy. This can be an integer specifying the number of folds or an instance of a cross-validation class.
  - Type: `int`, `cross-validation generator`, or `an iterable`

- **`scoring`** (optional): A string or callable to evaluate the predictions on the test set.
  - Type: `str` or `callable`

- **`n_jobs`** (optional): The number of jobs to run in parallel.
  - Type: `int`

- **`pre_dispatch`** (optional): Controls the number of jobs that get dispatched during parallel execution.
  - Type: `int` or `str`

- **`iid`** (optional): If `True`, the data is assumed to be identically distributed across the folds.
  - Type: `bool` (deprecated in version 0.22)

- **`refit`** (optional): Refit an estimator using the best found parameters on the whole dataset.
  - Type: `bool` or `str`

- **`verbose`** (optional): Controls the verbosity.
  - Type: `int`

- **`return_train_score`** (optional): If `False`, the `cv_results_` attribute will not include training scores.
  - Type: `bool`

### Attributes

- **`cv_results_`**: A dictionary containing cross-validation results.
  - Type: `dict`

- **`best_estimator_`**: The estimator that was chosen by the search, i.e., the estimator which gave the highest score (or smallest loss if specified) on the left-out data.
  - Type: `estimator object`

- **`best_score_`**: Mean cross-validated score of the best_estimator.
  - Type: `float`

- **`best_params_`**: Parameter setting that gave the best results on the hold-out data.
  - Type: `dict`

- **`best_index_`**: The index (of the `cv_results_` arrays) which corresponds to the best candidate parameter setting.
  - Type: `int`

- **`scorer_`**: Scorer function used on the held-out data to choose the best parameters for the model.
  - Type: `callable`

- **`n_splits_`**: The number of cross-validation splits (folds/iterations).
  - Type: `int`

- **`refit_time_`**: The time (in seconds) it took to refit the best estimator on the whole dataset.
  - Type: `float`

- **`multimetric_`**: Whether or not the scorers compute several metrics.
  - Type: `bool`

### Functions

- **`fit(X, y=None, groups=None, **fit_params)`**: Run fit with all sets of parameters.
  - **Parameters**:
    - `X`: Training data.
    - `y` (optional): Target variable.
    - `groups` (optional): Group labels for the samples used while splitting the dataset into train/test set.
    - `**fit_params` (optional): Additional parameters passed to the `fit` method of the estimator.

- **`predict(X)`**: Call predict on the estimator with the best found parameters.
  - **Parameters**:
    - `X`: Data to predict on.

- **`score(X, y=None)`**: Returns the score on the given data, if the estimator has been refit.
  - **Parameters**:
    - `X`: Test data.
    - `y` (optional): True labels for `X`.

- **`get_params(deep=True)`**: Get parameters for this estimator.
  - **Parameters**:
    - `deep` (optional): If `True`, will return the parameters for this estimator and contained subobjects that are estimators.

- **`set_params(**params)`**: Set the parameters of this estimator.
  - **Parameters**:
    - `**params`: Estimator parameters.
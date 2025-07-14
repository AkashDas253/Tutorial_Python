## Scikit-Learn

## Introduction to Scikit-Learn

## Installation

## Basic Concepts
### Estimators
### Transformers
### Pipelines

## Data Preprocessing
### Imputation
### Scaling
### Encoding Categorical Variables


## Model Selection
### Train-Test Split
### Cross-Validation
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Grid Search
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Randomized Search
#### Syntax
#### Parameters
#### Attributes
#### Functions

## Supervised Learning
### Classification
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Regression
#### Syntax
#### Parameters
#### Attributes
#### Functions

## Unsupervised Learning
### Clustering
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Dimensionality Reduction
#### Syntax
#### Parameters
#### Attributes
#### Functions

## Model Evaluation
### Metrics for Classification
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Metrics for Regression
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Confusion Matrix
#### Syntax
#### Parameters
#### Attributes
#### Functions
### ROC Curve
#### Syntax
#### Parameters
#### Attributes
#### Functions

## Advanced Topics
### Ensemble Methods
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Feature Selection
#### Syntax
#### Parameters
#### Attributes
#### Functions
### Hyperparameter Tuning
#### Syntax
#### Parameters
#### Attributes
#### Functions

## Practical Examples
### Example: Classification
### Example: Regression
### Example: Clustering

## Conclusion 

## Basic Concepts

### Estimators
Estimators are objects that learn from data to make predictions or transformations.

#### General Structure

- **Initialization**: Create an instance of the estimator with specific parameters.
- **Fitting**: Train the estimator using the training data.
- **Prediction**: Use the trained estimator to make predictions on new data.

#### Syntax

```python
from sklearn.<module> import <Estimator>
estimator = <Estimator>(param1=value1, param2=value2, ..., paramN=valueN)
estimator.fit(X_train, y_train)
y_pred = estimator.predict(X_test)
```

#### Parameters
- **`param1=value1`**: Description of param1.
- **`param2=value2`**: Description of param2.
- **`paramN=valueN`**: Description of paramN.



### Transformers
Transformers are used to modify or transform data.

#### General Structure

- **Initialization**: Create an instance of the transformer with specific parameters.
- **Fitting**: Learn the transformation from the training data.
- **Transformation**: Apply the learned transformation to new data.

#### Syntax

```python
from sklearn.<module> import <Transformer>
transformer = <Transformer>(param1=value1, param2=value2, ..., paramN=valueN)
transformer.fit(X_train)
X_transformed = transformer.transform(X_test)
```

#### Parameters
- **`param1=value1`**: Description of param1.
- **`param2=value2`**: Description of param2.
- **`paramN=valueN`**: Description of paramN.


#### Types

- **Standard Scaler**: Standardizes features by removing the mean and scaling to unit variance.

  #### Syntax

  ```python
  from sklearn.preprocessing import StandardScaler
  transformer = StandardScaler(with_mean=True, with_std=True, copy=True)
  X_transformed = transformer.fit_transform(X)
  ```

  #### Parameters
  - **`with_mean=True`**: If True, center the data before scaling.
  - **`with_std=True`**: If True, scale the data to unit variance.
  - **`copy=True`**: If True, a copy of X will be created.

- **Min-Max Scaler**: Transforms features by scaling each feature to a given range.

  #### Syntax

  ```python
  from sklearn.preprocessing import MinMaxScaler
  transformer = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
  X_transformed = transformer.fit_transform(X)
  ```

  #### Parameters
  - **`feature_range=(0, 1)`**: Desired range of transformed data.
  - **`copy=True`**: If True, a copy of X will be created.
  - **`clip=False`**: If True, clip the transformed values to the provided feature range.

- **Robust Scaler**: Scales features using statistics that are robust to outliers.

  #### Syntax

  ```python
  from sklearn.preprocessing import RobustScaler
  transformer = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
  X_transformed = transformer.fit_transform(X)
  ```

  #### Parameters
  - **`with_centering=True`**: If True, center the data before scaling.
  - **`with_scaling=True`**: If True, scale the data to interquartile range.
  - **`quantile_range=(25.0, 75.0)`**: Quantile range used to calculate scale.
  - **`copy=True`**: If True, a copy of X will be created.

- **Normalizer**: Scales individual samples to have unit norm.

  #### Syntax

  ```python
  from sklearn.preprocessing import Normalizer
  transformer = Normalizer(norm='l2', copy=True)
  X_transformed = transformer.fit_transform(X)
  ```

  #### Parameters
  - **`norm='l2'`**: Norm to use to normalize each non-zero sample.
  - **`copy=True`**: If True, a copy of X will be created.

### Pipelines
Pipelines are used to assemble several steps that can be cross-validated together while setting different parameters.

#### General Structure

- **Initialization**: Create a pipeline with a sequence of transformers and an estimator.
- **Fitting**: Train the entire pipeline using the training data.
- **Prediction**: Use the trained pipeline to make predictions on new data.

#### Syntax

```python
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('transformer1', <Transformer1>(param1=value1, ...)),
    ('transformer2', <Transformer2>(param1=value1, ...)),
    ('estimator', <Estimator>(param1=value1, ...))
])
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
```

#### Parameters
- **`steps`**: List of (name, transform) tuples (implementing fit/transform) that are chained, in the order in which they are chained.

#### Types

- **Basic Pipeline**: Chains multiple estimators into one.

  #### Syntax

  ```python
  from sklearn.pipeline import Pipeline
  pipeline = Pipeline(steps=[('scaler', StandardScaler()), ('classifier', LogisticRegression())], memory=None, verbose=False)
  pipeline.fit(X_train, y_train)
  ```

  #### Parameters
  - **`steps`**: List of (name, transform) tuples.
  - **`memory=None`**: Used to cache the fitted transformers of the pipeline.
  - **`verbose=False`**: If True, the time elapsed while fitting each step will be printed.

- **FeatureUnion**: Concatenates results of multiple transformer objects.

  #### Syntax

  ```python
  from sklearn.pipeline import FeatureUnion
  from sklearn.decomposition import PCA
  from sklearn.feature_selection import SelectKBest
  combined_features = FeatureUnion(transformer_list=[('pca', PCA(n_components=2)), ('kbest', SelectKBest(k=1))], n_jobs=None, transformer_weights=None, verbose=False)
  combined_features.fit(X, y)
  ```

  #### Parameters
  - **`transformer_list`**: List of (name, transformer) tuples.
  - **`n_jobs=None`**: Number of jobs to run in parallel.
  - **`transformer_weights=None`**: Multiplicative weights for features per transformer.
  - **`verbose=False`**: If True, the time elapsed while fitting each step will be printed.

- **ColumnTransformer**: Applies transformers to columns of an array or pandas DataFrame.

  #### Syntax

  ```python
  from sklearn.compose import ColumnTransformer
  from sklearn.preprocessing import StandardScaler, OneHotEncoder
  column_transformer = ColumnTransformer(transformers=[('num', StandardScaler(), ['age', 'income']), ('cat', OneHotEncoder(), ['gender', 'occupation'])], remainder='drop', sparse_threshold=0.3, n_jobs=None, transformer_weights=None, verbose=False)
  column_transformer.fit(X)
  ```

  #### Parameters
  - **`transformers`**: List of (name, transformer, columns) tuples.
  - **`remainder='drop'`**: Controls the behavior of the remaining columns.
  - **`sparse_threshold=0.3`**: Threshold to decide if the output will be a sparse matrix.
  - **`n_jobs=None`**: Number of jobs to run in parallel.
  - **`transformer_weights=None`**: Multiplicative weights for features per transformer.
  - **`verbose=False`**: If True, the time elapsed while fitting each step will be printed.

## Data Preprocessing

### Imputation
Imputation is the process of replacing missing data with substituted values.

#### Types

- **Simple Imputation**: Replaces missing values with a single value (mean, median, most frequent, or constant).

  #### Syntax

  ```python
  from sklearn.impute import SimpleImputer
  imputer = SimpleImputer(strategy='mean', fill_value=None, verbose=0, copy=True, add_indicator=False)
  X_imputed = imputer.fit_transform(X)
  ```

  #### Parameters
  - **`strategy='mean'`**: The imputation strategy.
  - **`fill_value=None`**: When strategy is 'constant', fill_value is used to replace all occurrences of missing_values.
  - **`verbose=0`**: Controls the verbosity of the imputer.
  - **`copy=True`**: If True, a copy of X will be created.
  - **`add_indicator=False`**: If True, a MissingIndicator transform will stack onto the output of the imputer’s transform.

- **K-Nearest Neighbors Imputation**: Uses the k-nearest neighbors to impute missing values.

  #### Syntax

  ```python
  from sklearn.impute import KNNImputer
  imputer = KNNImputer(n_neighbors=5, weights='uniform', metric='nan_euclidean', copy=True, add_indicator=False)
  X_imputed = imputer.fit_transform(X)
  ```

  #### Parameters
  - **`n_neighbors=5`**: Number of neighboring samples to use for imputation.
  - **`weights='uniform'`**: Weight function used in prediction.
  - **`metric='nan_euclidean'`**: Distance metric for searching neighbors.
  - **`copy=True`**: If True, a copy of X will be created.
  - **`add_indicator=False`**: If True, a MissingIndicator transform will stack onto the output of the imputer’s transform.

- **Multivariate Imputation by Chained Equations (MICE)**: Uses multiple regression models to impute missing values.

  #### Syntax

  ```python
  from sklearn.experimental import enable_iterative_imputer
  from sklearn.impute import IterativeImputer
  imputer = IterativeImputer(max_iter=10, tol=1e-3, n_nearest_features=None, initial_strategy='mean', imputation_order='ascending', skip_complete=False, min_value=-np.inf, max_value=np.inf, verbose=0, random_state=None, add_indicator=False)
  X_imputed = imputer.fit_transform(X)
  ```

  #### Parameters
  - **`max_iter=10`**: Maximum number of imputation iterations.
  - **`tol=1e-3`**: Tolerance to declare convergence.
  - **`n_nearest_features=None`**: Number of nearest features to use.
  - **`initial_strategy='mean'`**: Initial imputation strategy.
  - **`imputation_order='ascending'`**: Order in which to impute.
  - **`skip_complete=False`**: If True, features with no missing values are skipped.
  - **`min_value=-np.inf`**: Minimum possible imputed value.
  - **`max_value=np.inf`**: Maximum possible imputed value.
  - **`verbose=0`**: Controls the verbosity of the imputer.
  - **`random_state=None`**: Seed for random number generator.
  - **`add_indicator=False`**: If True, a MissingIndicator transform will stack onto the output of the imputer’s transform.

### Scaling
Scaling is the process of transforming features to a similar scale.

#### Types

- **Standard Scaling**: Centers the data and scales it to unit variance.

  #### Syntax

  ```python
  from sklearn.preprocessing import StandardScaler
  scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
  X_scaled = scaler.fit_transform(X)
  ```

  #### Parameters
  - **`copy=True`**: If False, try to avoid a copy and do inplace scaling instead.
  - **`with_mean=True`**: If True, center the data before scaling.
  - **`with_std=True`**: If True, scale the data to unit variance.

- **Min-Max Scaling**: Scales the data to a fixed range, typically [0, 1].

  #### Syntax

  ```python
  from sklearn.preprocessing import MinMaxScaler
  scaler = MinMaxScaler(feature_range=(0, 1), copy=True, clip=False)
  X_scaled = scaler.fit_transform(X)
  ```

  #### Parameters
  - **`feature_range=(0, 1)`**: Desired range of transformed data.
  - **`copy=True`**: If False, try to avoid a copy and do inplace scaling instead.
  - **`clip=False`**: If True, clip the transformed values to the provided feature range.

- **Robust Scaling**: Scales the data using statistics that are robust to outliers.

  #### Syntax

  ```python
  from sklearn.preprocessing import RobustScaler
  scaler = RobustScaler(with_centering=True, with_scaling=True, quantile_range=(25.0, 75.0), copy=True)
  X_scaled = scaler.fit_transform(X)
  ```

  #### Parameters
  - **`with_centering=True`**: If True, center the data before scaling.
  - **`with_scaling=True`**: If True, scale the data to interquartile range.
  - **`quantile_range=(25.0, 75.0)`**: Quantile range used to calculate scale.
  - **`copy=True`**: If False, try to avoid a copy and do inplace scaling instead.

### Encoding Categorical Variables
Encoding categorical variables is the process of converting categorical data into numerical format.

#### Types

- **One-Hot Encoding**: Converts categorical values into binary vectors.

  #### Syntax

  ```python
  from sklearn.preprocessing import OneHotEncoder
  encoder = OneHotEncoder(categories='auto', drop=None, sparse_output=True, dtype=np.float64, handle_unknown='error')
  X_encoded = encoder.fit_transform(X)
  ```

  #### Parameters
  - **`categories='auto'`**: Categories (unique values) per feature.
  - **`drop=None`**: Whether to drop one of the categories per feature to avoid collinearity.
  - **`sparse_output=True`**: Will return sparse matrix if set True else will return an array.
  - **`dtype=np.float64`**: Desired dtype of output.
  - **`handle_unknown='error'`**: Whether to raise an error or ignore if an unknown categorical feature is present during transform.

- **Label Encoding**: Converts categorical values into integer labels.

  #### Syntax

  ```python
  from sklearn.preprocessing import LabelEncoder
  encoder = LabelEncoder()
  X_encoded = encoder.fit_transform(X)
  ```

  #### Parameters
  - **No parameters for LabelEncoder**

- **Ordinal Encoding**: Converts categorical values into ordered integer labels.

  #### Syntax

  ```python
  from sklearn.preprocessing import OrdinalEncoder
  encoder = OrdinalEncoder(categories='auto', dtype=np.float64, handle_unknown='error', unknown_value=None)
  X_encoded = encoder.fit_transform(X)
  ```

  #### Parameters
  - **`categories='auto'`**: Categories (unique values) per feature.
  - **`dtype=np.float64`**: Desired dtype of output.
  - **`handle_unknown='error'`**: Whether to raise an error or ignore if an unknown categorical feature is present during transform.
  - **`unknown_value=None`**: When handle_unknown is set to 'use_encoded_value', this parameter is set to the integer value to be used for the unknown category.

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




```query
give these for multiple algorithms: ### Classification
#### Syntax
#### Parameters
#### Attributes
#### Functions current give 1 algorithm and for each parameter if there are multiple possible values give them as passoble value in breces at begining 
```
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

## Unsupervised Learning

### Clustering
Clustering is the task of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups.

#### Syntax

```python
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# KMeans
kmeans = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=1e-4, precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, algorithm='auto')
kmeans.fit(X)

# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5, metric='euclidean', metric_params=None, algorithm='auto', leaf_size=30, p=None, n_jobs=None)
dbscan.fit(X)

# Agglomerative Clustering
agglo = AgglomerativeClustering(n_clusters=2, affinity='euclidean', memory=None, connectivity=None, compute_full_tree='auto', linkage='ward', distance_threshold=None)
agglo.fit(X)
```

#### Parameters
- **`n_clusters=8`**: The number of clusters to form as well as the number of centroids to generate. {integer}
- **`init='k-means++'`**: Method for initialization. {‘k-means++’, ‘random’, ndarray, callable}
- **`n_init=10`**: Number of time the k-means algorithm will be run with different centroid seeds. {integer}
- **`max_iter=300`**: Maximum number of iterations of the k-means algorithm for a single run. {integer}
- **`tol=1e-4`**: Relative tolerance with regards to inertia to declare convergence. {float}
- **`precompute_distances='deprecated'`**: Precompute distances (faster but takes more memory). {‘auto’, True, False}
- **`verbose=0`**: Verbosity mode. {integer}
- **`random_state=None`**: Determines random number generation for centroid initialization. {None, integer}
- **`copy_x=True`**: When pre-computing distances it is more numerically accurate to center the data first. {True, False}
- **`algorithm='auto'`**: K-means algorithm to use. {‘auto’, ‘full’, ‘elkan’}
- **`eps=0.5`**: The maximum distance between two samples for one to be considered as in the neighborhood of the other. {float}
- **`min_samples=5`**: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. {integer}
- **`metric='euclidean'`**: The metric to use when calculating distance between instances in a feature array. {string, callable}
- **`metric_params=None`**: Additional keyword arguments for the metric function. {dict}
- **`leaf_size=30`**: Leaf size passed to BallTree or cKDTree. {integer}
- **`affinity='euclidean'`**: Metric used to compute the linkage. {string, callable}
- **`memory=None`**: Used to cache the output of the computation of the tree. {None, string, object}
- **`connectivity=None`**: Connectivity matrix. {array-like, callable, None}
- **`compute_full_tree='auto'`**: Stop early the construction of the tree at n_clusters. {‘auto’, True, False}
- **`linkage='ward'`**: Which linkage criterion to use. {‘ward’, ‘complete’, ‘average’, ‘single’}
- **`distance_threshold=None`**: The linkage distance threshold above which, clusters will not be merged. {float, None}

#### Attributes
- **`cluster_centers_`**: Coordinates of cluster centers.
- **`labels_`**: Labels of each point.
- **`inertia_`**: Sum of squared distances of samples to their closest cluster center.
- **`n_iter_`**: Number of iterations run.
- **`core_sample_indices_`**: Indices of core samples.
- **`components_`**: Copy of each core sample found by training.
- **`n_features_in_`**: Number of features seen during fit.
- **`feature_names_in_`**: Names of features seen during fit.
- **`n_clusters_`**: The number of clusters found by the algorithm.
- **`n_leaves_`**: Number of leaves in the hierarchical tree.
- **`n_connected_components_`**: The estimated number of connected components in the graph.
- **`children_`**: The children of each non-leaf node.

#### Functions
- **`fit(X, y=None)`**: Compute clustering.
- **`fit_predict(X, y=None)`**: Compute cluster centers and predict cluster index for each sample.
- **`fit_transform(X, y=None)`**: Compute clustering and transform X to cluster-distance space.
- **`predict(X)`**: Predict the closest cluster each sample in X belongs to.
- **`transform(X)`**: Transform X to a cluster-distance space.

### Dimensionality Reduction
Dimensionality reduction is the process of reducing the number of random variables under consideration by obtaining a set of principal variables.

#### Syntax

```python
from sklearn.decomposition import PCA, TruncatedSVD

# PCA
pca = PCA(n_components=None, copy=True, whiten=False, svd_solver='auto', tol=0.0, iterated_power='auto', random_state=None)
pca.fit(X)

# Truncated SVD
svd = TruncatedSVD(n_components=2, algorithm='randomized', n_iter=5, random_state=None, tol=0.0)
svd.fit(X)
```

#### Parameters
- **`n_components=None`**: Number of components to keep. {None, integer, float, 'mle', 'auto'}
- **`copy=True`**: If False, data passed to fit are overwritten and running fit(X). {True, False}
- **`whiten=False`**: When True (False by default) the `components_` vectors are multiplied by the square root of n_samples and then divided by the singular values to ensure uncorrelated outputs with unit component-wise variances. {True, False}
- **`svd_solver='auto'`**: The solver to use. {‘auto’, ‘full’, ‘arpack’, ‘randomized’}
- **`tol=0.0`**: Tolerance for singular values computed by svd_solver == ‘arpack’. {float}
- **`iterated_power='auto'`**: Number of iterations for the power method computed by svd_solver == ‘randomized’. {‘auto’, integer}
- **`random_state=None`**: Used when the ‘arpack’ or ‘randomized’ solvers are used. {None, integer}

#### Attributes
- **`components_`**: Principal axes in feature space, representing the directions of maximum variance in the data.
- **`explained_variance_`**: The amount of variance explained by each of the selected components.
- **`explained_variance_ratio_`**: Percentage of variance explained by each of the selected components.
- **`singular_values_`**: The singular values corresponding to each of the selected components.
- **`mean_`**: Per-feature empirical mean, estimated from the training set.
- **`n_components_`**: The estimated number of components.
- **`n_features_`**: Number of features in the training data.
- **`n_samples_`**: Number of samples in the training data.
- **`noise_variance_`**: The estimated noise covariance following the Probabilistic PCA model.

#### Functions
- **`fit(X, y=None)`**: Fit the model with X.
- **`fit_transform(X, y=None)`**: Fit the model with X and apply the dimensionality reduction on X.
- **`transform(X)`**: Apply dimensionality reduction to X.
- **`inverse_transform(X)`**: Transform data back to its original space.
- **`get_covariance()`**: Compute data covariance with the generative model.
- **`get_precision()`**: Compute data precision matrix with the generative model.
- **`score_samples(X)`**: Return the log-likelihood of each sample.


```query
give these for multiple algorithms: ### Classification
#### Syntax
#### Parameters
#### Attributes
#### Functions current give 1 algorithm and for each parameter if there are multiple possible values give them as passoble value in breces at begining 
```

## Model Evaluation

### Metrics for Classification
Metrics for classification are used to evaluate the performance of classification models.

#### Syntax

```python
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
report = classification_report(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='binary')
precision = precision_score(y_true, y_pred, average='binary')
recall = recall_score(y_true, y_pred, average='binary')
roc_auc = roc_auc_score(y_true, y_score)
```

#### Parameters
- **`y_true`**: Ground truth (correct) target values.
- **`y_pred`**: Estimated targets as returned by a classifier.
- **`average='binary'`**: {None, 'binary', 'micro', 'macro', 'samples', 'weighted'} Determines the type of averaging performed on the data.
- **`y_score`**: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.

#### Attributes
- **`classification_report`**: Build a text report showing the main classification metrics.
- **`accuracy_score`**: Accuracy classification score.
- **`f1_score`**: Compute the F1 score, also known as balanced F-score or F-measure.
- **`precision_score`**: Compute the precision.
- **`recall_score`**: Compute the recall.
- **`roc_auc_score`**: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.

#### Functions
- **`classification_report(y_true, y_pred, target_names=None, sample_weight=None, digits=2, output_dict=False, zero_division='warn')`**: Generate a classification report.
- **`accuracy_score(y_true, y_pred, normalize=True, sample_weight=None)`**: Calculate the accuracy score.
- **`f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`**: Calculate the F1 score.
- **`precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`**: Calculate the precision score.
- **`recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')`**: Calculate the recall score.
- **`roc_auc_score(y_true, y_score, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None)`**: Calculate the ROC AUC score.

### Metrics for Regression
Metrics for regression are used to evaluate the performance of regression models.

#### Syntax

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mse = mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)
mae = mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
r2 = r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')
```

#### Parameters
- **`y_true`**: Ground truth (correct) target values.
- **`y_pred`**: Estimated target values.
- **`sample_weight=None`**: Sample weights. {None, array-like}
- **`multioutput='uniform_average'`**: Defines aggregating of multiple output values. {'raw_values', 'uniform_average', 'variance_weighted'}
- **`squared=True`**: If True returns MSE value, if False returns RMSE value. {True, False} (only for `mean_squared_error`)

#### Attributes
- **`mean_squared_error`**: Mean squared error regression loss.
- **`mean_absolute_error`**: Mean absolute error regression loss.
- **`r2_score`**: R^2 (coefficient of determination) regression score function.

#### Functions
- **`mean_squared_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average', squared=True)`**: Compute mean squared error.
- **`mean_absolute_error(y_true, y_pred, sample_weight=None, multioutput='uniform_average')`**: Compute mean absolute error.
- **`r2_score(y_true, y_pred, sample_weight=None, multioutput='uniform_average')`**: Compute R^2 score.

### Confusion Matrix
A confusion matrix is a table used to evaluate the performance of a classification algorithm by comparing the actual and predicted classifications.

#### Syntax

```python
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)
```

#### Parameters
- **`y_true`**: Ground truth (correct) target values.
- **`y_pred`**: Estimated targets as returned by a classifier.
- **`labels=None`**: List of labels to index the matrix. {None, array-like}
- **`sample_weight=None`**: Sample weights. {None, array-like}
- **`normalize=None`**: Normalizes confusion matrix over the true (rows), predicted (columns) conditions or all the population. {None, 'true', 'pred', 'all'}

#### Attributes
- **`tn`**: True negatives.
- **`fp`**: False positives.
- **`fn`**: False negatives.
- **`tp`**: True positives.

#### Functions
- **`confusion_matrix(y_true, y_pred, labels=None, sample_weight=None, normalize=None)`**: Compute confusion matrix to evaluate the accuracy of a classification.

### ROC Curve
ROC (Receiver Operating Characteristic) curve is a graphical plot that illustrates the diagnostic ability of a binary classifier system as its discrimination threshold is varied.

#### Syntax

```python
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)
```

#### Parameters
- **`y_true`**: True binary labels in range {0, 1} or {-1, 1}.
- **`y_score`**: Target scores, can either be probability estimates of the positive class, confidence values, or binary decisions.
- **`pos_label=None`**: The label of the positive class. {None, int, str}
- **`sample_weight=None`**: Sample weights. {None, array-like}
- **`drop_intermediate=True`**: Whether to drop some suboptimal thresholds which would not appear on a plotted ROC curve. {True, False}

#### Attributes
- **`fpr`**: Array of false positive rates.
- **`tpr`**: Array of true positive rates.
- **`thresholds`**: Array of thresholds used to compute `fpr` and `tpr`.

#### Functions
- **`roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)`**: Compute Receiver operating characteristic (ROC).

## Advanced Topics

### Ensemble Methods
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

### Feature Selection
Feature selection is the process of selecting a subset of relevant features for use in model construction.

#### Syntax

```python
from sklearn.feature_selection import SelectKBest, chi2
selector = SelectKBest(score_func=chi2, k=10)
selector.fit(X_train, y_train)
X_new = selector.transform(X_test)
```

#### Parameters
- **`score_func=chi2`**: Function taking two arrays X and y, and returning a pair of arrays (scores, p-values). {chi2, f_classif, mutual_info_classif, ...} (type: callable)
- **`k=10`**: Number of top features to select. {integer, 'all'} (type: int or str)

#### Attributes
- **`scores_`**: Scores of features. (type: array-like, shape (n_features,))
- **`pvalues_`**: p-values of feature scores. (type: array-like, shape (n_features,))
- **`n_features_in_`**: Number of features seen during fit. (type: int)
- **`feature_names_in_`**: Names of features seen during fit. (type: array-like, shape (n_features,))

#### Functions
- **`fit(X, y=None)`**: Run score function on (X, y) and get the appropriate features.
- **`transform(X)`**: Reduce X to the selected features.
- **`fit_transform(X, y=None)`**: Fit to data, then transform it.
- **`get_support(indices=False)`**: Get a mask, or integer index, of the features selected.
- **`inverse_transform(X)`**: Reverse the transformation operation.

### Hyperparameter Tuning
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

## Practical Examples
### Example: Classification
### Example: Regression
### Example: Clustering

## Conclusion 
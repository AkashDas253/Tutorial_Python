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
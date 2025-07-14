
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

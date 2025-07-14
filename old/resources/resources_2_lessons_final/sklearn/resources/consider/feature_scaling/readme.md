## Feature Scaling

### Feature Scaling with `StandardScaler`

Feature scaling is an essential step in data preprocessing for machine learning. It ensures that all features contribute equally to the model by bringing them to a similar scale. `scikit-learn` provides the `StandardScaler` for standardizing features by removing the mean and scaling to unit variance.

#### Syntax

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler(copy=False, with_mean=False, with_std=True)
X_scaled = scaler.fit_transform(X)
```

#### Parameters

- `copy`: boolean, optional (default=True)
  - If False, try to avoid a copy and do inplace scaling instead.
- `with_mean`: boolean, optional (default=True)
  - If True, center the data before scaling.
- `with_std`: boolean, optional (default=True)
  - If True, scale the data to unit variance (or equivalently, unit standard deviation).

#### Return Type

- `X_scaled`: array-like, shape (n_samples, n_features)
  - The standardized features.

### General Guidelines

1. **Initialize the Scaler**: Create an instance of `StandardScaler`.
2. **Fit and Transform**: Use `fit_transform` on the training data to compute the mean and standard deviation and scale the data.
3. **Transform Test Data**: Use `transform` on the test data to scale it using the same parameters computed from the training data.




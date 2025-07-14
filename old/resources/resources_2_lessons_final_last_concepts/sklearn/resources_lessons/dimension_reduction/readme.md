## Dimensionality Reduction
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

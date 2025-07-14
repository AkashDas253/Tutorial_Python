
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
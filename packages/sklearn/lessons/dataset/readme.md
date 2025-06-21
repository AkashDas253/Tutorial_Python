## Dataset

### Split Dataset using `train_test_split` 

The `train_test_split` function from `sklearn.model_selection` is used to split arrays or matrices into random train and test subsets.

#### Syntax

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, train_size=None, random_state=None, shuffle=True, stratify=None)
```

#### Parameters

- `X`: array-like, shape (n_samples, n_features)
  - The input data to be split.
- `y`: array-like, shape (n_samples,) or (n_samples, n_outputs), optional
  - The target variable to be split.
- `test_size`: float, int, or None, optional (default=None)
  - If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the test split.
  - If int, represents the absolute number of test samples.
  - If None, the value is set to the complement of the train size.
- `train_size`: float, int, or None, optional (default=None)
  - If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the train split.
  - If int, represents the absolute number of train samples.
  - If None, the value is automatically set to the complement of the test size.
- `random_state`: int, RandomState instance, or None, optional (default=None)
  - Controls the shuffling applied to the data before applying the split. Pass an int for reproducible output across multiple function calls.
- `shuffle`: boolean, optional (default=True)
  - Whether or not to shuffle the data before splitting. If `shuffle=False`, `stratify` must be None.
- `stratify`: array-like or None, optional (default=None)
  - If not None, data is split in a stratified fashion, using this as the class labels.

#### Return Type

- `X_train`: array-like, shape (n_train_samples, n_features)
  - The training input samples.
- `X_test`: array-like, shape (n_test_samples, n_features)
  - The testing input samples.
- `y_train`: array-like, shape (n_train_samples,) or (n_train_samples, n_outputs)
  - The training target values.
- `y_test`: array-like, shape (n_test_samples,) or (n_test_samples, n_outputs)
  - The testing target values.

